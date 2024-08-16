import json
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from ._utils import (DictConversion, pad_vocab_size, release_gc,
                     str_dtype_to_torch, torch_to_numpy)
from .layers.linear import ColumnLinear
from .mapping import Mapping
from .models.convert_utils import (get_model_path, load_state_dict,
                                   split_matrix_tp)


def get_all_nemo_lora_weights(lora_weights):
    layer_weights = defaultdict(dict)
    adapter_key = "self_attention.adapter_layer.lora_kqv_adapter"
    layer_pattern = re.compile(r'.*\.layers\.(\d+)\..*')
    for key, weights in lora_weights.items():
        if adapter_key in key:
            if key.endswith('linear_in.weight'):
                inout = 'in'
            elif key.endswith('linear_out.weight'):
                inout = 'out'
            else:
                continue
            m = layer_pattern.match(key)
            layer_idx = int(m.group(1))
            layer_weights[layer_idx][inout] = weights
        else:
            raise KeyError(f"unsupported key {key} from Nemo LoRA weights")
    return layer_weights


# The pattern is {layer_prefix:1}.{layer_idx:2}.{module_prefix:3}.{module_name or {expert_name:5}.{expert_idx:6}.{module_name:7} :4}.lora_{A|B:8}.weight
HF_LORA_PATTERN = re.compile(
    r'(.*)\.(\d+)\.(\w+)\.(\w+|\w+\.\w+|(\w+)\.(\d+)\.(\w+))\.lora_(A|B)\.weight'
)


def iterate_hf_lora(iter_fn, lora_weights, hf_modules, component=None):
    all_weights = defaultdict(lambda: defaultdict(dict))
    pattern = HF_LORA_PATTERN
    for key, weights in lora_weights.items():
        m = pattern.match(key)
        if not m:
            if "lm_head" not in key and "embed_tokens" not in key:
                raise KeyError(f"unsupported key {key} from HF LoRA weights")
            continue
        if component is not None and component not in m.group(1):
            continue
        layer_idx = int(m.group(2))
        expert_idx = m.group(6)
        is_moe = expert_idx is not None
        if is_moe:
            expert_name = m.group(5)
            module_name = m.group(7)
            hf_module = m.group(3) + "." + expert_name + "." + module_name
        else:
            module_name = m.group(4)
            hf_module = m.group(3) + "." + module_name
        if hf_module not in hf_modules:
            hf_module = module_name
            assert hf_module in hf_modules
        inout = "in" if m.group(8) == "A" else "out"
        iter_fn(layer_idx, hf_module, expert_idx, inout, weights)
        if not is_moe:
            all_weights[layer_idx][hf_module][inout] = weights
        else:
            all_weights[layer_idx][hf_module].setdefault(expert_idx, {})
            all_weights[layer_idx][hf_module][expert_idx][inout] = weights
    return all_weights


def get_all_hf_lora_weights(lora_weights, hf_modules, component=None):

    def iter_fn(layer_idx, hf_module, expert_idx, inout, weights):
        if expert_idx is None:
            all_weights[layer_idx][hf_module][inout] = weights
        else:
            all_weights[layer_idx][hf_module].setdefault(expert_idx, {})
            all_weights[layer_idx][hf_module][expert_idx][inout] = weights

    all_weights = defaultdict(lambda: defaultdict(dict))
    iterate_hf_lora(iter_fn, lora_weights, hf_modules, component)
    return all_weights


def get_hf_target_modules(lora_weights, hf_modules):

    def iter_fn(layer_idx, hf_module, expert_idx, inout, weights):
        hf_target_modules.add(hf_module)

    hf_target_modules = set()
    iterate_hf_lora(iter_fn, lora_weights, hf_modules)
    return hf_target_modules


def invert_module_mapping(trtllm_modules_to_hf_modules):
    hf_modules_to_trtllm_modules = {}
    for k, hf_modules in trtllm_modules_to_hf_modules.items():
        if isinstance(hf_modules, list):
            for hf_module in hf_modules:
                hf_modules_to_trtllm_modules[hf_module] = k
        else:
            hf_modules_to_trtllm_modules[hf_modules] = k
    return hf_modules_to_trtllm_modules


@dataclass
class LoraConfig(DictConversion):
    lora_dir: List[str] = field(default_factory=list)
    lora_ckpt_source: str = 'hf'
    max_lora_rank: int = 64
    lora_target_modules: List[str] = field(default_factory=list)
    trtllm_modules_to_hf_modules: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        assert self.lora_ckpt_source in [
            'hf', 'nemo'
        ], f"lora_ckpt_source must be one of 'hf' or 'nemo', got {self.lora_ckpt_source}"


class HfLoraLoader:

    def __init__(self, lora_dirs: List[str]):
        self.lora_target_modules = []
        self.is_valid = False
        self.lm_head = None
        self.embed_tokens = None
        self.vocab_size = 0

        if len(lora_dirs) == 0:
            return

        for lora_dir in lora_dirs:
            model_path = get_model_path(lora_dir, "adapter_model")
            if model_path is None:
                raise ValueError(
                    f"adapter_model file does not exist in {lora_dir}")
            config_file = Path(f"{lora_dir}/adapter_config.json")
            if not config_file.exists():
                raise ValueError(f"{config_file} does not exist")
            if not config_file.is_file():
                raise ValueError(f"{config_file} is not a file")
        self.is_valid = True

        lora_dir = lora_dirs[0]
        with open(f"{lora_dir}/adapter_config.json") as f:
            adapter_config = json.load(f)

        lora_weight = load_state_dict(get_model_path(lora_dir, "adapter_model"))
        self.lora_weight = lora_weight
        if adapter_config["modules_to_save"] is not None:
            if "lm_head" in adapter_config["modules_to_save"]:
                self.lm_head = lora_weight["base_model.model.lm_head.weight"]
                self.vocab_size = self.lm_head.shape[0]

            if "embed_tokens" in adapter_config["modules_to_save"]:
                self.embed_tokens = lora_weight[
                    "base_model.model.model.embed_tokens.weight"]

    def get_target_modules(self, trtllm_modules_to_hf_modules):
        hf_modules_to_trtllm_modules = invert_module_mapping(
            trtllm_modules_to_hf_modules)
        lora_target_modules = set()
        if self.is_valid:
            hf_target_modules = get_hf_target_modules(
                self.lora_weight,
                hf_modules=set(hf_modules_to_trtllm_modules.keys()),
            )
            for m in hf_target_modules:
                trtllm_module = hf_modules_to_trtllm_modules[m]
                lora_target_modules.add(trtllm_module)
        return list(lora_target_modules)


class NemoLoraLoader:

    def __init__(self, lora_dirs: List[str]):
        self.lora_target_modules = []
        self.is_valid = False

        if len(lora_dirs) == 0:
            return

        for lora_file in lora_dirs:
            path = Path(lora_file)
            if not path.exists():
                raise ValueError(f"{path} does not exist")
            if not path.is_file():
                raise ValueError(f"{path} is not a file")
        self.is_valid = True
        # Hardcoded since LoraManager only supports this case now
        self.lora_target_modules = ["attn_qkv"]


def load_nemo_lora(model, lora_config: LoraConfig):
    lora_loader = NemoLoraLoader(lora_config.lora_dir)
    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.lora_target_modules


def get_default_trtllm_modules_to_hf_modules():
    return {
        "attn_q": "q_proj",
        "attn_k": "k_proj",
        "attn_v": "v_proj",
        "attn_dense": "o_proj",
        "mlp_h_to_4h": "gate_proj",
        "mlp_4h_to_h": "down_proj",
        "mlp_gate": "up_proj",
        "moe_h_to_4h": "w1",
        "moe_4h_to_h": "w2",
        "moe_gate": "w3",
        "moe_router": "gate",
    }


def load_hf_lora(
    model,
    lora_config: LoraConfig,
    trtllm_modules_to_hf_modules: Dict[str, str] = None,
):
    trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules or get_default_trtllm_modules_to_hf_modules(
    )
    lora_config.trtllm_modules_to_hf_modules = trtllm_modules_to_hf_modules

    lora_loader = HfLoraLoader(lora_config.lora_dir)

    if len(lora_config.lora_target_modules) == 0:
        lora_config.lora_target_modules = lora_loader.get_target_modules(
            trtllm_modules_to_hf_modules)

    if lora_loader.is_valid:
        config = model.config
        torch_dtype = str_dtype_to_torch(config.dtype)
        # the lora checkpoint might finetune the embedding
        if lora_loader.vocab_size != 0:
            config.vocab_size = lora_loader.vocab_size
        mapping = config.mapping
        if mapping.is_first_pp_rank() and lora_loader.embed_tokens is not None:
            weight = lora_loader.embed_tokens
            if config.use_parallel_embedding:
                weight = split_matrix_tp(
                    weight,
                    mapping.tp_size,
                    mapping.tp_rank,
                    dim=config.embedding_sharding_dim,
                )
            if model.transformer.vocab_embedding.weight.raw_value.shape != weight.shape:
                model.transformer.vocab_embedding = model.transformer.vocab_embedding.__class__(
                    num_embeddings=config.vocab_size,
                    embedding_dim=config.hidden_size,
                    dtype=config.dtype,
                    tp_size=mapping.tp_size
                    if config.use_parallel_embedding else 1,
                    tp_group=mapping.tp_group
                    if config.use_parallel_embedding else None,
                    sharding_dim=config.embedding_sharding_dim,
                    tp_rank=mapping.tp_rank,
                )
            model.transformer.vocab_embedding.weight.value = weight.to(
                torch_dtype)
        if mapping.is_last_pp_rank() and lora_loader.lm_head is not None:
            weight = lora_loader.lm_head
            vocab_size = lora_loader.vocab_size
            if vocab_size % mapping.tp_size != 0:
                # padding
                vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
                pad_width = vocab_size_padded - vocab_size

                weight = torch.from_numpy(
                    np.pad(torch_to_numpy(weight), ((0, pad_width), (0, 0)),
                           'constant',
                           constant_values=0))
            else:
                vocab_size_padded = vocab_size
            if model.lm_head.weight.raw_value.shape != weight.shape:
                model.lm_head = ColumnLinear(
                    config.hidden_size,
                    vocab_size_padded,
                    bias=False,
                    dtype=config.dtype,
                    tp_group=mapping.tp_group,
                    tp_size=mapping.tp_size,
                    gather_output=True,
                )
            model.lm_head.weight.value = split_matrix_tp(
                weight,
                mapping.tp_size,
                mapping.tp_rank,
                dim=0,
            ).to(torch_dtype)


def use_lora(
    model,
    lora_config: LoraConfig,
    trtllm_modules_to_hf_modules: Dict[str, str] = None,
):
    if lora_config.lora_ckpt_source == "nemo":
        load_nemo_lora(model, lora_config)
    elif lora_config.lora_ckpt_source == "hf":
        load_hf_lora(model, lora_config, trtllm_modules_to_hf_modules)
    else:
        raise ValueError(
            f"Unsupported lora_ckpt_source: {lora_config.lora_ckpt_source}")


def unpack_nemo_weights(nemo_archive_path):
    with tarfile.open(nemo_archive_path) as tar:
        try:
            model_weights = tar.extractfile("model_weights.ckpt")
            model_config = tar.extractfile("model_config.yaml")
        except KeyError:
            try:
                model_weights = tar.extractfile("./model_weights.ckpt")
                model_config = tar.extractfile("./model_config.yaml")
            except KeyError:
                err_str = "Both model_weights paths not found in the tar archive."
                raise Exception(err_str)
        return yaml.safe_load(model_config), torch.load(
            model_weights, map_location=torch.device("cpu"))


class LoraManager(object):
    LORA_MODULE_IDS = {
        "attn_qkv": 0,
        "attn_q": 1,
        "attn_k": 2,
        "attn_v": 3,
        "attn_dense": 4,
        "mlp_h_to_4h": 5,
        "mlp_4h_to_h": 6,
        "mlp_gate": 7,
        "cross_attn_qkv": 8,
        "cross_attn_q": 9,
        "cross_attn_k": 10,
        "cross_attn_v": 11,
        "cross_attn_dense": 12,
        "moe_h_to_4h": 13,
        "moe_4h_to_h": 14,
        "moe_gate": 15,
        "moe_router": 16,
        "mlp_router": 17,
    }

    def __init__(self):
        '''
        _lora_uid_to_low_ranks: dict[str -> dict[int -> dict[str -> int]]]
        {
            uid: {
                0: {
                    lora_module: int
                }, # layer_0_rank,
                1: {
                    lora_module: int
                }, # layer_1_rank,
                ...
            }
        }

        _lora_weights_pointers_list: dict[str -> dict[int -> dict[str -> [Tensor, Tensor]]]]
        {
            uid: {
                0: {
                    lora_module: [t_in, t_out]
                }, # layer_0,
                1: {
                    lora_module: [t_in, t_out]
                }, # layer_1,
                ...
            }
        }

        '''
        self._lora_uid_to_low_ranks = {}
        self._lora_weights = []
        self._lora_weights_pointers_list = {}
        self._lora_cpp_weights = {}
        self._lora_weight_config = {}
        self.missing_qkv_modules = []
        self.lora_target_modules = []

    @staticmethod
    def get_missing_qkv_modules(lora_target_modules):
        # In current design, q_lora_params, k_lora_params and v_lora_params should be all enabled or all disabled at the same time.
        # However, some lora checkpoint (e.g. BART) only contain two of them, so we use zero tensor to fill the missing ones.
        missing_qkv_modules = []
        if any(x in lora_target_modules
               for x in ["attn_q", "attn_k", "attn_v"]):
            for lora_module in ["attn_q", "attn_k", "attn_v"]:
                if lora_module not in lora_target_modules:
                    missing_qkv_modules.append(lora_module)
        if any(x in lora_target_modules
               for x in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]):
            for lora_module in ["cross_attn_q", "cross_attn_k", "cross_attn_v"]:
                if lora_module not in lora_target_modules:
                    missing_qkv_modules.append(lora_module)
        return missing_qkv_modules

    def load_from_ckpt(self, model_dir, model_config, runtime_mapping,
                       ckpt_source):
        if ckpt_source == "hf":
            self.load_from_hf(model_dir, model_config, runtime_mapping)
        elif ckpt_source == "nemo":
            self.load_from_nemo(model_dir, model_config, runtime_mapping)
        else:
            assert False, f"LoraManager does not support source {ckpt_source}"

    def load_from_nemo(self, model_files, model_config, runtime_mapping):
        tp_size = runtime_mapping.tp_size
        tp_rank = runtime_mapping.tp_rank
        lora_target_modules = model_config.lora_target_modules
        dtype = model_config.dtype
        uids = list(map(str, range(len(model_files))))
        self.lora_target_modules = lora_target_modules
        self.missing_qkv_modules = self.get_missing_qkv_modules(
            lora_target_modules)

        def load_from_model_file(uid, model_file):
            if uid not in self._lora_cpp_weights:
                self._lora_cpp_weights[uid] = []
            if uid not in self._lora_weight_config:
                self._lora_weight_config[uid] = []

            _, nemo_weights = unpack_nemo_weights(model_file)
            all_lora_weights = get_all_nemo_lora_weights(nemo_weights)

            self._lora_uid_to_low_ranks[uid] = {}
            self._lora_weights_pointers_list[uid] = {}
            for layer_idx in sorted(all_lora_weights.keys()):
                self._lora_uid_to_low_ranks[uid][layer_idx] = {}
                self._lora_weights_pointers_list[uid][layer_idx] = {}

                for lora_module in lora_target_modules:
                    if lora_module != "attn_qkv":
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = 0
                        continue

                    if lora_module == "attn_qkv":
                        t_in = all_lora_weights[layer_idx]["in"]
                        t_out = all_lora_weights[layer_idx]["out"]
                        assert t_out.shape[0] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[0] // tp_size,
                                            dim=0)[tp_rank].contiguous()
                    else:
                        t_in = None
                        t_out = None

                    if t_in is not None and t_out is not None:
                        t_in = t_in.cuda().to(
                            str_dtype_to_torch(dtype)).contiguous()
                        t_out = t_out.cuda().to(
                            str_dtype_to_torch(dtype)).contiguous()
                        rank = t_in.shape[0]
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = int(rank)
                        self._lora_weights_pointers_list[uid][layer_idx][
                            lora_module] = [t_in.data_ptr(),
                                            t_out.data_ptr()]

                        # prevent torch free this buffer
                        self._lora_weights.append(t_in)
                        self._lora_weights.append(t_out)
                        self._lora_cpp_weights[uid].append(
                            torch.concatenate([t_in.flatten(),
                                               t_out.flatten()]))
                        self._lora_weight_config[uid].append(
                            np.array([
                                self.LORA_MODULE_IDS[lora_module], layer_idx,
                                int(rank)
                            ],
                                     dtype=np.int32))

        for uid, model_file in zip(uids, model_files):
            load_from_model_file(uid, model_file)
            release_gc()

    def load_from_hf(self,
                     model_dirs,
                     model_config,
                     runtime_mapping,
                     component=None):
        '''
        lora config of https://huggingface.co/hfl/chinese-alpaca-2-lora-7b
        {
            "base_model_name_or_path": "/Llama-2-7b-hf",
            "bias": "none",
            "enable_lora": null,
            "fan_in_fan_out": false,
            "inference_mode": true,
            "lora_alpha": 128.0,
            "lora_dropout": 0.05,
            "merge_weights": false,
            "modules_to_save": [
                "embed_tokens",
                "lm_head"
            ],
            "peft_type": "LORA",
            "r": 64,
            "target_modules": [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "down_proj",
                "up_proj"
            ],
            "task_type": "CAUSAL_LM"

        }

        keys in adapter_model.bin:
            base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.k_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.k_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.self_attn.o_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight torch.Size([4096, 64])
            base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.gate_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.up_proj.lora_A.weight torch.Size([64, 4096])
            base_model.model.model.layers.0.mlp.up_proj.lora_B.weight torch.Size([11008, 64])
            base_model.model.model.layers.0.mlp.down_proj.lora_A.weight torch.Size([64, 11008])
            base_model.model.model.layers.0.mlp.down_proj.lora_B.weight torch.Size([4096, 64])
            ...

        '''
        tp_size = runtime_mapping.tp_size
        tp_rank = runtime_mapping.tp_rank

        lora_hf_configs = []
        uids = []
        for i, model_dir in enumerate(model_dirs):
            with open(f"{model_dir}/adapter_config.json", 'r') as f:
                config = json.load(f)
                lora_hf_configs.append(config)
                uids.append(str(i))

        lora_target_modules = model_config.lora_target_modules
        dtype = model_config.dtype
        hf_modules_to_trtllm_modules = invert_module_mapping(
            model_config.trtllm_modules_to_hf_modules)
        hf_modules = set(hf_modules_to_trtllm_modules.keys())
        missing_qkv_modules = self.get_missing_qkv_modules(lora_target_modules)
        self.lora_target_modules = lora_target_modules
        self.missing_qkv_modules = missing_qkv_modules

        def preprocess_lora_weights(lora_model):
            # Swap weights of gate_up_proj
            for key, value in lora_model.items():
                if "gate_up_proj.lora_B.weight" in key:
                    original_weights = value.contiguous().clone()
                    half_split = original_weights.shape[0] // 2
                    first_half = original_weights[:half_split, :]
                    second_half = original_weights[half_split:, :]
                    value = torch.cat((second_half, first_half), dim=0)
                    lora_model[key] = value
            return lora_model

        def load_from_model_dir(uid, model_dir, hf_config):
            if uid not in self._lora_cpp_weights:
                self._lora_cpp_weights[uid] = []
            if uid not in self._lora_weight_config:
                self._lora_weight_config[uid] = []

            lora_model = load_state_dict(
                get_model_path(model_dir, "adapter_model"))
            lora_model = preprocess_lora_weights(lora_model)
            all_weights = get_all_hf_lora_weights(lora_model, hf_modules,
                                                  component)
            rank = int(hf_config["r"])
            rs_lora = bool(hf_config.get("use_rslora", False))

            self._lora_uid_to_low_ranks[uid] = {}
            self._lora_weights_pointers_list[uid] = {}
            for layer_idx in sorted(all_weights.keys()):
                layer_weights = all_weights[layer_idx]
                self._lora_uid_to_low_ranks[uid][layer_idx] = {}
                self._lora_weights_pointers_list[uid][layer_idx] = {}

                for lora_module in missing_qkv_modules:
                    hf_module = model_config.trtllm_modules_to_hf_modules[
                        lora_module]
                    if isinstance(hf_module, list):
                        hf_module = hf_module[0]
                    layer_weights[hf_module] = {
                        "in": torch.zeros(rank, model_config.hidden_size),
                        "out": torch.zeros(model_config.hidden_size, rank),
                    }

                for hf_module, module_weights in layer_weights.items():
                    lora_module = hf_modules_to_trtllm_modules[hf_module]
                    if lora_module not in lora_target_modules:
                        self._lora_uid_to_low_ranks[uid][layer_idx][
                            lora_module] = 0
                        continue
                    if "in" not in module_weights:
                        is_moe = True
                        t_in = torch.stack([
                            module_weights[expert_idx]["in"]
                            for expert_idx in sorted(module_weights.keys())
                        ])
                        t_out = torch.stack([
                            module_weights[expert_idx]["out"]
                            for expert_idx in sorted(module_weights.keys())
                        ])
                    else:
                        is_moe = False
                        t_in = module_weights["in"]
                        t_out = module_weights["out"]
                    if lora_module in ["moe_router", "mlp_router"]:
                        pass
                    elif "moe" in lora_module and runtime_mapping.has_moe_ep():
                        pass
                    elif lora_module in [
                            "attn_dense",
                            "cross_attn_dense",
                            "mlp_4h_to_h",
                            "moe_4h_to_h",
                    ]:
                        # split by row
                        dim = 2 if is_moe else 1
                        assert t_in.shape[dim] % tp_size == 0
                        t_in = torch.split(t_in,
                                           t_in.shape[dim] // tp_size,
                                           dim=dim)[tp_rank].contiguous()
                    else:
                        # split by column
                        dim = 1 if is_moe else 0
                        assert t_out.shape[dim] % tp_size == 0
                        t_out = torch.split(t_out,
                                            t_out.shape[dim] // tp_size,
                                            dim=dim)[tp_rank].contiguous()

                    t_in = t_in.cuda().contiguous()
                    t_out = t_out.cuda().contiguous()
                    if rs_lora:
                        scale = float(hf_config["lora_alpha"]) / np.sqrt(rank)
                    else:
                        scale = float(hf_config["lora_alpha"]) / rank
                    t_out = t_out * scale
                    t_in = t_in.to(str_dtype_to_torch(dtype))
                    t_out = t_out.to(str_dtype_to_torch(dtype))

                    rank_dim = 1 if is_moe else 0
                    assert t_in.shape[rank_dim] == rank
                    self._lora_uid_to_low_ranks[uid][layer_idx][
                        lora_module] = rank
                    self._lora_weights_pointers_list[uid][layer_idx][
                        lora_module] = [t_in.data_ptr(),
                                        t_out.data_ptr()]

                    # prevent torch free this buffer
                    self._lora_weights.append(t_in)
                    self._lora_weights.append(t_out)
                    self._lora_cpp_weights[uid].append(
                        torch.concatenate([t_in.flatten(),
                                           t_out.flatten()]))
                    self._lora_weight_config[uid].append(
                        np.array([
                            self.LORA_MODULE_IDS[lora_module], layer_idx,
                            int(hf_config['r'])
                        ],
                                 dtype=np.int32))

        for uid, model_dir, hf_config in zip(uids, model_dirs, lora_hf_configs):
            load_from_model_dir(uid, model_dir, hf_config)
            release_gc()

    def save_lora_weights_to_bin(self, out_dir):

        def save_val(val, dir, key, tp_num=None, write_npy=False):
            ext = "npy" if write_npy else "bin"
            suffix = ext if tp_num is None else f"{tp_num}.{ext}"
            if write_npy:
                np.save(dir / f"model.{key}.{suffix}", val)
            else:
                val.tofile(dir / f"model.{key}.{suffix}")

        if isinstance(out_dir, str):
            out_dir_path = Path(out_dir)
        elif isinstance(out_dir, Path):
            out_dir_path = out_dir
        else:
            assert False
        for uid in self._lora_cpp_weights:
            if uid == '-1':
                continue

            all_weights = np.expand_dims(
                np.stack([
                    torch_to_numpy(w.flatten().contiguous())
                    for w in self._lora_cpp_weights[uid]
                ]), 0)
            all_configs = np.expand_dims(
                np.stack(self._lora_weight_config[uid]), 0)

            uid_path = out_dir_path / f"{uid}"
            uid_path.mkdir(parents=True, exist_ok=True)
            save_val(all_weights,
                     uid_path,
                     "lora_weights",
                     tp_num=None,
                     write_npy=True)
            save_val(all_configs,
                     uid_path,
                     "lora_config",
                     tp_num=None,
                     write_npy=True)

    def uid_to_low_ranks(self, uid: str):
        assert isinstance(uid, str)
        return self._lora_uid_to_low_ranks[uid]

    @property
    def lora_weights(self):
        return self._lora_weights

    @property
    def lora_weights_pointers_list(self):
        return self._lora_weights_pointers_list

    def input_buffers(self, lora_uids, mapping: Mapping, num_layers: int):
        inputs = {}
        for layer_idx in mapping.pp_layers(num_layers):
            for lora_module in (self.lora_target_modules +
                                self.missing_qkv_modules):
                lora_ranks_ = []
                lora_ptrs_ = []
                for lora_uid in lora_uids:
                    lora_rank = 0
                    lora_ptrs = [0, 0]

                    if lora_uid != "-1":
                        low_ranks = self.uid_to_low_ranks(lora_uid)

                        if (layer_idx in low_ranks
                                and lora_module in low_ranks[layer_idx].keys()
                                and low_ranks[layer_idx][lora_module] != 0):

                            lora_rank = low_ranks[layer_idx][lora_module]
                            lora_ptrs = self.lora_weights_pointers_list[
                                lora_uid][layer_idx][lora_module]

                    lora_ranks_.append(lora_rank)
                    lora_ptrs_.append(lora_ptrs)

                inputs[
                    f'{lora_module}_lora_ranks_{layer_idx}'] = torch.IntTensor(
                        lora_ranks_)
                inputs[
                    f'{lora_module}_lora_weights_pointers_{layer_idx}'] = torch.LongTensor(
                        lora_ptrs_)
        return inputs
