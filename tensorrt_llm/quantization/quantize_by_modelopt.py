# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from examples/quantization/hf_ptq.py
"""

import contextlib
import copy
import json
import os
import random
import shutil
import sys
import tarfile
import tempfile
import time

import numpy as np
import safetensors
import torch
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ..logger import logger
from ..mapping import Mapping
from .mode import QuantAlgo

EMPTY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": False,
        },
        "*input_quantizer": {
            "enable": False
        },
        "*lm_head*": {
            "enable": False
        },
        "*output_layer*": {
            "enable": False
        },
        "default": {
            "enable": False
        },
    },
    "algorithm": "max",
}

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}



def quant_cfg_choices():
    import modelopt.torch.quantization as atq
    QUANT_CFG_CHOICES = {
        "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
        "fp8": atq.FP8_DEFAULT_CFG,
        "int4_awq": atq.INT4_AWQ_CFG,
        "w4a8_awq": atq.W4A8_AWQ_BETA_CFG,
        "int8_wo": EMPTY_CFG,
        "int4_wo": EMPTY_CFG,
        "full_prec": EMPTY_CFG,
        "int8_mix" : EMPTY_CFG
    }
    return QUANT_CFG_CHOICES


MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt2",
    "Xverse": "llama",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
    "Gemma": "gemma",
    "MixtralForCausalLM": "llama",
    "ArcticForCausalLM": "llama",
    "Phi3SmallForCausalLM": "phi3small",
    "Phi3ForCausalLM": "phi3",
    "Starcoder2ForCausalLM": "gptnext",
}


def get_tokenizer(ckpt_path, max_seq_length=2048, model_type=None):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_length,
        padding_side="left",
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        if model_type and model_type == "qwen":
            # qwen use token id 151643 as pad and eos tokens
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)
            tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        else:
            tokenizer.pad_token = tokenizer.eos_token
    assert tokenizer.pad_token is not None, f"Pad token for {model_type} cannot be set!"

    return tokenizer


def _get_vila_model(model_dir):
    sys.path.append(model_dir + "/../VILA")
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        model_dir,
        device_map='auto',
        trust_remote_code=True,
    )
    return model.llm


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16" or dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = torch.float16
    elif dtype == "fp32" or dtype == "float32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    hf_config = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    model_cls = AutoModelForCausalLM
    if hf_config.model_type == "llava":
        from transformers import LlavaForConditionalGeneration
        model_cls = LlavaForConditionalGeneration
    if "vila" in ckpt_path:
        model = _get_vila_model(ckpt_path)
    else:
        model = model_cls.from_pretrained(
            ckpt_path,
            device_map="auto" if device != "cpu" else "cpu",
            torch_dtype="auto",
            trust_remote_code=True)
        if hf_config.model_type == "llava":
            model = model.language_model
    model.eval()

    model_dtype = next(model.parameters()).dtype
    if dtype != model_dtype:
        print(
            f"[TensorRT-LLM][WARNING] The manually set model data type is {dtype}, "
            f"but the data type of the HuggingFace model is {model_dtype}.")

    return model


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512):
    print("Loading calibration dataset")
    if dataset_name_or_dir == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        dataset = dataset["text"][:calib_size]
    elif "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir, name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    elif os.path.isdir(dataset_name_or_dir):
        print(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir, split="train")
        dataset = dataset["text"][:calib_size]
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding=True,
                                                truncation=True,
                                                max_length=block_size)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded,
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, calib_dataloader=None):
    import modelopt.torch.quantization as atq

    def calibrate_loop():
        if calib_dataloader is None:
            return
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            # model might be mapped to different device because the device_map is auto
            data = data.to(model.device)
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print("Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                                 start_time))

    return model


def combine_medusa_weight(tp_size, pp_size, base_model_output_dir,
                          num_medusa_heads, num_medusa_layers, max_draft_len,
                          medusa_hidden_act, medusa_model_dir,
                          quant_medusa_head):

    with open(f"{medusa_model_dir}/config.json", "r") as fp:
        medusa_config = json.load(fp)

    num_medusa_heads_from_config = medusa_config.get('medusa_num_heads',
                                                     num_medusa_heads)
    num_medusa_layers = medusa_config.get('medusa_num_layers',
                                          num_medusa_layers)
    if num_medusa_heads is None:
        num_medusa_heads = num_medusa_heads_from_config

    assert max_draft_len > 0, "should have max_draft_len > 0"

    world_size = tp_size * pp_size
    # Process for each rank
    for rank in range(world_size):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=tp_size,
                          pp_size=pp_size)
        # 1. Load medusa weight for each rank
        from tensorrt_llm.models.medusa.weight import load_medusa_hf
        medusa_weights = load_medusa_hf(medusa_path=medusa_model_dir,
                                        num_medusa_heads=num_medusa_heads,
                                        num_medusa_layers=num_medusa_layers,
                                        mapping=mapping,
                                        dtype="float16")
        # 2. Load base model safetensors (after quant)
        base_model_weights = load_file(
            f"{base_model_output_dir}/rank{rank}.safetensors")

        # 3. Combine and save weight
        base_model_weights.update(medusa_weights)
        save_file(base_model_weights,
                  f"{base_model_output_dir}/rank{rank}.safetensors")

    # 4. Add medusa config into config.json
    with open(f"{base_model_output_dir}/config.json", 'r') as f:
        base_model_config = json.load(f)
        f.close()

    with open(f"{base_model_output_dir}/config.json", 'w') as f:
        base_model_config['architecture'] = "MedusaForCausalLM"
        base_model_config['quantization']['exclude_modules'] = [
            'lm_head',
            '*router',
            '*vocab_embedding',
            '*position_embedding',
            '*block_embedding',
        ]
        if not quant_medusa_head:
            base_model_config['quantization']['exclude_modules'].append(
                '*medusa_heads*')

        base_model_config['max_draft_len'] = max_draft_len
        base_model_config['num_medusa_heads'] = num_medusa_heads
        base_model_config['num_medusa_layers'] = num_medusa_layers
        json.dump(base_model_config, f, indent=4)

    torch.cuda.empty_cache()
    print("Combine medusa heads' weight, done.")


def quantize_and_export(*,
                        model_dir,
                        device,
                        calib_dataset,
                        dtype,
                        qformat,
                        kv_cache_dtype,
                        calib_size,
                        batch_size,
                        calib_max_seq_length,
                        awq_block_size,
                        output_dir,
                        tp_size,
                        pp_size,
                        seed,
                        tokenizer_max_seq_length,
                        num_medusa_heads=None,
                        num_medusa_layers=None,
                        max_draft_len=None,
                        medusa_hidden_act=None,
                        medusa_model_dir=None,
                        quant_medusa_head=None):
    '''
        Load model from the model_dir, call Modelopt to quantize the model, and then export
        the quantized model as TRT-LLM checkpoint
    '''
    try:
        import modelopt  # noqa
    except ImportError as e:
        logger.error(
            "Failed to import modelopt, pls check the Modelopt installation. Currently it is known to be unsupported on Windows OS"
        )
        raise e

    from modelopt.torch.export import export_tensorrt_llm_checkpoint

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(seed)
    np.random.seed(seed)

    model = get_model(model_dir, dtype, device=device)
    model_type = get_model_type(model)
    if "vila" in model_dir:
        tokenizer = get_tokenizer(model_dir + "/llm",
                                  max_seq_length=tokenizer_max_seq_length,
                                  model_type=model_type)
    else:
        tokenizer = get_tokenizer(model_dir,
                                  max_seq_length=tokenizer_max_seq_length,
                                  model_type=model_type)
    print("tensorrt_llm/quantization/quantize_by_modelopt.py")
    print("quantize_and_export")

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        print(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                print(
                    f"AWQ calibration could take longer with calib_size = {calib_size}, Using"
                    " calib_size=32 instead")
                calib_size = 32
            print(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )
        if "mix" not in qformat:
            calib_dataloader = get_calib_dataloader(
                dataset_name_or_dir=calib_dataset,
                tokenizer=tokenizer,
                batch_size=batch_size,
                calib_size=calib_size,
                block_size=calib_max_seq_length,
            )

        else:
            calib_dataloader = {}
        if qformat in quant_cfg_choices():
            quant_cfg = quant_cfg_choices()[qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {qformat}")

        if "awq" in qformat:
            quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
            weight_quantizer = quant_cfg["quant_cfg"][
                "*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = awq_block_size

        if kv_cache_dtype is not None:
            if kv_cache_dtype == "fp8":
                for value in KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

        model = quantize_model(model, quant_cfg, calib_dataloader)

    with torch.inference_mode():
        if model_type is None:
            print(
                f"Unknown model type {type(model).__name__}. Continue exporting..."
            )
            model_type = f"unknown:{type(model).__name__}"

        export_path = output_dir
        start_time = time.time()

        print("----------export_tensorrt_llm_checkpoint--------------")
        export_tensorrt_llm_checkpoint(model,
                                       model_type,
                                       getattr(torch, dtype),
                                       export_dir=export_path,
                                       inference_tensor_parallel=tp_size,
                                       inference_pipeline_parallel=pp_size)

        with open(f"{export_path}/config.json", "r") as f:
            tensorrt_llm_config = json.load(f)

        # Workaround for wo quantization
        if qformat in ["int8_wo", "int4_wo", "full_prec"]:
            if qformat == "int8_wo":
                tensorrt_llm_config["quantization"][
                    "quant_algo"] = QuantAlgo.W8A16
            elif qformat == "int4_wo":
                tensorrt_llm_config["quantization"][
                    "quant_algo"] = QuantAlgo.W4A16
            else:
                tensorrt_llm_config["quantization"]["quant_algo"] = None

        # HF uses rope_scaling while tensorrt_llm uses rotary_scaling
        if hasattr(
                model.config,
                "rope_scaling") and "rotary_scaling" not in tensorrt_llm_config:
            tensorrt_llm_config["rotary_scaling"] = getattr(
                model.config, "rope_scaling")
        with open(f"{export_path}/config.json", "w") as f:
            json.dump(tensorrt_llm_config, f, indent=4)

        # Workaround for Modelopt 0.9.x fp8_kv_cache knob issue
        if qformat == 'fp8' and kv_cache_dtype is None:
            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)
            tensorrt_llm_config["quantization"]["kv_cache_quant_algo"] = None
            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

        # Workaround for share_embedding_table
        if pp_size == 1:
            with safetensors.safe_open(f"{export_path}/rank0.safetensors",
                                       framework='pt',
                                       device='cpu') as f:
                share_embedding_table = 'lm_head.weight' not in f.keys()
            if share_embedding_table:
                with open(f"{export_path}/config.json", "r") as f:
                    tensorrt_llm_config = json.load(f)
                tensorrt_llm_config["share_embedding_table"] = True
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

        # Workaround for qwen version
        if model_type == 'qwen':
            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)
            qwen_config = AutoConfig.from_pretrained(model_dir,
                                                     trust_remote_code=True)
            tensorrt_llm_config["qwen_type"] = qwen_config.model_type
            if qwen_config.model_type == "qwen2":
                tensorrt_llm_config["norm_epsilon"] = qwen_config.rms_norm_eps
                tensorrt_llm_config["rotary_base"] = qwen_config.rope_theta
            tensorrt_llm_config[
                "intermediate_size"] = qwen_config.intermediate_size
            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

        # Set rotary parameters correctly for chatglm.
        if model_type == 'chatglm':
            rotary_base = 10000.0
            rotary_embedding_scaling = None
            chatglm_config = AutoConfig.from_pretrained(model_dir,
                                                        trust_remote_code=True)
            chatglm_version = tensorrt_llm_config['chatglm_version']
            rope_ratio = tensorrt_llm_config.get('rope_ratio', 1.0)
            if chatglm_version == 'chatglm2':
                if rope_ratio > 1:
                    rotary_embedding_scaling = {
                        'type': 'linear',
                        'factor': rope_ratio
                    }
            elif chatglm_version == 'chatglm3':
                rotary_base *= rope_ratio

            with open(f"{export_path}/config.json", "r") as f:
                tensorrt_llm_config = json.load(f)
            tensorrt_llm_config['rotary_base'] = rotary_base
            tensorrt_llm_config['rotary_scaling'] = rotary_embedding_scaling
            tensorrt_llm_config['rotary_pct'] = 0.5
            with open(f"{export_path}/config.json", "w") as f:
                json.dump(tensorrt_llm_config, f, indent=4)

        torch.cuda.empty_cache(
        )  # otherwise torch is keeping using GPU, other routine like build engine has less free GPU to use

        # Workaround for combining medusa head
        # TODO: move these integration into modelopt to avoid redundant reading and writing
        if medusa_model_dir is not None:
            combine_medusa_weight(tp_size, pp_size, export_path,
                                  num_medusa_heads, num_medusa_layers,
                                  max_draft_len, medusa_hidden_act,
                                  medusa_model_dir, quant_medusa_head)
        end_time = time.time()
        print(
            "Quantized model exported to {} \nTotal time used {:.2f} s.".format(
                export_path, end_time - start_time))


def load_config(model_file: str):
    """Load model config from extracted directory or '.nemo' tarball."""
    from modelopt.torch.utils import print_rank_0
    from omegaconf import OmegaConf

    if os.path.isfile(model_file):
        with tempfile.TemporaryDirectory() as tmp, tarfile.open(
                model_file, "r:") as tar:
            try:
                tar.extract("./model_config.yaml", path=tmp)
            except KeyError:
                print_rank_0("File name not found, trying legacy name...")
                tar.extract("model_config.yaml", path=tmp)
            model_config = OmegaConf.load(os.path.join(tmp,
                                                       "model_config.yaml"))
    elif os.path.isdir(model_file):
        model_config = OmegaConf.load(
            os.path.join(model_file, "model_config.yaml"))
    else:
        raise FileNotFoundError(model_file)

    return model_config


def save_artifacts(model, output_dir: str, use_abspath: bool = False) -> None:
    """Save all model artifacts and tokenizer config to a given output directory."""
    from modelopt.torch.utils import print_rank_0
    from nemo.utils import AppState
    from omegaconf import OmegaConf

    app_state = AppState()
    model_file = app_state.model_restore_path
    model_cfg = copy.deepcopy(model.cfg)
    if not hasattr(model, "artifacts"):
        if hasattr(model_cfg, "tokenizer"):
            OmegaConf.save(model_cfg.tokenizer,
                           os.path.join(output_dir, "tokenizer_config.yaml"))
        return

    # Setup model file handling context: directory or tarball
    if os.path.isfile(model_file):
        model_file_handler = tarfile.open
        kwargs = {"name": model_file, "mode": "r:"}
    elif os.path.isdir(model_file):
        model_file_handler = contextlib.nullcontext
        kwargs = {}
    else:
        raise FileNotFoundError(model_file)

    # Copy or extract artifacts depending on the context
    with model_file_handler(**kwargs) as maybe_tar:
        for arti_name, arti_item in model.artifacts.items():
            _, arti_file = arti_item.path.split("nemo:")
            arti_path = os.path.join(output_dir, arti_name)
            if maybe_tar is not None:
                try:
                    maybe_tar.extract(f"./{arti_file}", path=output_dir)
                except KeyError:
                    print_rank_0("File name not found, trying legacy name...")
                    maybe_tar.extract(f"{arti_file}", path=output_dir)
                os.rename(os.path.join(output_dir, arti_file), arti_path)
            else:
                shutil.copy(os.path.join(model_file, arti_file), arti_path)
            # Store artifact path as basename by default. Otherwise save absolute path but bear in mind
            # that in this case output directory should be permanent for correct artifact recovery later
            arti_path = os.path.abspath(
                arti_path) if use_abspath else os.path.basename(arti_path)
            OmegaConf.update(model_cfg, arti_name, arti_path)

    if hasattr(model_cfg, "tokenizer"):
        OmegaConf.save(model_cfg.tokenizer,
                       os.path.join(output_dir, "tokenizer_config.yaml"))


def unwrap_model(model, module_instances=None):
    from megatron.core import DistributedDataParallel as DDP
    from megatron.core.transformer.module import Float16Module

    if module_instances is None:
        module_instances = (DDP, Float16Module)

    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def get_nemo_calib_dataloader(dataset_name_or_dir="cnn_dailymail",
                              batch_size=64,
                              calib_size=512,
                              max_sequence_length=512):
    if dataset_name_or_dir == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        text_column = "text"
    elif "wikitext" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir,
                               "wikitext-103-v1",
                               split="train")
        text_column = "text"
    elif "cnn_dailymail" in dataset_name_or_dir:
        dataset = load_dataset(dataset_name_or_dir, name="3.0.0", split="train")
        text_column = "article"
    elif os.path.isdir(dataset_name_or_dir):
        print(
            f"Recognized local dataset repo {dataset_name_or_dir} for calibration; "
            "assuming the calibration data are in the train split and text column."
        )
        dataset = load_dataset(dataset_name_or_dir, split="train")
        text_column = "text"
    else:
        raise NotImplementedError(
            f"Unsupported dataset name or local repo directory: {dataset_name_or_dir}."
        )
    calib_size = max(min(len(dataset), calib_size), batch_size)
    for i in range(calib_size // batch_size):
        batch = dataset[i * batch_size:(i + 1) * batch_size][text_column]
        for j in range(len(batch)):
            batch[j] = batch[j][:max_sequence_length]
        yield batch


def quantize_nemo_and_export(*, nemo_ckpt_path, decoder_type, calib_dataset,
                             calib_tp_size, calib_pp_size, dtype, qformat,
                             kv_cache_dtype, calib_size, batch_size,
                             calib_max_seq_length, awq_block_size, output_dir,
                             tp_size, pp_size, seed):
    try:
        import modelopt  # noqa
    except ImportError as e:
        logger.error(
            "Failed to import modelopt, pls check the modelopt installation. Currently it is known to be unsupported on Windows OS"
        )
        raise e

    import modelopt.torch.quantization as atq
    from megatron.core import parallel_state
    from megatron.core.transformer.module import Float16Module
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    from modelopt.torch.utils import print_rank_0
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import \
        MegatronGPTModel
    from nemo.collections.nlp.parts.nlp_overrides import (
        NLPDDPStrategy, NLPSaveRestoreConnector)
    from omegaconf.omegaconf import open_dict
    from pytorch_lightning.trainer.trainer import Trainer

    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for the inference.")

    random.seed(seed)
    np.random.seed(seed)

    # dtype is used for non-quantized layers
    supported_dtype = ["float16", "bfloat16"]
    assert (dtype in supported_dtype
            ), f"{dtype} not supported. Supported dtypes are {supported_dtype}"
    torch_dtype = getattr(torch, dtype)

    model_cfg = load_config(nemo_ckpt_path)

    with open_dict(model_cfg):
        model_cfg.activations_checkpoint_method = None
        model_cfg.activations_checkpoint_granularity = None
        model_cfg.tensor_model_parallel_size = calib_tp_size
        model_cfg.pipeline_model_parallel_size = calib_pp_size
        model_cfg.sequence_parallel = False
        # Only custom modelopt spec is supported for PTQ: this custom spec is largely based on local Megatron-LM
        # layer definitions to avoid Transformer Engine implementations that are currently not supported.
        model_cfg.name = "ammo"

    # trainer required for restoring model parallel models
    trainer_config = {
        'devices': calib_tp_size * calib_pp_size,
        'num_nodes': 1,
        'accelerator': 'gpu',
        'logger': False,
        'precision': model_cfg.precision,
        'enable_checkpointing': False,
    }
    trainer = Trainer(strategy=NLPDDPStrategy(), **trainer_config)
    connector = NLPSaveRestoreConnector()

    model = MegatronGPTModel.restore_from(
        restore_path=nemo_ckpt_path,
        trainer=trainer,
        override_config_path=model_cfg,
        save_restore_connector=connector,
    )
    model.freeze()

    print_rank_0(model)
    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.module.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    # Check whether the DDP is initialized
    if parallel_state.is_unitialized():

        def dummy():
            return

        if model.trainer.strategy.launcher is not None:
            model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
        model.trainer.strategy.setup_environment()

    inference_config = {
        'greedy': False,
        'top_k': 0,
        'top_p': 0.9,
        'temperature': 1.0,
        'add_BOS': True,
        'tokens_to_generate': 30,
        'all_probs': False,
        'repetition_penalty': 1.2,
        'min_tokens_to_generate': 0,
        'compute_logprob': False,
        'batch_size': batch_size,
        'max_context_length': calib_max_seq_length,
    }
    model.set_inference_config(inference_config)

    if qformat in ["full_prec", "int8_wo", "int4_wo"
                   ] and kv_cache_dtype is None:
        print_rank_0(f"No quantization applied, export {dtype} model")
    else:
        if "awq" in qformat:
            if calib_size > 32:
                print_rank_0(
                    "AWQ calibration could take longer with calib_size ="
                    f" {calib_size}, Using calib_size=32 instead")
                calib_size = 32
            print_rank_0(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument inference.batch_size=<batch_size> to the command"
                " line.\n")

        dataloader = get_nemo_calib_dataloader(
            dataset_name_or_dir=calib_dataset,
            batch_size=batch_size,
            calib_size=calib_size,
            max_sequence_length=calib_max_seq_length,
        )

        # =================== Start Quantization ====================
        if qformat in quant_cfg_choices():
            quant_cfg = quant_cfg_choices()[qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {qformat}")

        if "awq" in qformat:
            quant_cfg = copy.deepcopy(quant_cfg_choices()[qformat])
            weight_quantizer = quant_cfg["quant_cfg"][
                "*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = awq_block_size

        if kv_cache_dtype is not None:
            if kv_cache_dtype == "fp8":
                for value in KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

        print_rank_0(quant_cfg)

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we use int8 kv cache.
        # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for nemotron.
        # quant_cfg["quant_cfg"]["*output_quantizer"] = {  # type: ignore[index]
        #     "num_bits": 8 if args.qformat == "int8_sq" else (4, 3),
        #     "axis": None,
        #     "enable": args.decoder_type != "gptnext",
        # }

        dataloader = [data for data in dataloader]

        def forward_loop(model):
            for i, batch in enumerate(dataloader):
                print_rank_0(f"Calibrating batch {i}")
                model.predict_step(batch, i)

        start_time = time.time()
        model = atq.quantize(model, quant_cfg,
                             forward_loop)  # type: ignore[arg-type]
        end_time = time.time()
        tot_time = end_time - start_time
        tput = calib_size / tot_time
        print_rank_0(
            f"Quantization done. Total time used {tot_time}s. Throughput {tput} samples/s"
        )
        # =================== End Quantization ======================

        if decoder_type == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if qformat == "fp8":
                maxbound = 448
            elif qformat == "int8_sq":
                maxbound = 127
            model = atq.postprocess_amax(
                model, "*input_quantizer",
                lambda amax: torch.clamp(amax, min=0.01 * maxbound))

        if torch.distributed.get_rank() == 0:
            atq.print_quant_summary(model)

    if model_cfg.megatron_amp_O2:
        model.model = unwrap_model(model.model, Float16Module)

    start_time = time.time()
    export_tensorrt_llm_checkpoint(
        model,
        decoder_type,
        torch_dtype,
        export_dir=output_dir,
        inference_tensor_parallel=tp_size,
        inference_pipeline_parallel=pp_size,
    )

    torch.cuda.empty_cache(
    )  # otherwise torch is keeping using GPU, other routine like build engine has less free GPU to use
    end_time = time.time()
    print_rank_0(
        f"Model config exported to: {output_dir}. Total time used {end_time - start_time}s"
    )
    if torch.distributed.get_rank() == 0:
        save_artifacts(model, output_dir, use_abspath=True)
