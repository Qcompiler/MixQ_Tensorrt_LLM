# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Code that export optimized models to the TensorRT-LLM checkpoint."""

import copy
import json
import math
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import save_file

try:
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.transformer.module import MegatronModule

    has_mcore = True
except ImportError:
    has_mcore = False
from modelopt import __version__
from modelopt.torch.utils import distributed as dist

from . import QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ
from .layer_utils import (
    build_decoder_config,
    build_embedding_config,
    build_layernorm_config,
    build_linear_config,
    build_medusa_heads_config,
    check_model_compatibility,
    get_kv_cache_dtype,
    get_quantization_format,
    get_transformer_layers,
    is_attention,
    is_decoder_list,
    is_embedding,
    is_layernorm,
    is_linear,
    is_quantlinear,
)
from .model_config import (
    CURRENT_VERSION,
    QUANTIZATION_NONE,
    ModelConfig,
)
from .model_config_utils import (
    merge_fc1_gate,
    merge_qkv,
    model_config_to_dict,
    naive_quantization,
    pack_linear_weights,
    split_config_and_weights,
    to_quantized_weight,
)
from .postprocess import (
    check_weight_shape_valid,
    pad_embedding_lm_head,
    postprocess_model_config,
    postprocess_tensors,
)
from .scaling_factor_utils import adjust_attn_amax_values
from .tensorrt_llm_utils import (
    convert_to_tensorrt_llm_config,
    is_tensorrt_llm_0_8_or_9,
    prepare_enc_dec_decoder_layer,
    prepare_enc_dec_export_dir,
    weights_to_npz,
)


def torch_to_tensorrt_llm_checkpoint(
    model: nn.Module,
    decoder_type: str,
    dtype: torch.dtype = torch.float16,
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    export_npz: bool = False,
    naive_fp8_quantization: bool = False,
    workspace_path: Optional[Union[Path, str]] = None,
) -> Iterator[Tuple[Dict[str, Any], Dict[str, torch.Tensor]]]:
    """Converts the torch model to the TensorRT-LLM checkpoint per GPU rank.

    TensorRT-LLM checkpoint is the LLM model format that can be used by the TensorRT-LLM build API.
    for the engine building process.
    https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/architecture/checkpoint.md

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt2, gptj, llama or gptnext.
        dtype: the weights data type to export the unquantized layers.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
            We will merge or split the calibration pipeline parallelism to inference.
            Default is 1, meaning no pipeline parallelism.
        export_npz: Whether or not to export the model_config to the old NPZ format for backward
            compatibility.
        naive_fp8_quantization: Quantize the model naively to FP8 without calibration.
            All scaling factors are set to 1.
        workspace_path: the path to the NFS directory for postprocess cross rank communication.

    Yields:
        A tuple of
            tensorrt_llm_config: A dict that maps to the ``PretrainedConfig`` in TensorRT-LLM.
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/modeling_utils.py
            weights: A dict that stores all model weights and scaling factors for each rank.
    """
    if export_npz:
        print("Warning: export_npz is going to be deprecated soon and replaced by safetensors.")

    if dtype not in [torch.float16, torch.bfloat16]:
        print(
            f"Warning: dtype {dtype} not fully compatible with TensorRT-LLM optimizations, Default to float16."
        )
        dtype = torch.float16

    if has_mcore and isinstance(model, MegatronModule):
        if not isinstance(model, MCoreGPTModel):
            raise ValueError("Only megatron.core.models.gpt.GPTModel is supported!")
        # MCoreGPTModel.config has type TransformerConfig
        #
        # We choose to deepcopy here since TransformerConfig deserialization is sensitive to
        # additional attributes.
        model_metadata_config = copy.deepcopy(model.config.__dict__)
        vocab_size = model.vocab_size
        model_metadata_config["max_position_embeddings"] = model.max_position_embeddings
        model_metadata_config["rotary_percent"] = model.rotary_percent
    elif hasattr(model, "config"):
        # Huggingface models
        model_metadata_config = model.config.__dict__
        vocab_size = model.config.vocab_size

        # For Baichuan 13B, we check if alibi is used with the alibi_mask property.
        if hasattr(model, "model") and hasattr(model.model, "alibi_mask"):
            model_metadata_config["alibi"] = True

        # For MPT, DBRX
        for config_key in ["attn_config", "ffn_config"]:
            config_value = model_metadata_config.get(config_key, None)
            if config_value:
                model_metadata_config.update(
                    config_value if isinstance(config_value, dict) else config_value.to_dict()
                )

    elif hasattr(model, "cfg"):
        # NeMo MegatronGPTModel
        model_metadata_config = dict(model.cfg)
        vocab_size = model.tokenizer.vocab_size
    else:
        raise ValueError("Cannot find valid model metadata config in model")

    if "multi_query_group_num" in model_metadata_config.keys():
        if model_metadata_config["multi_query_group_num"] % inference_tensor_parallel != 0:
            raise ValueError(
                "Cannot divide {} kv_heads into {} gpus".format(
                    model_metadata_config["multi_query_group_num"], inference_tensor_parallel
                )
            )

    training_pipeline_parallel = model_metadata_config.get("pipeline_model_parallel_size", 1)
    training_tensor_parallel = dist.size() // training_pipeline_parallel
    model_metadata_config["training_pipeline_parallel"] = training_pipeline_parallel
    model_metadata_config["training_tensor_parallel"] = training_tensor_parallel

    if "make_vocab_size_divisible_by" in model_metadata_config:
        # For some nemo models, the vocab_size is pre-padded.
        # We calculate the pre-padded vocab_size with this config: make_vocab_size_divisible_by.
        make_vocab_size_divisible_by = model_metadata_config["make_vocab_size_divisible_by"]
        make_vocab_size_divisible_by_with_tp = (
            make_vocab_size_divisible_by * training_tensor_parallel
        )
        vocab_size = int(
            math.ceil(vocab_size / make_vocab_size_divisible_by_with_tp)
            * make_vocab_size_divisible_by_with_tp
        )
        print(
            f"the new vocab_size is updated: {vocab_size}, make_vocab_size_divisible_by"
            f" {make_vocab_size_divisible_by}, training_tensor_parallel"
            f" {training_tensor_parallel}."
        )

    models = [model]
    if decoder_type in ["t5"]:
        model_lm_head = model.lm_head
        # For T5 model with encoder and decoder, we process the checkpoint separately.
        models = [model.encoder, model.decoder]

    for model in models:
        transformer_layers = get_transformer_layers(model)
        if training_pipeline_parallel == 1:
            compatible, has_position_embedding, has_embedding_layernorm = check_model_compatibility(
                transformer_layers
            )
        else:
            # For Megatron models with more than one PP,
            # we skip the compatibility check as not all ranks have the full model.
            # For Megatron Core GPTModel, both gptnext and llama do not have position embedding
            # nor embedding layernorm.
            compatible = len(transformer_layers) > 0
            has_position_embedding = False
            has_embedding_layernorm = False
        assert compatible, "The model is not supported"

        config = ModelConfig(
            version=CURRENT_VERSION,
            dtype=str(dtype).split(".")[1],
            rank=dist.rank(),
            tensor_parallel=training_tensor_parallel,
            pipeline_parallel=training_pipeline_parallel,
            vocab_size=vocab_size,
        )


        # For Encoder-Decoder Model like T5
        if decoder_type in ["t5"]:
            if model.is_decoder is False:
                config.enc_dec = "enc"
                model_metadata_config["enc_dec"] = "enc"
            else:
                config.enc_dec = "dec"
                model_metadata_config["enc_dec"] = "dec"

        # Build the full model_config dict layer by layer.
        for module in transformer_layers:
            if is_embedding(module):
                if config.vocab_embedding is None:
                    # We assume the first embedding in the list the vocab_embedding.

                    normalization_constant = 1
                    # Normalize vocab embedding for gemma.
                    if (
                        decoder_type == "gemma" and is_tensorrt_llm_0_8_or_9()
                    ) or decoder_type == "recurrentgemma":
                        normalization_constant = model_metadata_config["hidden_size"] ** 0.5

                    config.vocab_embedding = build_embedding_config(
                        module, dtype, normalization_constant=normalization_constant
                    )
                elif has_position_embedding:
                    config.position_embedding = build_embedding_config(module, dtype)
            elif is_decoder_list(module):
                layers = []
                for layer in module.children():
                    # Special process due to T5 model structure's specialty
                    if decoder_type in ["t5"]:
                        layer = layer.layer
                    layer_config = build_decoder_config(
                        layer, model_metadata_config, decoder_type, dtype
                    )
                    print("lianxiangTRT/modelopt/torch/export/model_config_export.py")
                    print("----layer_config---")
                    print(layer_config)
                    # exit()
                    # Special process for each decoder layer of Encoder-Decoder Model
                    if decoder_type in ["t5"]:
                        prepare_enc_dec_decoder_layer(
                            layer_config, model.config, model_metadata_config["enc_dec"], layers
                        )
                    layers.append(layer_config)
                config.layers = layers
            elif is_layernorm(module):
                if has_embedding_layernorm and config.ln_embed is None:
                    # Assume embedding_layernorm is placed before the ln_f.
                    config.ln_embed = build_layernorm_config(module, dtype)
                else:
                    config.ln_f = build_layernorm_config(module, dtype)
            elif is_linear(module):
                if model_metadata_config.get("share_embeddings_and_output_weights", False):
                    # NeMo/MCore models with shared embeddings - for example Gemma -
                    # the model head weight is None so we just skip processing
                    config.share_embedding_table = True
                    continue
                # TRT LLM forces the embedding table to be shared for the following models.
                force_share_embedding_table = decoder_type in ["gemma", "gemma2"]
                if force_share_embedding_table and torch.equal(
                    module.weight.to(dtype), config.vocab_embedding.weight
                ):
                    config.share_embedding_table = True
                else:
                    config.lm_head = build_linear_config(module, "column", dtype)

        # For decoder of Encoder-Decoder model, it needs some encoder information
        if decoder_type in ["t5"]:
            if model_metadata_config["enc_dec"] == "dec":
                config.encoder_hidden_size = models[0].config.d_model
                config.encoder_head_size = models[0].config.d_kv
                config.encoder_num_heads = models[0].config.num_heads

        # For the training time PP, not all ranks will have the lm_head layer.
        if config.lm_head is None:
            if decoder_type in ["t5"]:
                config.share_embedding_table = False
                if model_metadata_config["enc_dec"] == "dec":
                    config.lm_head = build_linear_config(model_lm_head, "column", dtype)
            elif training_pipeline_parallel == 1:
                # Models that share weights for lm_head and vocab_embedding
                assert decoder_type in [
                    "mpt",
                    "gpt2",
                    "gemma",
                    "gemma2",
                ], f"lm_head not available for decoder {decoder_type}"
                config.share_embedding_table = True

        # Handle Medusa Heads
        # TODO (chenhany): post-processing is not implemented yet
        config.medusa_heads = build_medusa_heads_config(model, dtype)
        print("modelopt/torch/export/model_config_export.py")
        print("-----------config.layers -----")
        print(config.layers[0].quantization)

        config.quantization = config.layers[0].quantization
        if config.quantization in [QUANTIZATION_INT4_AWQ, QUANTIZATION_W4A8_AWQ]:
            if config.vocab_size % 64 != 0:
                # TODO: Check if this works for Mixtral
                assert (
                    training_tensor_parallel == 1
                ), "We do not support padding for training time TP"
                print("Padding vocab_embedding and lm_head for AWQ weights export")
                pad_embedding_lm_head(config)

        check_weight_shape_valid(
            config,
            inference_tensor_parallel,
            training_tensor_parallel,
        )

        # Set this value to export unsharded model config. This is only for some models like phi,
        # so we don't split model config and will overwrite mapping in config.json.
        # TODO: check PP support for phi
        tp_size_overwrite = None
        if decoder_type in ["phi"]:
            tp_size_overwrite = inference_tensor_parallel

        # If inference_tensor_parallel or inference_pipeline_parallel is different from world_size,
        # we try to merge or split the model configs based on the rank selected.
        if (
            inference_tensor_parallel > 0
            or inference_pipeline_parallel > 0
            or training_pipeline_parallel > 1
        ):
            model_configs = postprocess_model_config(
                config,
                1 if tp_size_overwrite else inference_tensor_parallel,
                inference_pipeline_parallel,
                training_pipeline_parallel=training_pipeline_parallel,
                workspace_path=workspace_path,
            )
        else:
            model_configs = [config]

        for model_config in model_configs:
            print("modelopt/torch/export/model_config_export.py")
            print("checkmodel_config")
            #print(model_config)
            assert model_config.rank >= 0, "Invalid model_config, postprocess_model_config fails."

            if not model_config.quantization and naive_fp8_quantization:
                naive_quantization(model_config)

            if export_npz:
                # The npz format is not compatible with modelopt.deploy.llm for AWQ.
                model_config.version = 0.8
            else:
                merge_qkv(model_config)
                merge_fc1_gate(model_config)
                pack_linear_weights(model_config)
                print("pack_linear_weights")
                print("lianxiang/lianxiangTRT/modelopt/torch/export/model_config_export.py")
                # print(model_config)
                
                # exit()
                # Postprocess the tensors in the model_config.
                # Exporting the safetensors also allows the tensor to be a view.
                postprocess_tensors(
                    model_config, force_cpu=True, force_contiguous=True, force_non_view=False
                )

            weights = {}
            model_config_dict = model_config_to_dict(model_config)
            # We split the weights from model_config and save them separately as two files.
            split_config_and_weights(model_config_dict, weights)

            # We only export the json once across ranks as all jsons should be the same except for the rank.
            tensorrt_llm_config = convert_to_tensorrt_llm_config(model_config, tp_size_overwrite)

            yield tensorrt_llm_config, weights


def export_tensorrt_llm_checkpoint(
    model: nn.Module,
    decoder_type: str,
    dtype: torch.dtype = torch.float16,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
    inference_tensor_parallel: int = 0,
    inference_pipeline_parallel: int = 1,
    export_npz: bool = False,
    naive_fp8_quantization: bool = False,
    use_nfs_workspace: bool = False,
):
    """Exports the torch model to the TensorRT-LLM checkpoint and save to the export_dir.

    Args:
        model: the torch model.
        decoder_type: the type of the decoder, e.g. gpt2, gptj, llama or gptnext.
        dtype: the weights data type to export the unquantized layers.
        export_dir: the target export path.
        inference_tensor_parallel: The target inference time tensor parallel.
            We will merge or split the calibration tensor parallelism to inference.
            Default is 0, meaning using the calibration without manual config merge or split.
        inference_pipeline_parallel: The target inference time pipeline parallel.
            We will merge or split the calibration pipeline parallelism to inference.
            Default is 1, meaning no pipeline parallelism.
        inference_pipeline_parallel: The target inference time pipeline parallel.
        export_npz: Whether or not to export the model_config to the old NPZ format for backward
            compatibility.
        naive_fp8_quantization: Quantize the model naively to FP8 without calibration.
            All scaling factors are set to 1.
        use_nfs_workspace: if True, the an NFS workspace will be created under the export_dir and
            used as a shared memory for cross process/node communication.

    For tensorrt_llm deployment, save the representation under ``export_dir``.
    We will save the model_config as two files:

        * ``.json``: The nested dict that maps to the ``PretrainedConfig`` in TensorRT-LLM.
            https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/models/modeling_utils.py.
        * ``.safetensors``: The file for the list of weights as safetensors. Unique for each rank.
    """
    export_dir = Path(export_dir)
    export_root = export_dir
    export_dir.mkdir(parents=True, exist_ok=True)
    # Create a NFS workspace under the export folder which is also assumed to be NFS.
    workspace_path = None
    if use_nfs_workspace:
        workspace_path = export_dir.joinpath("workspace")
        workspace_path.mkdir(parents=True, exist_ok=True)
    try:
        for tensorrt_llm_config, weights in torch_to_tensorrt_llm_checkpoint(
            model=model,
            decoder_type=decoder_type,
            dtype=dtype,
            inference_tensor_parallel=inference_tensor_parallel,
            inference_pipeline_parallel=inference_pipeline_parallel,
            export_npz=export_npz,
            naive_fp8_quantization=naive_fp8_quantization,
            workspace_path=workspace_path,
        ):
            rank = tensorrt_llm_config["rank"]
            # For T5 model
            if tensorrt_llm_config["decoder"] in ["t5"]:
                export_dir = prepare_enc_dec_export_dir(tensorrt_llm_config, export_root)
                export_dir = Path(export_dir)
                export_dir.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                # We only export the json once across ranks as all jsons should be the same except for the rank.

                with open(export_dir / "config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

            # Hacky implementation for T5 for now
            new_weights = {}
            if tensorrt_llm_config["decoder"] == "t5":
                for key in weights.keys():
                    if key == "transformer.vocab_embedding.weight":
                        new_key = "embedding.vocab_embedding.weight"
                    elif key.startswith("transformer.layers"):
                        # For encoder
                        if tensorrt_llm_config["architecture"] == "EncoderModel":
                            new_key = key.replace("transformer.layers", "encoder_layers")
                        # For decoder
                        else:
                            new_key = key.replace("transformer.layers", "decoder_layers")
                    elif key == "transformer.ln_f.weight":
                        new_key = "final_layernorm.weight"
                    elif key == "lm_head.weight":
                        new_key = key
                    new_weights[new_key] = weights[key]
                weights = new_weights
            # End of hacky implementation for T5 for now

            if export_npz:
                weights_to_npz(weights, tensorrt_llm_config, export_dir)
            else:
                weights_path = export_dir / f"rank{rank}.safetensors"
                save_file(weights, weights_path)

    except Exception as e:
        fallback_model_path = export_dir / f"modelopt_model.{dist.rank()}.pth"
        torch.save(model.state_dict(), fallback_model_path)
        print(
            "Cannot export model to the model_config. The modelopt-optimized model state_dict"
            f" (including the quantization factors) is saved to {fallback_model_path} using"
            " torch.save for further inspection."
        )
        print(f"Detailed export error: {e}")
        traceback.print_exc()


def export_hf_checkpoint(
    model: nn.Module,
    dtype: torch.dtype = torch.float16,
    export_dir: Union[Path, str] = tempfile.gettempdir(),
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Exports the torch model to the packed checkpoint with original HF naming and save to the export_dir.

    Args:
        model: the torch model.
        dtype: the weights data type to export the unquantized layers.
        export_dir: the target export path.

    The packed checkpoint will be consumed by the TensorRT-LLM unified converter.
    """
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    for _, sub_module in model.model.layers.named_modules():
        if is_attention(sub_module):
            adjust_attn_amax_values(sub_module)

            # find kv cache quant format
            kv_cache_format = get_kv_cache_dtype(sub_module)

    for _, sub_module in model.model.layers.named_modules():
        if is_quantlinear(sub_module):

            quantization_format = get_quantization_format(sub_module)
            weight_scaling_factor = torch.tensor(
                sub_module.weight_quantizer.amax.item() / sub_module.weight_quantizer.maxbound
            )

            quantized_weight = to_quantized_weight(
                sub_module.weight.to(dtype),
                weight_scaling_factor,
                quantization_format,
            )

            sub_module.weight = nn.Parameter(quantized_weight, requires_grad=False)

    post_state_dict = model.state_dict()

    # Create the quantization config
    # TODO: add support for customized mixed precision config
    quant_config: Dict[str, Any] = {
        "producer": {
            "name": "modelopt",
            "version": __version__,
        },
        "quantization": {"quant_algo": None, "kv_cache_quant_algo": None},
    }

    if quantization_format == "fp8":
        quant_config["quantization"].update({"quant_algo": "FP8"})
    elif quantization_format == QUANTIZATION_NONE:
        quant_config["quantization"].update(
            {
                "quant_algo": None,
            }
        )
    else:
        quant_config["quantization"].update(
            {
                "quant_algo": quantization_format,
            }
        )
    if kv_cache_format is not None:
        quant_config["quantization"].update(
            {
                "kv_cache_quant_algo": kv_cache_format,
            }
        )

    return post_state_dict, quant_config
