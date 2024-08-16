"""
TensorRT-LLM Python bindings for C++ runtime
"""
from __future__ import annotations
import os
import torch
import typing
from . import BuildInfo
from . import executor
from . import tensor_names
__all__ = ['BF16', 'BOOL', 'BuildInfo', 'DataType', 'FLOAT', 'FP8', 'GptJsonConfig', 'GptManager', 'GptModelVariant', 'HALF', 'INT32', 'INT64', 'INT8', 'InferenceRequest', 'KvCacheConfig', 'LlmRequest', 'LlmRequestState', 'MemoryCounters', 'ModelConfig', 'MpiComm', 'NamedTensor', 'PeftCacheManagerConfig', 'QuantMode', 'SamplingConfig', 'TrtGptModelOptionalParams', 'TrtGptModelType', 'UINT8', 'WorldConfig', 'executor', 'tensor_names']
class DataType:
    """
    Members:
    
      FLOAT
    
      HALF
    
      INT8
    
      INT32
    
      BOOL
    
      UINT8
    
      FP8
    
      BF16
    
      INT64
    """
    BF16: typing.ClassVar[DataType]  # value = <DataType.BF16: 7>
    BOOL: typing.ClassVar[DataType]  # value = <DataType.BOOL: 4>
    FLOAT: typing.ClassVar[DataType]  # value = <DataType.FLOAT: 0>
    FP8: typing.ClassVar[DataType]  # value = <DataType.FP8: 6>
    HALF: typing.ClassVar[DataType]  # value = <DataType.HALF: 1>
    INT32: typing.ClassVar[DataType]  # value = <DataType.INT32: 3>
    INT64: typing.ClassVar[DataType]  # value = <DataType.INT64: 8>
    INT8: typing.ClassVar[DataType]  # value = <DataType.INT8: 2>
    UINT8: typing.ClassVar[DataType]  # value = <DataType.UINT8: 5>
    __members__: typing.ClassVar[dict[str, DataType]]  # value = {'FLOAT': <DataType.FLOAT: 0>, 'HALF': <DataType.HALF: 1>, 'INT8': <DataType.INT8: 2>, 'INT32': <DataType.INT32: 3>, 'BOOL': <DataType.BOOL: 4>, 'UINT8': <DataType.UINT8: 5>, 'FP8': <DataType.FP8: 6>, 'BF16': <DataType.BF16: 7>, 'INT64': <DataType.INT64: 8>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class GptJsonConfig:
    @staticmethod
    def parse(json: str) -> GptJsonConfig:
        ...
    @staticmethod
    def parse_file(path: os.PathLike) -> GptJsonConfig:
        ...
    def __init__(self, name: str, version: str, precision: str, tensor_parallelism: int, pipeline_parallelism: int, gpus_per_node: int, model_config: ModelConfig) -> None:
        ...
    @typing.overload
    def engine_filename(self, world_config: WorldConfig, model: str) -> str:
        ...
    @typing.overload
    def engine_filename(self, world_config: WorldConfig) -> str:
        ...
    @property
    def gpus_per_node(self) -> int:
        ...
    @property
    def model_config(self) -> ModelConfig:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def pipeline_parallelism(self) -> int:
        ...
    @property
    def precision(self) -> str:
        ...
    @property
    def tensor_parallelism(self) -> int:
        ...
    @property
    def version(self) -> str:
        ...
    @property
    def world_size(self) -> int:
        ...
class GptManager:
    def __enter__(self) -> typing.Any:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    def __init__(self, trt_engine_path: os.PathLike, model_type: TrtGptModelType, get_inference_requests_cb: typing.Callable[[int], list[InferenceRequest]], send_response_cb: typing.Callable[[int, list[NamedTensor], bool, str], None], poll_stop_signal_cb: typing.Callable[[], set[int]] = None, return_batch_manager_stats_cb: typing.Callable[[str], None] = None, optional_params: TrtGptModelOptionalParams = ..., terminate_req_id: int | None = None) -> None:
        ...
    def shutdown(self) -> None:
        ...
class GptModelVariant:
    """
    Members:
    
      GPT
    
      GLM
    
      CHATGLM
    
      MAMBA
    
      RECURRENTGEMMA
    """
    CHATGLM: typing.ClassVar[GptModelVariant]  # value = <GptModelVariant.CHATGLM: 1>
    GLM: typing.ClassVar[GptModelVariant]  # value = <GptModelVariant.GLM: 2>
    GPT: typing.ClassVar[GptModelVariant]  # value = <GptModelVariant.GPT: 0>
    MAMBA: typing.ClassVar[GptModelVariant]  # value = <GptModelVariant.MAMBA: 3>
    RECURRENTGEMMA: typing.ClassVar[GptModelVariant]  # value = <GptModelVariant.RECURRENTGEMMA: 4>
    __members__: typing.ClassVar[dict[str, GptModelVariant]]  # value = {'GPT': <GptModelVariant.GPT: 0>, 'GLM': <GptModelVariant.GLM: 2>, 'CHATGLM': <GptModelVariant.CHATGLM: 1>, 'MAMBA': <GptModelVariant.MAMBA: 3>, 'RECURRENTGEMMA': <GptModelVariant.RECURRENTGEMMA: 4>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class InferenceRequest:
    bad_words_list: torch.Tensor
    beam_width: torch.Tensor
    draft_input_ids: torch.Tensor
    draft_logits: torch.Tensor
    early_stopping: torch.Tensor
    embedding_bias: torch.Tensor
    end_id: torch.Tensor
    frequency_penalty: torch.Tensor
    input_ids: torch.Tensor
    is_streaming: bool
    length_penalty: torch.Tensor
    lora_config: torch.Tensor
    lora_task_id: torch.Tensor
    lora_weights: torch.Tensor
    max_new_tokens: torch.Tensor
    min_length: torch.Tensor
    no_repeat_ngram_size: torch.Tensor
    pad_id: torch.Tensor
    presence_penalty: torch.Tensor
    prompt_embedding_table: torch.Tensor
    prompt_vocab_size: torch.Tensor
    random_seed: torch.Tensor
    repetition_penalty: torch.Tensor
    return_context_logits: torch.Tensor
    return_generation_logits: torch.Tensor
    return_log_probs: torch.Tensor
    runtime_top_k: torch.Tensor
    runtime_top_p: torch.Tensor
    stop_words_list: torch.Tensor
    temperature: torch.Tensor
    def __getstate__(self) -> bytearray:
        ...
    @typing.overload
    def __init__(self, request_id: int, logits_post_processor_callback: typing.Callable[[int, torch.Tensor, list[list[int]], torch.Stream, int | None], None] | None = None) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: int, arg1: dict[str, torch.Tensor]) -> None:
        """
        deprecated: use direct tensor access instead
        """
    def __setstate__(self, arg0: bytearray) -> None:
        ...
    @property
    def request_id(self) -> int:
        ...
class KvCacheConfig:
    __hash__: typing.ClassVar[None] = None
    enable_block_reuse: bool
    free_gpu_memory_fraction: float | None
    max_attention_window: int | None
    max_tokens: int | None
    sink_token_length: int | None
    def __eq__(self, arg0: KvCacheConfig) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, max_tokens: int | None = None, max_attention_window: int | None = None, sink_token_length: int | None = None, free_gpu_memory_fraction: float | None = None, enable_block_reuse: bool = False) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class LlmRequest:
    context_chunk_size: int
    draft_tokens: list[int]
    end_id: int | None
    is_streaming: bool
    max_new_tokens: int
    max_sent_token_len: int
    pad_id: int | None
    prompt_len: int
    request_id: int
    sampling_config: SamplingConfig
    seq_slot: int | None
    state: LlmRequestState
    def __init__(self, request_id: int, max_new_tokens: int, input_tokens: list[int], sampling_config: SamplingConfig, is_streaming: bool, end_id: int | None = None, pad_id: int | None = None, embedding_bias: torch.Tensor | None = None, bad_words_list: torch.Tensor | None = None, stop_words_list: torch.Tensor | None = None, prompt_embedding_table: torch.Tensor | None = None, prompt_vocab_size: int | None = None, lora_task_id: int | None = None, lora_weights: torch.Tensor | None = None, lora_config: torch.Tensor | None = None, return_log_probs: bool = False, return_context_logits: bool = False, return_generation_logits: bool = False, draft_tokens: list[int] | None = None, draft_logits: torch.Tensor | None = None, exclude_input_from_output: bool = False, logits_post_processor: typing.Callable[[int, torch.Tensor, list[list[int]], torch.Stream, int | None], None] | None = None, apply_logits_post_processor_batched: bool = False, encoder_input_tokens: list[int] | None = None, return_encoder_output: bool = False, client_id: int | None = None, priority: float = 0.5) -> None:
        ...
    def add_new_token(self, token: int, beam: int) -> None:
        ...
    def add_new_tokens(self, beam_tokens: list[int]) -> None:
        ...
    def get_context_remaining_length(self) -> int:
        ...
    def get_log_probs(self, arg0: int) -> list[float]:
        ...
    def get_num_tokens(self, beam: int) -> int:
        ...
    def get_return_encoder_output(self) -> bool:
        ...
    def get_token(self, beam: int, pos: int) -> int:
        ...
    @typing.overload
    def get_tokens(self, beam: int) -> list[int]:
        ...
    @typing.overload
    def get_tokens(self) -> list[list[int]]:
        ...
    def has_draft_tokens(self) -> bool:
        ...
    def is_first_context_chunk(self) -> bool:
        ...
    def is_full_context_request(self) -> bool:
        ...
    def is_last_context_chunk(self) -> bool:
        ...
    def move_to_next_context_chunk(self) -> None:
        ...
    def pause(self, max_input_len: int) -> None:
        ...
    def priority(self) -> float:
        ...
    def set_cum_log_prob(self, cum_log_prob: float, beam: int) -> None:
        ...
    def set_generated_tokens(self, generated_beam_tokens: list[list[int]]) -> None:
        ...
    def set_log_probs(self, log_probs: list[float], beam: int) -> None:
        ...
    def set_priority(self, arg0: float) -> None:
        ...
    def set_return_encoder_output(self, return_encoder_output: bool) -> None:
        ...
    @property
    def bad_words_list(self) -> torch.Tensor | None:
        ...
    @property
    def context_current_position(self) -> int:
        ...
    @property
    def cum_log_probs(self) -> list[float]:
        ...
    @property
    def draft_logits(self) -> torch.Tensor | None:
        ...
    @draft_logits.setter
    def draft_logits(self, arg1: torch.Tensor) -> None:
        ...
    @property
    def embedding_bias(self) -> torch.Tensor | None:
        ...
    @property
    def log_probs(self) -> list[list[float]]:
        ...
    @property
    def lora_config(self) -> torch.Tensor | None:
        ...
    @property
    def lora_task_id(self) -> int | None:
        ...
    @property
    def lora_weights(self) -> torch.Tensor | None:
        ...
    @property
    def max_beam_num_tokens(self) -> int:
        ...
    @property
    def max_num_generated_tokens(self) -> int:
        ...
    @property
    def orig_prompt_len(self) -> int:
        ...
    @property
    def prompt_embedding_table(self) -> torch.Tensor | None:
        ...
    @property
    def prompt_vocab_size(self) -> int | None:
        ...
    @property
    def return_context_logits(self, arg1: bool) -> None:
        ...
    @property
    def return_generation_logits(self, arg1: bool) -> None:
        ...
    @property
    def return_log_probs(self) -> bool:
        ...
    @property
    def stop_words_list(self) -> torch.Tensor | None:
        ...
class LlmRequestState:
    """
    Members:
    
      REQUEST_STATE_UNKNOWN
    
      REQUEST_STATE_ENCODER_INIT
    
      REQUEST_STATE_CONTEXT_INIT
    
      REQUEST_STATE_GENERATION_IN_PROGRESS
    
      REQUEST_STATE_GENERATION_TO_COMPLETE
    
      REQUEST_STATE_GENERATION_COMPLETE
    """
    REQUEST_STATE_CONTEXT_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_CONTEXT_INIT: 2>
    REQUEST_STATE_ENCODER_INIT: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_ENCODER_INIT: 1>
    REQUEST_STATE_GENERATION_COMPLETE: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_GENERATION_COMPLETE: 5>
    REQUEST_STATE_GENERATION_IN_PROGRESS: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_GENERATION_IN_PROGRESS: 3>
    REQUEST_STATE_GENERATION_TO_COMPLETE: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_GENERATION_TO_COMPLETE: 4>
    REQUEST_STATE_UNKNOWN: typing.ClassVar[LlmRequestState]  # value = <LlmRequestState.REQUEST_STATE_UNKNOWN: 0>
    __members__: typing.ClassVar[dict[str, LlmRequestState]]  # value = {'REQUEST_STATE_UNKNOWN': <LlmRequestState.REQUEST_STATE_UNKNOWN: 0>, 'REQUEST_STATE_ENCODER_INIT': <LlmRequestState.REQUEST_STATE_ENCODER_INIT: 1>, 'REQUEST_STATE_CONTEXT_INIT': <LlmRequestState.REQUEST_STATE_CONTEXT_INIT: 2>, 'REQUEST_STATE_GENERATION_IN_PROGRESS': <LlmRequestState.REQUEST_STATE_GENERATION_IN_PROGRESS: 3>, 'REQUEST_STATE_GENERATION_TO_COMPLETE': <LlmRequestState.REQUEST_STATE_GENERATION_TO_COMPLETE: 4>, 'REQUEST_STATE_GENERATION_COMPLETE': <LlmRequestState.REQUEST_STATE_GENERATION_COMPLETE: 5>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class MemoryCounters:
    @staticmethod
    def instance() -> MemoryCounters:
        ...
    @property
    def cpu(self) -> int:
        ...
    @property
    def gpu(self) -> int:
        ...
    @property
    def pinned(self) -> int:
        ...
    @property
    def uvm(self) -> int:
        ...
class ModelConfig:
    compute_context_logits: bool
    compute_generation_logits: bool
    head_size: int
    max_batch_size: int
    max_beam_width: int
    max_input_len: int
    max_num_tokens: int | None
    max_prompt_embedding_table_size: int
    model_variant: GptModelVariant
    num_kv_heads: int
    quant_mode: QuantMode
    tokens_per_block: int
    use_gpt_attention_plugin: bool
    use_packed_input: bool
    use_paged_kv_cache: bool
    def __init__(self, vocab_size: int, num_attention_layers: int, num_rnn_layers: int, num_heads: int, hidden_size: int, data_type: DataType) -> None:
        ...
    def num_attention_layers(self, pipeline_parallelism: int = 1) -> int:
        ...
    def num_rnn_layers(self, pipeline_parallelism: int = 1) -> int:
        ...
    def vocab_size_padded(self, world_size: int) -> int:
        ...
    @property
    def data_type(self) -> DataType:
        ...
    @property
    def hidden_size(self) -> int:
        ...
    @property
    def max_seq_len(self) -> int:
        ...
    @max_seq_len.setter
    def max_seq_len(self) -> int:
        ...
    @property
    def num_heads(self) -> int:
        ...
    @property
    def size_per_head(self) -> int:
        ...
    @property
    def supports_inflight_batching(self) -> bool:
        ...
    @property
    def use_prompt_tuning(self) -> bool:
        ...
    @property
    def vocab_size(self) -> int:
        ...
class MpiComm:
    @staticmethod
    def local_init() -> None:
        ...
    @staticmethod
    def local_size() -> int:
        ...
    @staticmethod
    def rank() -> int:
        ...
    @staticmethod
    def size() -> int:
        ...
    @staticmethod
    def split(arg0: int, arg1: int) -> None:
        ...
class NamedTensor:
    tensor: torch.Tensor | None
    def __init__(self, tensor: torch.Tensor | None, name: str) -> None:
        ...
    @property
    def name(self) -> str:
        ...
class PeftCacheManagerConfig:
    device_cache_percent: float | None
    host_cache_size: int | None
    max_adapter_size: int
    max_pages_per_block_device: int
    max_pages_per_block_host: int
    num_copy_streams: int
    num_device_module_layer: int
    num_ensure_workers: int
    num_host_module_layer: int
    num_put_workers: int
    optimal_adapter_size: int
    def __init__(self, num_host_module_layer: int = 0, num_device_module_layer: int = 0, optimal_adapter_size: int = 8, max_adapter_size: int = 64, num_put_workers: int = 1, num_ensure_workers: int = 1, num_copy_streams: int = 1, max_pages_per_block_host: int = 24, max_pages_per_block_device: int = 8, device_cache_percent: float | None = None, host_cache_size: int | None = None) -> None:
        ...
class QuantMode:
    __hash__: typing.ClassVar[None] = None
    @staticmethod
    def activations() -> QuantMode:
        ...
    @staticmethod
    def fp8_kv_cache() -> QuantMode:
        ...
    @staticmethod
    def fp8_qdq() -> QuantMode:
        ...
    @staticmethod
    def from_description(quantize_weights: bool = False, quantize_activations: bool = False, per_token: bool = False, per_channel: bool = False, per_group: bool = False, use_int4_weights: bool = False, use_int8_kv_cache: bool = False, use_fp8_kv_kache: bool = False, use_fp8_qdq: bool = False, use_fp8_rowwise: bool = False) -> QuantMode:
        ...
    @staticmethod
    def from_quant_algo(quant_algo: str | None = None, kv_cache_quant_algo: str | None = None) -> QuantMode:
        ...
    @staticmethod
    def int4_weights() -> QuantMode:
        ...
    @staticmethod
    def int8_kv_cache() -> QuantMode:
        ...
    @staticmethod
    def int8_weights() -> QuantMode:
        ...
    @staticmethod
    def none() -> QuantMode:
        ...
    @staticmethod
    def per_channel_scaling() -> QuantMode:
        ...
    @staticmethod
    def per_group_scaling() -> QuantMode:
        ...
    @staticmethod
    def per_token_scaling() -> QuantMode:
        ...
    @staticmethod
    def use_smooth_quant(per_token: bool = False, per_channel: bool = False) -> QuantMode:
        ...
    @staticmethod
    def use_weight_only(use_int4_weights: bool = False, per_group: bool = False) -> QuantMode:
        ...
    def __add__(self, arg0: QuantMode) -> QuantMode:
        ...
    def __eq__(self, arg0: QuantMode) -> bool:
        ...
    def __iadd__(self, arg0: QuantMode) -> QuantMode:
        ...
    def __isub__(self, arg0: QuantMode) -> QuantMode:
        ...
    def __ne__(self, arg0: QuantMode) -> bool:
        ...
    def __sub__(self, arg0: QuantMode) -> QuantMode:
        ...
    def is_set(self, mode: QuantMode) -> bool:
        ...
    @property
    def has_activations(self) -> bool:
        ...
    @property
    def has_fp8_kv_cache(self) -> bool:
        ...
    @property
    def has_fp8_qdq(self) -> bool:
        ...
    @property
    def has_int4_weights(self) -> bool:
        ...
    @property
    def has_int8_kv_cache(self) -> bool:
        ...
    @property
    def has_int8_weights(self) -> bool:
        ...
    @property
    def has_kv_cache_quant(self) -> bool:
        ...
    @property
    def has_per_channel_scaling(self) -> bool:
        ...
    @property
    def has_per_group_scaling(self) -> bool:
        ...
    @property
    def has_per_token_scaling(self) -> bool:
        ...
    @property
    def has_static_activation_scaling(self) -> bool:
        ...
    @property
    def value(self) -> int:
        ...
class SamplingConfig:
    __hash__: typing.ClassVar[None] = None
    beam_search_diversity_rate: list[float] | None
    beam_width: int
    early_stopping: list[int] | None
    frequency_penalty: list[float] | None
    length_penalty: list[float] | None
    min_length: list[int] | None
    no_repeat_ngram_size: list[int] | None
    presence_penalty: list[float] | None
    random_seed: list[int] | None
    repetition_penalty: list[float] | None
    temperature: list[float] | None
    top_k: list[int] | None
    top_p: list[float] | None
    top_p_decay: list[float] | None
    top_p_min: list[float] | None
    top_p_reset_ids: list[int] | None
    def __eq__(self, arg0: SamplingConfig) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, beam_width: int = 1) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class TrtGptModelOptionalParams:
    __hash__: typing.ClassVar[None] = None
    decoding_config: executor.DecodingConfig
    device_ids: list[int] | None
    enable_chunked_context: bool
    enable_trt_overlap: bool
    gpu_weights_percent: float
    kv_cache_config: KvCacheConfig
    max_beam_width: int | None
    normalize_log_probs: bool
    scheduler_config: executor.SchedulerConfig
    def __eq__(self, arg0: TrtGptModelOptionalParams) -> bool:
        ...
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, kv_cache_config: KvCacheConfig = ..., enable_trt_overlap: bool = False, device_ids: list[int] | None = None, normalize_log_probs: bool = True, enable_chunked_context: bool = False, peft_cache_manager_config: PeftCacheManagerConfig = ...) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class TrtGptModelType:
    """
    Members:
    
      V1
    
      InflightBatching
    
      InflightFusedBatching
    """
    InflightBatching: typing.ClassVar[TrtGptModelType]  # value = <TrtGptModelType.InflightBatching: 1>
    InflightFusedBatching: typing.ClassVar[TrtGptModelType]  # value = <TrtGptModelType.InflightFusedBatching: 2>
    V1: typing.ClassVar[TrtGptModelType]  # value = <TrtGptModelType.V1: 0>
    __members__: typing.ClassVar[dict[str, TrtGptModelType]]  # value = {'V1': <TrtGptModelType.V1: 0>, 'InflightBatching': <TrtGptModelType.InflightBatching: 1>, 'InflightFusedBatching': <TrtGptModelType.InflightFusedBatching: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class WorldConfig:
    @staticmethod
    def mpi(gpus_per_node: int = 8, tensor_parallelism: int | None = None, pipeline_parallelism: int | None = None, device_ids: list[int] | None = None) -> WorldConfig:
        ...
    def __init__(self, tensor_parallelism: int = 1, pipeline_parallelism: int = 1, rank: int = 0, gpus_per_node: int = 8, device_ids: list[int] | None = None) -> None:
        ...
    @property
    def device(self) -> int:
        ...
    @property
    def gpus_per_group(self) -> int:
        ...
    @property
    def gpus_per_node(self) -> int:
        ...
    @property
    def is_pipeline_parallel(self) -> bool:
        ...
    @property
    def is_tensor_parallel(self) -> bool:
        ...
    @property
    def local_rank(self) -> int:
        ...
    @property
    def node_rank(self) -> int:
        ...
    @property
    def pipeline_parallel_rank(self) -> int:
        ...
    @property
    def pipeline_parallelism(self) -> int:
        ...
    @property
    def rank(self) -> int:
        ...
    @property
    def size(self) -> int:
        ...
    @property
    def tensor_parallel_rank(self) -> int:
        ...
    @property
    def tensor_parallelism(self) -> int:
        ...
BF16: DataType  # value = <DataType.BF16: 7>
BOOL: DataType  # value = <DataType.BOOL: 4>
FLOAT: DataType  # value = <DataType.FLOAT: 0>
FP8: DataType  # value = <DataType.FP8: 6>
HALF: DataType  # value = <DataType.HALF: 1>
INT32: DataType  # value = <DataType.INT32: 3>
INT64: DataType  # value = <DataType.INT64: 8>
INT8: DataType  # value = <DataType.INT8: 2>
UINT8: DataType  # value = <DataType.UINT8: 5>
