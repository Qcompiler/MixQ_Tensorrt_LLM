"""
Executor bindings
"""
from __future__ import annotations
import datetime
import os
import torch
import typing
__all__ = ['BatchingType', 'CapacitySchedulerPolicy', 'CommunicationMode', 'CommunicationType', 'ContextChunkingPolicy', 'DecodingConfig', 'DecodingMode', 'Executor', 'ExecutorConfig', 'ExtendedRuntimePerfKnobConfig', 'ExternalDraftTokensConfig', 'InflightBatchingStats', 'IterationStats', 'KvCacheConfig', 'KvCacheStats', 'LookaheadDecodingConfig', 'LoraConfig', 'ModelType', 'OrchestratorConfig', 'OutputConfig', 'ParallelConfig', 'PeftCacheConfig', 'PromptTuningConfig', 'Request', 'RequestStage', 'RequestStats', 'RequestStatsPerIteration', 'Response', 'Result', 'SamplingConfig', 'SchedulerConfig', 'StaticBatchingStats']
class BatchingType:
    """
    Members:
    
      STATIC
    
      INFLIGHT
    """
    INFLIGHT: typing.ClassVar[BatchingType]  # value = <BatchingType.INFLIGHT: 1>
    STATIC: typing.ClassVar[BatchingType]  # value = <BatchingType.STATIC: 0>
    __members__: typing.ClassVar[dict[str, BatchingType]]  # value = {'STATIC': <BatchingType.STATIC: 0>, 'INFLIGHT': <BatchingType.INFLIGHT: 1>}
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
class CapacitySchedulerPolicy:
    """
    Members:
    
      MAX_UTILIZATION
    
      GUARANTEED_NO_EVICT
    """
    GUARANTEED_NO_EVICT: typing.ClassVar[CapacitySchedulerPolicy]  # value = <CapacitySchedulerPolicy.GUARANTEED_NO_EVICT: 1>
    MAX_UTILIZATION: typing.ClassVar[CapacitySchedulerPolicy]  # value = <CapacitySchedulerPolicy.MAX_UTILIZATION: 0>
    __members__: typing.ClassVar[dict[str, CapacitySchedulerPolicy]]  # value = {'MAX_UTILIZATION': <CapacitySchedulerPolicy.MAX_UTILIZATION: 0>, 'GUARANTEED_NO_EVICT': <CapacitySchedulerPolicy.GUARANTEED_NO_EVICT: 1>}
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
class CommunicationMode:
    """
    Members:
    
      LEADER
    
      ORCHESTRATOR
    """
    LEADER: typing.ClassVar[CommunicationMode]  # value = <CommunicationMode.LEADER: 0>
    ORCHESTRATOR: typing.ClassVar[CommunicationMode]  # value = <CommunicationMode.ORCHESTRATOR: 1>
    __members__: typing.ClassVar[dict[str, CommunicationMode]]  # value = {'LEADER': <CommunicationMode.LEADER: 0>, 'ORCHESTRATOR': <CommunicationMode.ORCHESTRATOR: 1>}
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
class CommunicationType:
    """
    Members:
    
      MPI
    """
    MPI: typing.ClassVar[CommunicationType]  # value = <CommunicationType.MPI: 0>
    __members__: typing.ClassVar[dict[str, CommunicationType]]  # value = {'MPI': <CommunicationType.MPI: 0>}
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
class ContextChunkingPolicy:
    """
    Members:
    
      EQUAL_PROGRESS
    
      FIRST_COME_FIRST_SERVED
    """
    EQUAL_PROGRESS: typing.ClassVar[ContextChunkingPolicy]  # value = <ContextChunkingPolicy.EQUAL_PROGRESS: 1>
    FIRST_COME_FIRST_SERVED: typing.ClassVar[ContextChunkingPolicy]  # value = <ContextChunkingPolicy.FIRST_COME_FIRST_SERVED: 0>
    __members__: typing.ClassVar[dict[str, ContextChunkingPolicy]]  # value = {'EQUAL_PROGRESS': <ContextChunkingPolicy.EQUAL_PROGRESS: 1>, 'FIRST_COME_FIRST_SERVED': <ContextChunkingPolicy.FIRST_COME_FIRST_SERVED: 0>}
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
class DecodingConfig:
    def __init__(self, decoding_mode: DecodingMode | None = None, lookahead_decoding_config: LookaheadDecodingConfig | None = None, medusa_choices: list[list[int]] | None = None) -> None:
        ...
    @property
    def decoding_mode(self) -> DecodingMode | None:
        ...
    @decoding_mode.setter
    def decoding_mode(self, arg1: DecodingMode) -> None:
        ...
    @property
    def lookahead_decoding_config(self) -> LookaheadDecodingConfig | None:
        ...
    @lookahead_decoding_config.setter
    def lookahead_decoding_config(self, arg1: LookaheadDecodingConfig) -> None:
        ...
    @property
    def medusa_choices(self) -> list[list[int]] | None:
        ...
    @medusa_choices.setter
    def medusa_choices(self, arg1: list[list[int]]) -> None:
        ...
class DecodingMode:
    @staticmethod
    def Auto() -> DecodingMode:
        ...
    @staticmethod
    def BeamSearch() -> DecodingMode:
        ...
    @staticmethod
    def Lookahead() -> DecodingMode:
        ...
    @staticmethod
    def Medusa() -> DecodingMode:
        ...
    @staticmethod
    def TopK() -> DecodingMode:
        ...
    @staticmethod
    def TopKTopP() -> DecodingMode:
        ...
    @staticmethod
    def TopP() -> DecodingMode:
        ...
    def isAuto(self) -> bool:
        ...
    def isBeamSearch(self) -> bool:
        ...
    def isLookahead(self) -> bool:
        ...
    def isMedusa(self) -> bool:
        ...
    def isTopK(self) -> bool:
        ...
    def isTopKandTopP(self) -> bool:
        ...
    def isTopKorTopP(self) -> bool:
        ...
    def isTopP(self) -> bool:
        ...
class Executor:
    def __enter__(self) -> typing.Any:
        ...
    def __exit__(self, arg0: typing.Any, arg1: typing.Any, arg2: typing.Any) -> None:
        ...
    @typing.overload
    def __init__(self, model_path: os.PathLike, model_type: ModelType, executor_config: ExecutorConfig) -> None:
        ...
    @typing.overload
    def __init__(self, encoder_model_path: os.PathLike, decoder_model_path: os.PathLike, model_type: ModelType, executor_config: ExecutorConfig) -> None:
        ...
    @typing.overload
    def __init__(self, engine_buffer: str, json_config_str: str, model_type: ModelType, executor_config: ExecutorConfig) -> None:
        ...
    @typing.overload
    def __init__(self, encoder_engine_buffer: str, encoder_json_config_str: str, decoder_engine_buffer: str, decoder_json_config_str: str, model_type: ModelType, executor_config: ExecutorConfig) -> None:
        ...
    @typing.overload
    def await_responses(self, timeout: datetime.timedelta | None = None) -> list[Response]:
        ...
    @typing.overload
    def await_responses(self, id: int, timeout: datetime.timedelta | None = None) -> list[Response]:
        ...
    @typing.overload
    def await_responses(self, ids: list[int], timeout: datetime.timedelta | None = None) -> list[list[Response]]:
        ...
    def can_enqueue_requests(self) -> bool:
        ...
    def cancel_request(self, id: int = None) -> None:
        ...
    def enqueue_request(self, request: Request) -> int:
        ...
    def enqueue_requests(self, requests: list[Request]) -> list[int]:
        ...
    def get_latest_iteration_stats(self) -> list[IterationStats]:
        ...
    def get_latest_request_stats(self) -> list[RequestStatsPerIteration]:
        ...
    def get_num_responses_ready(self, id: int | None = None) -> int:
        ...
    def shutdown(self) -> None:
        ...
class ExecutorConfig:
    batching_type: BatchingType
    enable_chunked_context: bool
    extended_runtime_perf_knob_config: ExtendedRuntimePerfKnobConfig
    gpu_weights_percent: float
    iter_stats_max_iterations: int
    kv_cache_config: KvCacheConfig
    max_beam_width: int
    max_queue_size: int | None
    normalize_log_probs: bool
    request_stats_max_iterations: int
    scheduler_config: SchedulerConfig
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, max_beam_width: int = 1, scheduler_config: SchedulerConfig = ..., kv_cache_config: KvCacheConfig = ..., enable_chunked_context: bool = False, normalize_log_probs: bool = True, iter_stats_max_iterations: int = 1000, request_stats_max_iterations: int = 0, batching_type: BatchingType = ..., max_batch_size: int | None = None, max_num_tokens: int | None = None, parallel_config: ParallelConfig | None = None, peft_cache_config: PeftCacheConfig = ..., logits_post_processor_map: dict[str, typing.Callable[[int, torch.Tensor, list[list[int]], int, int | None], None]] | None = None, logits_post_processor_batched: typing.Callable[[list[int], list[torch.Tensor], list[list[list[int]]], int, list[int | None]], None] | None = None, decoding_config: DecodingConfig | None = None, gpu_weights_percent: float = 1.0, max_queue_size: int | None = None, extended_runtime_perf_knob_config: ExtendedRuntimePerfKnobConfig = ...) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def decoding_config(self) -> DecodingConfig | None:
        ...
    @decoding_config.setter
    def decoding_config(self, arg1: DecodingConfig) -> None:
        ...
    @property
    def logits_post_processor_batched(self) -> typing.Callable[[list[int], list[torch.Tensor], list[list[list[int]]], int, list[int | None]], None] | None:
        ...
    @logits_post_processor_batched.setter
    def logits_post_processor_batched(self, arg1: typing.Callable[[list[int], list[torch.Tensor], list[list[list[int]]], int, list[int | None]], None]) -> None:
        ...
    @property
    def logits_post_processor_map(self) -> dict[str, typing.Callable[[int, torch.Tensor, list[list[int]], int, int | None], None]] | None:
        ...
    @logits_post_processor_map.setter
    def logits_post_processor_map(self, arg1: dict[str, typing.Callable[[int, torch.Tensor, list[list[int]], int, int | None], None]]) -> None:
        ...
    @property
    def max_batch_size(self) -> int | None:
        ...
    @max_batch_size.setter
    def max_batch_size(self, arg1: int) -> None:
        ...
    @property
    def max_num_tokens(self) -> int | None:
        ...
    @max_num_tokens.setter
    def max_num_tokens(self, arg1: int) -> None:
        ...
    @property
    def parallel_config(self) -> ParallelConfig | None:
        ...
    @parallel_config.setter
    def parallel_config(self, arg1: ParallelConfig) -> None:
        ...
    @property
    def peft_cache_config(self) -> PeftCacheConfig | None:
        ...
    @peft_cache_config.setter
    def peft_cache_config(self, arg1: PeftCacheConfig) -> None:
        ...
class ExtendedRuntimePerfKnobConfig:
    enable_context_fmha_fp32_acc: bool
    multi_block_mode: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, multi_block_mode: bool = False, enable_context_fmha_fp32_acc: bool = False) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
class ExternalDraftTokensConfig:
    def __init__(self, tokens: list[int], logits: torch.Tensor | None = None, acceptance_threshold: float | None = None) -> None:
        ...
    @property
    def acceptance_threshold(self) -> float | None:
        ...
    @property
    def logits(self) -> torch.Tensor | None:
        ...
    @property
    def tokens(self) -> list[int]:
        ...
class InflightBatchingStats:
    avg_num_decoded_tokens_per_iter: float
    micro_batch_id: int
    num_context_requests: int
    num_ctx_tokens: int
    num_gen_requests: int
    num_paused_requests: int
    num_scheduled_requests: int
    def __init__(self) -> None:
        ...
class IterationStats:
    cpu_mem_usage: int
    gpu_mem_usage: int
    inflight_batching_stats: InflightBatchingStats | None
    iter: int
    iter_latency_ms: float
    kv_cache_stats: KvCacheStats | None
    max_num_active_requests: int
    num_active_requests: int
    num_queued_requests: int
    pinned_mem_usage: int
    static_batching_stats: StaticBatchingStats | None
    timestamp: str
    def __init__(self) -> None:
        ...
    def to_json_str(self) -> str:
        ...
class KvCacheConfig:
    enable_block_reuse: bool
    onboard_blocks: bool
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, enable_block_reuse: bool = False, max_tokens: int | None = None, max_attention_window: int | None = None, sink_token_length: int | None = None, free_gpu_memory_fraction: float | None = None, host_cache_size: int | None = None, onboard_blocks: bool = True) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def free_gpu_memory_fraction(self) -> float | None:
        ...
    @free_gpu_memory_fraction.setter
    def free_gpu_memory_fraction(self, arg1: float) -> None:
        ...
    @property
    def host_cache_size(self) -> int | None:
        ...
    @host_cache_size.setter
    def host_cache_size(self, arg1: int) -> None:
        ...
    @property
    def max_attention_window(self) -> int | None:
        ...
    @max_attention_window.setter
    def max_attention_window(self, arg1: int) -> None:
        ...
    @property
    def max_tokens(self) -> int | None:
        ...
    @max_tokens.setter
    def max_tokens(self, arg1: int) -> None:
        ...
    @property
    def sink_token_length(self) -> int | None:
        ...
    @sink_token_length.setter
    def sink_token_length(self, arg1: int) -> None:
        ...
class KvCacheStats:
    alloc_new_blocks: int
    alloc_total_blocks: int
    free_num_blocks: int
    max_num_blocks: int
    reused_blocks: int
    tokens_per_block: int
    used_num_blocks: int
    def __init__(self) -> None:
        ...
class LookaheadDecodingConfig:
    def __init__(self, max_window_size: int, max_ngram_size: int, max_verification_set_size: int) -> None:
        ...
    @property
    def max_ngram_size(self) -> int:
        ...
    @property
    def max_verification_set_size(self) -> int:
        ...
    @property
    def max_window_size(self) -> int:
        ...
class LoraConfig:
    def __init__(self, task_id: int, weights: torch.Tensor | None = None, config: torch.Tensor | None = None) -> None:
        ...
    @property
    def config(self) -> torch.Tensor | None:
        ...
    @property
    def task_id(self) -> int:
        ...
    @property
    def weights(self) -> torch.Tensor | None:
        ...
class ModelType:
    """
    Members:
    
      DECODER_ONLY
    
      ENCODER_ONLY
    
      ENCODER_DECODER
    """
    DECODER_ONLY: typing.ClassVar[ModelType]  # value = <ModelType.DECODER_ONLY: 0>
    ENCODER_DECODER: typing.ClassVar[ModelType]  # value = <ModelType.ENCODER_DECODER: 2>
    ENCODER_ONLY: typing.ClassVar[ModelType]  # value = <ModelType.ENCODER_ONLY: 1>
    __members__: typing.ClassVar[dict[str, ModelType]]  # value = {'DECODER_ONLY': <ModelType.DECODER_ONLY: 0>, 'ENCODER_ONLY': <ModelType.ENCODER_ONLY: 1>, 'ENCODER_DECODER': <ModelType.ENCODER_DECODER: 2>}
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
class OrchestratorConfig:
    is_orchestrator: bool
    worker_executable_path: str
    def __init__(self, is_orchestrator: bool = True, worker_executable_path: str = '') -> None:
        ...
class OutputConfig:
    exclude_input_from_output: bool
    return_context_logits: bool
    return_encoder_output: bool
    return_generation_logits: bool
    return_log_probs: bool
    def __init__(self, return_log_probs: bool = False, return_context_logits: bool = False, return_generation_logits: bool = False, exclude_input_from_output: bool = False, return_encoder_output: bool = False) -> None:
        ...
class ParallelConfig:
    communication_mode: CommunicationMode
    communication_type: CommunicationType
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, communication_type: CommunicationType = ..., communication_mode: CommunicationMode = ..., device_ids: list[int] | None = None, participant_ids: list[int] | None = None, orchestrator_config: OrchestratorConfig | None = None) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def device_ids(self) -> list[int] | None:
        ...
    @device_ids.setter
    def device_ids(self, arg1: list[int]) -> None:
        ...
    @property
    def orchestrator_config(self) -> OrchestratorConfig | None:
        ...
    @orchestrator_config.setter
    def orchestrator_config(self, arg1: OrchestratorConfig) -> None:
        ...
    @property
    def participant_ids(self) -> list[int] | None:
        ...
    @participant_ids.setter
    def participant_ids(self, arg1: list[int]) -> None:
        ...
class PeftCacheConfig:
    def __getstate__(self) -> tuple:
        ...
    def __init__(self, num_host_module_layer: int = 0, num_device_module_layer: int = 0, optimal_adapter_size: int = 8, max_adapter_size: int = 64, num_put_workers: int = 1, num_ensure_workers: int = 1, num_copy_streams: int = 1, max_pages_per_block_host: int = 24, max_pages_per_block_device: int = 8, device_cache_percent: float | None = None, host_cache_size: int | None = None) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def device_cache_percent(self) -> float | None:
        ...
    @property
    def host_cache_size(self) -> int | None:
        ...
    @property
    def max_adapter_size(self) -> int:
        ...
    @property
    def max_pages_per_block_device(self) -> int:
        ...
    @property
    def max_pages_per_block_host(self) -> int:
        ...
    @property
    def num_copy_streams(self) -> int:
        ...
    @property
    def num_device_module_layer(self) -> int:
        ...
    @property
    def num_ensure_workers(self) -> int:
        ...
    @property
    def num_host_module_layer(self) -> int:
        ...
    @property
    def num_put_workers(self) -> int:
        ...
    @property
    def optimal_adapter_size(self) -> int:
        ...
class PromptTuningConfig:
    def __init__(self, embedding_table: torch.Tensor) -> None:
        ...
    @property
    def embedding_table(self) -> torch.Tensor:
        ...
class Request:
    BATCHED_POST_PROCESSOR_NAME: typing.ClassVar[str] = 'batched'
    output_config: OutputConfig
    return_all_generated_tokens: bool
    sampling_config: SamplingConfig
    streaming: bool
    def __init__(self, input_token_ids: list[int], max_new_tokens: int, streaming: bool = False, sampling_config: SamplingConfig = ..., output_config: OutputConfig = ..., end_id: int | None = None, pad_id: int | None = None, bad_words: list[list[int]] | None = None, stop_words: list[list[int]] | None = None, embedding_bias: torch.Tensor | None = None, external_draft_tokens_config: ExternalDraftTokensConfig | None = None, prompt_tuning_config: PromptTuningConfig | None = None, lora_config: LoraConfig | None = None, logits_post_processor_name: str | None = None, encoder_input_token_ids: list[int] | None = None, client_id: int | None = None, return_all_generated_tokens: bool = False) -> None:
        ...
    @property
    def bad_words(self) -> list[list[int]] | None:
        ...
    @bad_words.setter
    def bad_words(self, arg1: list[list[int]]) -> None:
        ...
    @property
    def client_id(self) -> int | None:
        ...
    @client_id.setter
    def client_id(self, arg1: int) -> None:
        ...
    @property
    def embedding_bias(self) -> torch.Tensor | None:
        ...
    @embedding_bias.setter
    def embedding_bias(self, arg1: torch.Tensor) -> None:
        ...
    @property
    def encoder_input_token_ids(self) -> list[int] | None:
        ...
    @encoder_input_token_ids.setter
    def encoder_input_token_ids(self, arg1: list[int]) -> None:
        ...
    @property
    def end_id(self) -> int | None:
        ...
    @end_id.setter
    def end_id(self, arg1: int) -> None:
        ...
    @property
    def external_draft_tokens_config(self) -> ExternalDraftTokensConfig | None:
        ...
    @external_draft_tokens_config.setter
    def external_draft_tokens_config(self, arg1: ExternalDraftTokensConfig) -> None:
        ...
    @property
    def input_token_ids(self) -> list[int]:
        ...
    @property
    def logits_post_processor_name(self) -> str | None:
        ...
    @logits_post_processor_name.setter
    def logits_post_processor_name(self, arg1: str) -> None:
        ...
    @property
    def lora_config(self) -> LoraConfig | None:
        ...
    @lora_config.setter
    def lora_config(self, arg1: LoraConfig) -> None:
        ...
    @property
    def max_new_tokens(self) -> int:
        ...
    @property
    def pad_id(self) -> int | None:
        ...
    @pad_id.setter
    def pad_id(self, arg1: int) -> None:
        ...
    @property
    def prompt_tuning_config(self) -> PromptTuningConfig | None:
        ...
    @prompt_tuning_config.setter
    def prompt_tuning_config(self, arg1: PromptTuningConfig) -> None:
        ...
    @property
    def stop_words(self) -> list[list[int]] | None:
        ...
    @stop_words.setter
    def stop_words(self, arg1: list[list[int]]) -> None:
        ...
class RequestStage:
    """
    Members:
    
      QUEUED
    
      ENCODER_IN_PROGRESS
    
      CONTEXT_IN_PROGRESS
    
      GENERATION_IN_PROGRESS
    
      GENERATION_COMPLETE
    """
    CONTEXT_IN_PROGRESS: typing.ClassVar[RequestStage]  # value = <RequestStage.CONTEXT_IN_PROGRESS: 2>
    ENCODER_IN_PROGRESS: typing.ClassVar[RequestStage]  # value = <RequestStage.ENCODER_IN_PROGRESS: 1>
    GENERATION_COMPLETE: typing.ClassVar[RequestStage]  # value = <RequestStage.GENERATION_COMPLETE: 4>
    GENERATION_IN_PROGRESS: typing.ClassVar[RequestStage]  # value = <RequestStage.GENERATION_IN_PROGRESS: 3>
    QUEUED: typing.ClassVar[RequestStage]  # value = <RequestStage.QUEUED: 0>
    __members__: typing.ClassVar[dict[str, RequestStage]]  # value = {'QUEUED': <RequestStage.QUEUED: 0>, 'ENCODER_IN_PROGRESS': <RequestStage.ENCODER_IN_PROGRESS: 1>, 'CONTEXT_IN_PROGRESS': <RequestStage.CONTEXT_IN_PROGRESS: 2>, 'GENERATION_IN_PROGRESS': <RequestStage.GENERATION_IN_PROGRESS: 3>, 'GENERATION_COMPLETE': <RequestStage.GENERATION_COMPLETE: 4>}
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
class RequestStats:
    avg_num_decoded_tokens_per_iter: float
    context_prefill_position: int
    id: int
    num_generated_tokens: int
    paused: bool
    scheduled: bool
    stage: RequestStage
    def __init__(self) -> None:
        ...
    def to_json_str(self) -> str:
        ...
class RequestStatsPerIteration:
    iter: int
    request_stats: list[RequestStats]
    def __init__(self) -> None:
        ...
    def to_json_str(self) -> str:
        ...
class Response:
    @typing.overload
    def __init__(self, request_id: int, error_msg: str) -> None:
        ...
    @typing.overload
    def __init__(self, request_id: int, result: Result) -> None:
        ...
    def has_error(self) -> bool:
        ...
    @property
    def error_msg(self) -> str:
        ...
    @property
    def request_id(self) -> int:
        ...
    @property
    def result(self) -> Result:
        ...
class Result:
    context_logits: torch.Tensor | None
    cum_log_probs: list[float] | None
    encoder_output: torch.Tensor | None
    generation_logits: torch.Tensor | None
    is_final: bool
    log_probs: list[list[float]] | None
    output_token_ids: list[list[int]]
    def __init__(self) -> None:
        ...
class SamplingConfig:
    beam_search_diversity_rate: float | None
    beam_width: int
    early_stopping: int | None
    frequency_penalty: float | None
    length_penalty: float | None
    min_length: int | None
    no_repeat_ngram_size: int | None
    presence_penalty: float | None
    random_seed: int | None
    repetition_penalty: float | None
    temperature: float | None
    top_k: int | None
    top_p: float | None
    top_p_decay: float | None
    top_p_min: float | None
    top_p_reset_ids: int | None
    def __init__(self, beam_width: int = 1, top_k: int | None = None, top_p: float | None = None, top_p_min: float | None = None, top_p_reset_ids: int | None = None, top_p_decay: float | None = None, random_seed: int | None = None, temperature: float | None = None, min_length: int | None = None, beam_search_diversity_rate: float | None = None, repetition_penalty: float | None = None, presence_penalty: float | None = None, frequency_penalty: float | None = None, length_penalty: float | None = None, early_stopping: int | None = None, no_repeat_ngram_size: int | None = None) -> None:
        ...
class SchedulerConfig:
    def __getstate__(self) -> tuple:
        ...
    @typing.overload
    def __init__(self, capacity_scheduler_policy: CapacitySchedulerPolicy = ...) -> None:
        ...
    @typing.overload
    def __init__(self, capacity_scheduler_policy: CapacitySchedulerPolicy, context_chunking_policy: ContextChunkingPolicy | None) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    @property
    def capacity_scheduler_policy(self) -> CapacitySchedulerPolicy:
        ...
    @property
    def context_chunking_policy(self) -> ContextChunkingPolicy | None:
        ...
class StaticBatchingStats:
    empty_gen_slots: int
    num_context_requests: int
    num_ctx_tokens: int
    num_gen_tokens: int
    num_scheduled_requests: int
    def __init__(self) -> None:
        ...
__version__: str = '0.12.0.dev2024080600'
