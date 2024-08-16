import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

from transformers import PreTrainedTokenizerBase

from .. import bindings as tllm
from ..bindings import executor as tllm
from ..executor import GenerationExecutor, GenerationResult
from ..logger import logger
from .llm_utils import (CachedModelLoader, LlmArgs, LlmBuildStats, ModelLoader,
                        _ModelRuntimeContext)
from .mpi_session import (MpiCommSession, MpiPoolSession, MpiSession,
                          external_mpi_comm_available)
from .tokenizer import TokenizerBase
# TODO[chunweiy]: move the following symbols back to utils scope, and remove the following import
from .utils import SamplingParams, exception_handler, get_device_count


class RequestOutput(GenerationResult):
    """The output data of a completion request to the LLM.

    Fields:
        request_id (int): The unique ID of the request.
        prompt (str): The prompt string of the request.
        prompt_token_ids (List[int]): The token ids of the prompt.
        outputs (List[CompletionOutput]): The output sequences of the request.
        context_logits (torch.Tensor): The logits on the prompt token ids.
        finished (bool): Whether the whole request is finished.
    """

    def __init__(self,
                 generation_result: GenerationResult,
                 prompt: Optional[str] = None,
                 tokenizer: Optional[TokenizerBase] = None) -> None:
        self.__dict__.update(generation_result.__dict__)
        self.prompt = prompt
        self.tokenizer = tokenizer

    def handle_generation_msg(self, tensors: tuple, error: str):
        super().handle_generation_msg(tensors, error)

        if self.tokenizer is not None:
            for beam_output in self.outputs:
                beam_output.text = self.tokenizer.decode(beam_output.token_ids)

    def _repr_fields(self):
        return [
            'request_id', 'prompt', 'prompt_token_ids', 'outputs', 'finished'
        ]


class LLM:

    def __init__(self,
                 model: str,
                 tokenizer: Optional[Union[str, Path, TokenizerBase,
                                           PreTrainedTokenizerBase]] = None,
                 skip_tokenizer_init: bool = False,
                 tensor_parallel_size: int = 1,
                 dtype: str = "auto",
                 revision: Optional[str] = None,
                 tokenizer_revision: Optional[str] = None,
                 **kwargs: Any):
        '''
        Args:
            model(str): The model name or a local path to the model directory. It could be a HuggingFace(HF) model name,
                a local path to the HF model, or a local path to the TRT-LLM engine or checkpoint.
            tokenizer(Optional[Union[str, Path, TokenizerBase, PreTrainedTokenizerBase]]): The tokenizer name or a local
                path to the tokenizer directory.
            skip_tokenizer_init: If true, skip initialization of tokenizer and detokenizer. generate and generate_async
                will accept prompt token ids as input only.
            tensor_parallel_size(int): The number of processes for tensor parallelism.
            dtype(str): The data type for the model weights and activations.
            revision(Optional[str]): The revision of the model.
            tokenzier_revision(Optional[str]): The revision of the tokenizer.

            kwargs: Contains the optional arguments for expert users, please refer to `llm_utils.LlmArgs` for more details.
        '''
        try:
            self.args = LlmArgs.from_kwargs(
                model=model,
                tokenizer=tokenizer,
                skip_tokenizer_init=skip_tokenizer_init,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                revision=revision,
                tokenizer_revision=tokenizer_revision,
                **kwargs)
        except Exception as e:
            logger.error(
                f"Failed to parse the arguments for the LLM constructor: {e}")
            raise e

        self.mpi_session: Optional[MpiSession] = None
        if self.args.parallel_config.is_multi_gpu:
            if get_device_count() < self.args.parallel_config.world_size:
                raise RuntimeError(
                    f"Only {get_device_count()} GPUs are available, but {self.args.parallel_config.world_size} are required."
                )

            logger.info(
                f'start MpiSession with {self.args.parallel_config.world_size} workers'
            )
            if not external_mpi_comm_available(
                    self.args.parallel_config.world_size):
                self.mpi_session = MpiPoolSession(
                    n_workers=self.args.parallel_config.world_size)
            else:
                self.mpi_session = MpiCommSession(
                    n_workers=self.args.parallel_config.world_size)

        # Due to the Executor can only accept a engine path, we need to save the engine to a directory
        self._engine_dir: Optional[Path] = None
        self._executor: Optional[GenerationExecutor] = None
        self._workspace = tempfile.TemporaryDirectory("llm-workspace")

        self.runtime_context: Optional[_ModelRuntimeContext] = None
        self.llm_build_stats = LlmBuildStats()

        self._build_model()
        exception_handler.register(self)

    @property
    def workspace(self) -> Path:
        return Path(self._workspace.name)

    def generate(
        self,
        prompts: Union[str, Iterable[str], List[int], Iterable[List[int]]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None
    ) -> Union[RequestOutput, List[RequestOutput]]:
        ''' Generate output for the given prompts in the synchronous mode.
        Synchronous generation accepts either single prompt or batched prompts.

        Args:
            prompts: The prompt text or token ids; could be either single prompt or batched prompts.
            sampling_params: The sampling params for the generation, a default one will be used if not provided.
        '''
        if isinstance(prompts, str) or isinstance(prompts[0], str):
            unbatched = isinstance(prompts, str)
        else:
            unbatched = isinstance(prompts[0], int)

        if unbatched:
            prompts = [prompts]

        futures = []
        for i, prompt in enumerate(prompts):
            if isinstance(sampling_params, list):
                sp = sampling_params[i]
            else:
                sp = sampling_params
            future = self.generate_async(prompt,
                                         sampling_params=sp,
                                         streaming=False)
            futures.append(future)

        for future in futures:
            future.result()

        if unbatched:
            futures = futures[0]

        return futures

    def generate_async(
        self,
        prompt: Union[str, List[int]],
        sampling_params: Optional[SamplingParams] = None,
        streaming: bool = False,
    ) -> RequestOutput:
        ''' Generate output for the given prompt in the asynchronous mode.
        Asynchronous generation accepts single prompt only.

        Args:
            prompt: The prompt text or token ids; must be single prompt.
            sampling_params: The sampling params for the generation, a default one will be used if not provided.
            streaming: Whether to use the streaming mode for the generation.
        '''
        if isinstance(prompt, str):
            prompt_token_ids = self._prepare_prompt_token_ids(prompt)
        elif isinstance(prompt, list) and isinstance(prompt[0], int):
            prompt_token_ids = prompt
            prompt = None
        else:
            raise TypeError(
                f"The prompt must be type str or list of int, but got {type(prompt)}"
            )

        sampling_params = self._prepare_sampling_params(sampling_params)
        self._check_arguments(prompt_token_ids, sampling_params)

        result = self._executor.generate_async(
            prompt_token_ids,
            sampling_params=sampling_params,
            streaming=streaming,
        )
        return RequestOutput(result, prompt, self.tokenizer)

    def _prepare_prompt_token_ids(self, prompt: str) -> List[int]:
        if self.tokenizer is None:
            raise ValueError("tokenizer is required to tokenize string prompt")
        return self.tokenizer.encode(prompt)

    def _prepare_sampling_params(
            self,
            sampling_params: Optional[SamplingParams] = None) -> SamplingParams:
        if sampling_params is None:
            if self.tokenizer is None:
                raise ValueError(
                    "tokenizer is required to initialize a default sampling_params, or you can explicitly specify a sampling_params"
                )
            return SamplingParams(end_id=self.tokenizer.eos_token_id,
                                  pad_id=self.tokenizer.pad_token_id)
        elif isinstance(sampling_params, SamplingParams):
            if sampling_params.end_id is None:
                if self.tokenizer is None:
                    raise ValueError(
                        "tokenizer is required to reset end_id if it is None, or you can explicitly specify the end_id for sampling_params"
                    )
            return sampling_params.setup(self.tokenizer)
        else:
            raise TypeError(
                f"The sampling_params must be type SamplingParams or None, but got {type(sampling_params)}"
            )

    def _check_arguments(self, prompt_token_ids: List[int],
                         sampling_params: SamplingParams) -> None:

        build_config = self.args.build_config
        prompt_len = len(prompt_token_ids)

        if prompt_len + sampling_params.max_new_tokens > build_config.max_seq_len:
            raise ValueError(
                f"The sum of prompt length ({prompt_len}) and max_new_tokens ({sampling_params.max_new_tokens}) should not exceed "
                f"max_seq_len ({build_config.max_seq_len})")
        if sampling_params.beam_width > build_config.max_beam_width:
            raise ValueError(
                f"sampling_params's beam_width ({sampling_params.beam_width}) should not exceed max_beam_width ({build_config.max_beam_width})"
            )

    def _build_model(self):
        model_loader = CachedModelLoader(self.args,
                                         mpi_session=self.mpi_session,
                                         workspace=self.workspace,
                                         llm_build_stats=self.llm_build_stats)
        self._engine_dir = model_loader()

        executor_config = tllm.ExecutorConfig(
            max_beam_width=self.args.build_config.max_beam_width,
            scheduler_config=self.args.scheduler_config,
            batching_type=tllm.BatchingType.INFLIGHT)
        if self.args.kv_cache_config is not None:
            executor_config.kv_cache_config = self.args.kv_cache_config
        if self.args.peft_cache_config is not None:
            executor_config.peft_cache_config = self.args.peft_cache_config
        if self.args.decoding_config is not None:
            executor_config.decoding_config = self.args.decoding_config
        if self.args.logits_post_processor_map:
            executor_config.logits_post_processor_map = self.args.logits_post_processor_map
        executor_config.normalize_log_probs = self.args.normalize_log_probs
        executor_config.enable_chunked_context = self.args.enable_chunked_context
        executor_config.max_beam_width = self.args.build_config.max_beam_width

        self._executor = GenerationExecutor.create(
            self._engine_dir,
            executor_config=executor_config,
            model_world_size=self.args.parallel_config.world_size,
            mpi_session=self.mpi_session,
            reuse_mpi_comm=external_mpi_comm_available(
                self.args.parallel_config.world_size))

    @property
    def tokenizer(self) -> TokenizerBase:
        if self.args.skip_tokenizer_init:
            return None
        if self.args.tokenizer is not None:
            return self.args.tokenizer
        if self.runtime_context is not None:
            return self.runtime_context.tokenizer

        try:
            self.args.tokenizer = ModelLoader.load_hf_tokenizer(
                self.args.model_dir)
        except:
            pass

        return self.args.tokenizer

    def save(self, engine_dir: str):
        ''' Save the built engine to the given path. '''
        logger.info(f"Save model to {engine_dir}")
        if self._engine_dir is None:
            raise RuntimeError("The engine is not built yet.")
        if self._engine_dir.absolute() != os.path.abspath(engine_dir):
            shutil.copytree(self._engine_dir, engine_dir, dirs_exist_ok=True)

    def shutdown(self):
        if hasattr(self, "_executor") and self._executor is not None:
            self._executor.shutdown()

        if self.mpi_session is not None:
            self.mpi_session.shutdown()
            self.mpi_session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        del exc_value, traceback
        self.shutdown()
        return exc_type is not None

    def __getstate__(self):
        raise RuntimeError("LLM object can not be pickled.")

    def __del__(self):
        self.shutdown()
