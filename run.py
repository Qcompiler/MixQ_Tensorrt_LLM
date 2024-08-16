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

import argparse
import ast
import csv
import os
from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   add_common_args, load_tokenizer, read_decoder_start_token_id,
                   read_model_name, supports_inflight_batching,
                   throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    # see `add_common_args` for extended list of arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_input_length', type=int, default=923)
    parser.add_argument('--max_output_len', type=int, required=True)
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["Born in north-east France, Soyer trained as a"])
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--output_log_probs_npy',
                        type=str,
                        help='Numpy file where the log_probs are stored',
                        default=None)
    parser.add_argument('--output_cum_log_probs_npy',
                        type=str,
                        help='Numpy file where the cum_log_probs are stored',
                        default=None)
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser = add_common_args(parser)

    return parser.parse_args(args=args)


def parse_input(tokenizer,
                input_text=None,
                prompt_template=None,
                input_file=None,
                add_special_tokens=True,
                max_input_length=923,
                pad_id=None,
                num_prepend_vtokens=[],
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    batch_input_ids = []
    if input_file is None:
        for curr_text in input_text:
            if prompt_template is not None:
                curr_text = prompt_template.format(input_text=curr_text)
            input_ids = tokenizer.encode(curr_text,
                                         add_special_tokens=add_special_tokens,
                                         truncation=True,
                                         max_length=max_input_length)
            batch_input_ids.append(input_ids)
    else:
        if input_file.endswith('.csv'):
            with open(input_file, 'r') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for line in csv_reader:
                    input_ids = np.array(line, dtype='int32')
                    batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.npy'):
            inputs = np.load(input_file)
            for row in inputs:
                input_ids = row[row != pad_id]
                batch_input_ids.append(input_ids[-max_input_length:])
        elif input_file.endswith('.txt'):
            with open(input_file, 'r', encoding='utf-8',
                      errors='replace') as txt_file:
                input_text = txt_file.readlines()
                batch_input_ids = tokenizer(
                    input_text,
                    add_special_tokens=add_special_tokens,
                    truncation=True,
                    max_length=max_input_length)["input_ids"]
        else:
            print('Input file format not supported.')
            raise SystemExit

    if num_prepend_vtokens:
        assert len(num_prepend_vtokens) == len(batch_input_ids)
        base_vocab_size = tokenizer.vocab_size - len(
            tokenizer.special_tokens_map.get('additional_special_tokens', []))
        for i, length in enumerate(num_prepend_vtokens):
            batch_input_ids[i] = list(
                range(base_vocab_size,
                      base_vocab_size + length)) + batch_input_ids[i]

    if input_file is None and 'GLM' in model_name and model_version == 'glm':
        for ids in batch_input_ids:
            ids.append(tokenizer.sop_token_id)

    batch_input_ids = [
        torch.tensor(x, dtype=torch.int32) for x in batch_input_ids
    ]
    return batch_input_ids


def print_output(tokenizer,
                 output_ids,
                 input_lengths,
                 sequence_lengths,
                 output_csv=None,
                 output_npy=None,
                 context_logits=None,
                 generation_logits=None,
                 cum_log_probs=None,
                 log_probs=None,
                 output_logits_npy=None,
                 output_cum_log_probs_npy=None,
                 output_log_probs_npy=None):
    batch_size, num_beams, _ = output_ids.size()
    if output_csv is None and output_npy is None:
        for batch_idx in range(batch_size):
            inputs = output_ids[batch_idx][0][:input_lengths[batch_idx]].tolist(
            )
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {batch_idx}]: \"{input_text}\"')
            for beam in range(num_beams):
                output_begin = input_lengths[batch_idx]
                output_end = sequence_lengths[batch_idx][beam]
                outputs = output_ids[batch_idx][beam][
                    output_begin:output_end].tolist()
                output_text = tokenizer.decode(outputs)
                print(
                    f'Output [Text {batch_idx} Beam {beam}]: \"{output_text}\"')

    output_ids = output_ids.reshape((-1, output_ids.size(2)))
    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = output_ids.tolist()
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(outputs)

    if output_npy is not None:
        output_file = Path(output_npy)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        outputs = np.array(output_ids.cpu().contiguous(), dtype='int32')
        np.save(output_file, outputs)

    # Save context logits
    if context_logits is not None and output_logits_npy is not None:
        context_logits = torch.cat(context_logits, axis=0)
        vocab_size_padded = context_logits.shape[-1]
        context_logits = context_logits.reshape([1, -1, vocab_size_padded])

        output_context_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_context"
        output_context_logits_file = Path(output_context_logits_npy)
        context_outputs = np.array(
            context_logits.squeeze(0).cpu().contiguous(),
            dtype='float32')  # [promptLengthSum, vocabSize]
        np.save(output_context_logits_file, context_outputs)

    # Save generation logits
    if generation_logits is not None and output_logits_npy is not None and num_beams == 1:
        output_generation_logits_npy = output_logits_npy.split(
            '.npy')[0] + "_generation"
        output_generation_logits_file = Path(output_generation_logits_npy)
        generation_outputs = np.array(generation_logits.cpu().contiguous(),
                                      dtype='float32')
        np.save(output_generation_logits_file, generation_outputs)

    # Save cum log probs
    if cum_log_probs is not None and output_cum_log_probs_npy is not None:
        cum_log_probs_file = Path(output_cum_log_probs_npy)
        cum_log_probs_outputs = np.array(cum_log_probs.cpu().contiguous(),
                                         dtype='float32')
        np.save(cum_log_probs_file, cum_log_probs_outputs)

    # Save cum log probs
    if log_probs is not None and output_log_probs_npy is not None:
        log_probs_file = Path(output_log_probs_npy)
        log_probs_outputs = np.array(log_probs.cpu().contiguous(),
                                     dtype='float32')
        np.save(log_probs_file, log_probs_outputs)


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    # different handling if encoder-decoder models
    is_enc_dec = {
        name
        for name in os.listdir(args.engine_dir)
        if os.path.isdir(os.path.join(args.engine_dir, name))
    } == {'encoder', 'decoder'}
    if is_enc_dec:
        logger.warning(
            "This path is an encoder-decoder model. Using different handling.")
        assert not args.use_py_session, "Encoder-decoder models don't have a unified python runtime, please use its own examples/enc_dec/run.py instead."

    model_name, model_version = read_model_name(
        args.engine_dir) if not is_enc_dec else ("", "")
    if args.tokenizer_dir is None and model_name in DEFAULT_HF_MODEL_DIRS:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    if args.end_id:
        end_id = args.end_id

    prompt_template = None
    if args.use_prompt_template and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]
    batch_input_ids = parse_input(tokenizer=tokenizer,
                                  input_text=args.input_text,
                                  prompt_template=prompt_template,
                                  input_file=args.input_file,
                                  add_special_tokens=args.add_special_tokens,
                                  max_input_length=args.max_input_length,
                                  pad_id=pad_id,
                                  num_prepend_vtokens=args.num_prepend_vtokens,
                                  model_name=model_name,
                                  model_version=model_version)

    stop_words_list = None
    if args.stop_words:
        stop_words_list = tensorrt_llm.runtime.decode_words_list(
            args.stop_words, tokenizer)
    if model_version == 'glm4':  # add default stop token ids for GLM-4
        glm4_stop_ids = [[151329], [151336], [151338]]
        if stop_words_list is None:
            stop_words_list = [glm4_stop_ids] * len(batch_input_ids)
        else:
            for req_stop_words_list in stop_words_list:
                req_stop_words_list.extend(glm4_stop_ids)

    bad_words_list = None
    if args.bad_words:
        bad_words_list = tensorrt_llm.runtime.decode_words_list(
            args.bad_words, tokenizer)

    if is_enc_dec:
        encoder_input_ids = batch_input_ids
        decoder_start_token_id = read_decoder_start_token_id(
            os.path.join(args.engine_dir, "decoder"))
        decoder_input_ids = [
            torch.tensor([decoder_start_token_id], dtype=torch.int32)
            for _ in batch_input_ids
        ]

    input_lengths = [x.size(0) for x in decoder_input_ids
                     ] if is_enc_dec else [x.size(0) for x in batch_input_ids]
    encoder_input_lengths = [x.size(0)
                             for x in encoder_input_ids] if is_enc_dec else None

    if not args.use_py_session and not supports_inflight_batching(
            os.path.join(args.engine_dir, "decoder") if is_enc_dec else args.
            engine_dir):
        logger.warning(
            "The given engine does not support in-flight batching, fallback to python session"
        )
        args.use_py_session = True

    if not PYTHON_BINDINGS and not args.use_py_session:
        logger.warning(
            "Python bindings of C++ session is unavailable, fallback to Python session."
        )
        args.use_py_session = True
    if args.debug_mode and not args.use_py_session:
        logger.warning(
            "Debug mode is not supported in C++ session for now, fallback to Python session."
        )
        args.use_py_session = True
    if args.return_all_generated_tokens and args.use_py_session:
        raise ValueError(
            "Returning all the generated tokens at each step is not supported in the Python session, use C++ session instead."
        )
    if (not args.return_all_generated_tokens) and args.streaming and (
            args.num_beams > 1):
        logger.warning(
            "Setting return_all_generated_tokens to True since streaming AND beam search are done simultaneously. "
            "Returning the full beams at each streaming step is needed because beam search + streaming can change previous outputs. "
            "WARNING: using this option may increase network usage significantly (quadratically w.r.t output length)."
        )
        args.return_all_generated_tokens = True
    runner_cls = ModelRunner if args.use_py_session else ModelRunnerCpp
    runner_kwargs = dict(
        engine_dir=args.engine_dir,
        lora_dir=args.lora_dir,
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        lora_ckpt_source=args.lora_ckpt_source,
        gpu_weights_percent=args.gpu_weights_percent,
        max_output_len=args.max_output_len,
    )
    if not args.use_py_session:
        runner_kwargs.update(is_enc_dec=is_enc_dec)
    if args.medusa_choices is not None:
        args.medusa_choices = ast.literal_eval(args.medusa_choices)
        assert args.temperature == 1.0, "Medusa should use temperature == 1.0"
        assert args.num_beams == 1, "Medusa should use num_beams == 1"
        runner_kwargs.update(medusa_choices=args.medusa_choices)
    if not args.use_py_session:
        runner_kwargs.update(
            max_batch_size=len(batch_input_ids),
            max_input_len=max(
                encoder_input_lengths if is_enc_dec else input_lengths),
            max_beam_width=args.num_beams,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
            kv_cache_free_gpu_memory_fraction=args.
            kv_cache_free_gpu_memory_fraction,
            enable_chunked_context=args.enable_chunked_context,
            multi_block_mode=args.multi_block_mode)
    runner_kwargs.update(
        enable_context_fmha_fp32_acc=args.enable_context_fmha_fp32_acc)
    runner = runner_cls.from_dir(**runner_kwargs)

    with torch.no_grad():
        outputs = runner.generate(
            batch_input_ids=decoder_input_ids
            if is_enc_dec else batch_input_ids,
            encoder_input_ids=encoder_input_ids if is_enc_dec else None,
            max_new_tokens=args.max_output_len,
            max_attention_window_size=args.max_attention_window_size,
            sink_token_length=args.sink_token_length,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
            early_stopping=args.early_stopping,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            frequency_penalty=args.frequency_penalty,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            output_cum_log_probs=(args.output_cum_log_probs_npy != None),
            output_log_probs=(args.output_log_probs_npy != None),
            random_seed=args.random_seed,
            lora_uids=args.lora_task_uids,
            prompt_table=args.prompt_table_path,
            prompt_tasks=args.prompt_tasks,
            streaming=args.streaming,
            output_sequence_lengths=True,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            return_dict=True,
            medusa_choices=args.medusa_choices,
            return_all_generated_tokens=args.return_all_generated_tokens)
        torch.cuda.synchronize()

    if args.streaming:
        for curr_outputs in throttle_generator(outputs,
                                               args.streaming_interval):
            if runtime_rank == 0:
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                cum_log_probs = None
                log_probs = None
                if args.output_cum_log_probs_npy != None:
                    cum_log_probs = outputs['cum_log_probs']
                if args.output_log_probs_npy != None:
                    log_probs = outputs['log_probs']
                print_output(
                    tokenizer,
                    output_ids,
                    input_lengths,
                    sequence_lengths,
                    output_csv=args.output_csv,
                    output_npy=args.output_npy,
                    cum_log_probs=cum_log_probs,
                    log_probs=log_probs,
                    output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                    output_log_probs_npy=args.output_log_probs_npy)
    else:
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            context_logits = None
            generation_logits = None
            cum_log_probs = None
            log_probs = None
            if runner.gather_context_logits:
                context_logits = outputs['context_logits']
            if runner.gather_generation_logits:
                generation_logits = outputs['generation_logits']
            if args.output_cum_log_probs_npy != None:
                cum_log_probs = outputs['cum_log_probs']
            if args.output_log_probs_npy != None:
                log_probs = outputs['log_probs']
            print_output(tokenizer,
                         output_ids,
                         input_lengths,
                         sequence_lengths,
                         output_csv=args.output_csv,
                         output_npy=args.output_npy,
                         context_logits=context_logits,
                         generation_logits=generation_logits,
                         output_logits_npy=args.output_logits_npy,
                         cum_log_probs=cum_log_probs,
                         log_probs=log_probs,
                         output_cum_log_probs_npy=args.output_cum_log_probs_npy,
                         output_log_probs_npy=args.output_log_probs_npy)

    if args.run_profiling:
        ite = 10
        # warmup
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    output_cum_log_probs=(args.output_cum_log_probs_npy !=
                                          None),
                    output_log_probs=(args.output_log_probs_npy != None),
                    random_seed=args.random_seed,
                    lora_uids=args.lora_task_uids,
                    prompt_table=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True,
                    return_all_generated_tokens=args.return_all_generated_tokens
                )
                torch.cuda.synchronize()

        tensorrt_llm.profiler.start("tmp")
        for _ in range(ite):
            with torch.no_grad():
                outputs = runner.generate(
                    batch_input_ids,
                    max_new_tokens=args.max_output_len,
                    max_attention_window_size=args.max_attention_window_size,
                    end_id=end_id,
                    pad_id=pad_id,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=args.early_stopping,
                    repetition_penalty=args.repetition_penalty,
                    presence_penalty=args.presence_penalty,
                    frequency_penalty=args.frequency_penalty,
                    stop_words_list=stop_words_list,
                    bad_words_list=bad_words_list,
                    output_cum_log_probs=(args.output_cum_log_probs_npy !=
                                          None),
                    output_log_probs=(args.output_log_probs_npy != None),
                    random_seed=args.random_seed,
                    lora_uids=args.lora_task_uids,
                    prompt_table=args.prompt_table_path,
                    prompt_tasks=args.prompt_tasks,
                    streaming=args.streaming,
                    output_sequence_lengths=True,
                    return_dict=True,
                    return_all_generated_tokens=args.return_all_generated_tokens
                )
                torch.cuda.synchronize()
        tensorrt_llm.profiler.stop("tmp")

        print(
            f"batch_size: {len(batch_input_ids)}, avg latency of {ite} iterations: : {tensorrt_llm.profiler.elapsed_time_in_sec('tmp') / ite} sec"
        )


if __name__ == '__main__':
    args = parse_arguments()
    main(args)