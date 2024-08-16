# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
import subprocess  # nosec
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, List, Optional, Tuple

from ..._runtime.common import read_bytes, timeit, write_bytes, write_string
from ..._runtime.tensorrt.layerwise_profiling import process_layerwise_result
from ...utils import OnnxBytes
from .constants import (
    DEFAULT_NUM_INFERENCE_PER_RUN,
    SHA_256_HASH_LENGTH,
    TRTEXEC_PATH,
    WARMUP_TIME_MS,
    TRTMode,
)
from .tensorrt_utils import convert_shape_to_string, prepend_hash_to_bytes

try:
    from trex import EnginePlan, layer_type_formatter, render_dot, to_dot

    HAVE_TREX = True
except ImportError:
    HAVE_TREX = False


# TODO: Get rid of this function or get approval for `# nosec` usage if we want to include this
#   as a non-compiled python file in the release.
def _run_command(cmd: str, cwd: str = None) -> Tuple[int, str]:
    """Util function to execute a command.

    This util will not direct stdout and stderr to console if the cmd succeeds.

    Args:
        cmd: the command line string
        cwd: current working directory

    Return:
        return code: O means successful.
        log_string: the stdout and stderr output as a string.

    """
    logging.info(cmd)
    with NamedTemporaryFile("w+") as log:
        p = subprocess.Popen(cmd.split(), stdout=log, stderr=log, cwd=cwd)  # nosec
        p.wait()
        log.seek(0)
        log_string = log.read()
        if p.returncode != 0:
            logging.error(log_string)
        return p.returncode, log_string


def _get_profiling_params(profiling_runs: int) -> List[str]:
    return [
        f"--warmUp={WARMUP_TIME_MS}",
        f"--avgRuns={DEFAULT_NUM_INFERENCE_PER_RUN}",
        f"--iterations={profiling_runs * DEFAULT_NUM_INFERENCE_PER_RUN}",
    ]


def _get_trtexec_params(
    engine_path: str,
    builder_optimization_level: str,
    timing_cache_file: str = None,
) -> List[str]:

    cmd = ["--saveEngine=" + engine_path]

    cmd.append("--skipInference")

    if timing_cache_file:
        cmd.append("--timingCacheFile=" + timing_cache_file)

    # set the builder optimization level to 4
    cmd.append("--builderOptimizationLevel=" + builder_optimization_level)

    # Generates an engine layer graph file for visualization.
    cmd += [
        "--verbose",
        f"--exportLayerInfo={engine_path}.graph.json",
    ]

    return cmd


def _draw_engine(engine_json_fname: str) -> Optional[bytes]:
    if not HAVE_TREX:
        logging.exception("`trex` is not installed. Skipping engine graph drawing.")
        return None

    plan = EnginePlan(engine_json_fname)
    formatter = layer_type_formatter
    display_regions = True
    expand_layer_details = False

    graph = to_dot(
        plan, formatter, display_regions=display_regions, expand_layer_details=expand_layer_details
    )
    render_dot(graph, engine_json_fname, "svg")

    with open(f"{engine_json_fname}.svg", "rb") as svg_file:
        return svg_file.read()


@timeit
def build_engine(
    onnx_bytes: OnnxBytes,
    trt_mode: str = TRTMode.FLOAT32,
    calib_cache: str = None,
    dynamic_shapes: dict = None,
    plugin_config: dict = None,
    builder_optimization_level: str = "4",
    draw_engine: bool = False,
) -> Tuple[Optional[bytes], bytes, Optional[bytes]]:
    """This method produces serialized TensorRT engine from an ONNX model.

    Args:
        onnx_bytes: Data of the ONNX model stored as an OnnxBytes object.
        trt_mode: TensorRT conversion mode. Supported modes are:
            - TRTMode.FLOAT32
            - TRTMode.FLOAT16
            - TRTMode.INT8
            - TRTMode.FLOAT8
            - TRTMode.INT4
        calib_cache: Calibration data cache.
        dynamic_shapes: Dictionary of dynamic shapes for the input tensors. Example is as follows:
            {
                "minShapes": {"input": [1,3,244,244]},
                "optShapes": {"input": [16,3,244,244]},
                "maxShapes": {"input": [32,3,244,244]}
            }
        plugin_config: Dictionary of plugin configurations. Example is as follows:
            {
                "staticPlugins": ["staticPluginOne.so", "staticPluginTwo.so", ...],
                "dynamicPlugins": ["dynamicPluginOne.so", "dynamicPluginTwo.so", ...],
                "setPluginsToSerialize": ["dynamicPluginOne.so", "pluginSerializeOne.so", ...],
                "ignoreParsedPluginLibs": False
            }
        builder_optimization_level: Optimization level for the TensorRT builder.
        draw_engine: True or False based on whether to draw the engine graph or not.

    Returns:
        The generated engine file data.

        The stdout log produced by trtexec tool. \
            [If there is subprocess.CalledProcessError, this byte variable is transferred to str]

        The generated trt engine graph as svg.
    """
    with TemporaryDirectory() as working_dir:
        onnx_path = os.path.join(working_dir, "onnx")
        calib_cache_path = os.path.join(working_dir, "calib_cache")

        onnx_bytes.write_to_disk(onnx_path)
        engine_path = os.path.join(
            working_dir, f"{onnx_bytes.model_name}/{onnx_bytes.model_name}.engine"
        )
        # TODO: Enable timing cache

        cmd = [
            TRTEXEC_PATH,
            "--onnx=" + onnx_path + f"/{onnx_bytes.model_name}.onnx",
        ]
        if trt_mode == TRTMode.FLOAT16:
            cmd.append("--fp16")
        elif trt_mode == TRTMode.INT8:
            cmd.extend(["--fp16", "--int8"])
            if calib_cache:
                write_string(calib_cache, calib_cache_path)
                cmd.append(f"--calib={calib_cache_path}")
        elif trt_mode == TRTMode.FLOAT8:
            cmd.extend(["--fp16", "--fp8"])
        elif trt_mode == TRTMode.INT4:
            cmd.append("--fp16")

        if dynamic_shapes:
            # TODO: Add case where only optShapes are provided
            shapes = ["minShapes", "optShapes", "maxShapes"]
            for shape in shapes:
                if shape in dynamic_shapes:
                    cmd.extend(
                        [
                            f"--{shape}=" + convert_shape_to_string(dynamic_shapes[shape]),
                        ]
                    )

        if plugin_config:
            if "staticPlugins" in plugin_config:
                for plugin in plugin_config["staticPlugins"]:
                    cmd.append(f"--staticPlugins={plugin}")
            if "dynamicPlugins" in plugin_config:
                for plugin in plugin_config["dynamicPlugins"]:
                    cmd.append(f"--dynamicPlugins={plugin}")
            if "setPluginsToSerialize" in plugin_config:
                for plugin in plugin_config["setPluginsToSerialize"]:
                    cmd.append(f"--setPluginsToSerialize={plugin}")
            if "ignoreParsedPluginLibs" in plugin_config:
                cmd.append("--ignoreParsedPluginLibs")

        cmd += _get_trtexec_params(
            engine_path,
            builder_optimization_level,
        )

        try:
            engine_directory = os.path.dirname(engine_path)
            os.makedirs(engine_directory, exist_ok=True)
            trtexec_cmd = " ".join(cmd)
            print(f"Building the TRT engine with command: {trtexec_cmd}")
            ret_code, out = _run_command(trtexec_cmd)
            if ret_code != 0:
                return None, out.encode(), None

        # _run_command will handle the exceptions from trtexec
        # For other errors, they will be caught in the block below
        except Exception as e:
            out = str(e)
            logging.exception(out)
            return None, out.encode(), None

        engine_bytes = read_bytes(engine_path)
        engine_graph_bytes = _draw_engine(f"{engine_path}.graph.json") if draw_engine else None

        engine_bytes = prepend_hash_to_bytes(engine_bytes)

        return engine_bytes, out.encode(), engine_graph_bytes


@timeit
def profile_engine(
    engine_bytes: bytes,
    onnx_node_names: List[str],
    profiling_runs: int = 1,
    enable_layerwise_profiling: bool = False,
) -> Tuple[Optional[Dict[str, float]], bytes]:
    """This method produces profiles a TensorRT engine and returns the detailed results.

    Args:
        engine_bytes: Bytes of the serialized TensorRT engine, prepended with a SHA256 hash.
        onnx_node_names: List of node names in the onnx model.
        profiling_runs: number of profiling runs. Each run runs `DEFAULT_NUM_INFERENCE_PER_RUN` inferences.
        enable_layerwise_profiling:
            True or False based on whether layerwise profiling is required or not.

    Returns:
        Layerwise profiling output as a json string.
        Stdout log produced by trtexec tool.
    """
    with TemporaryDirectory() as working_dir:
        engine_path = os.path.join(working_dir, "engine")
        profile_path = os.path.join(working_dir, "profile")

        engine_bytes = engine_bytes[SHA_256_HASH_LENGTH:]  # Remove the hash

        write_bytes(engine_bytes, engine_path)

        cmd = [TRTEXEC_PATH, "--loadEngine=" + engine_path]

        cmd += _get_profiling_params(profiling_runs)

        if enable_layerwise_profiling:
            cmd += [
                "--dumpProfile",
                "--separateProfileRun",
                "--exportProfile=" + profile_path,
            ]

        try:
            _, out = _run_command(" ".join(cmd))

            layerwise_results = {}  # empty dictionary
            if enable_layerwise_profiling and os.path.exists(profile_path):
                layerwise_results = process_layerwise_result(profile_path, onnx_node_names)

            return layerwise_results, out.encode()
        # If _run_command has error,
        # the error will be subprocess.CalledProcessError
        # We will catch this error in this block and send back the str version of the error message(e.output)
        # The reason we have this independent block is that the error message, e.output,
        # is byte type with "utf-8" coding, different from other error.
        except subprocess.CalledProcessError as e:
            out = str(e)
            logging.exception(out)
            return None, out.encode()
        # For all other errors, they will be caught in the block below, error info str(e) will be sent back
        except Exception as e:
            out = str(e)
            logging.exception(out)
            return None, out.encode()
