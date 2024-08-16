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
import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor,_add_plugin_info, cast
from tensorrt_llm.module import Module
from tensorrt_llm._common import default_net, default_trtnet

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'
LAYER_NAME = 'MixQLayer'
FMHA_KERNEL_BLOCK_SIZE = 128


def _load_lianxiang_plugin_lib():
    triton_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = triton_plugin_dir / 'build/libtrt_llm_custom_plugins.so'
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Triton Plugin is unavailable')
    handle.initOpenAiTritonPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initOpenAiTritonPlugins.restype = ctypes.c_bool
    assert handle.initOpenAiTritonPlugins(
        None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))

print("loading plugging ")
_load_lianxiang_plugin_lib()
print("done! ")





plugin_creator = trt.get_plugin_registry().get_plugin_creator(
    'MixQ', '1', TRT_LLM_PLUGIN_NAMESPACE)
assert plugin_creator is not None