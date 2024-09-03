
# //header begin//

import ctypes
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np
import tensorrt as trt

from tensorrt_llm._common import default_trtnet
from tensorrt_llm._utils import str_dtype_to_trt
from tensorrt_llm.functional import Tensor, _create_tensor
from tensorrt_llm.module import Module

TRT_LLM_PLUGIN_NAMESPACE = 'tensorrt_llm'

def _load_triton_plugin_lib():
    triton_plugin_dir = Path(__file__).parent.absolute()
    plugin_lib = "./tmp/build/libtriton_plugins.so"
    handle = ctypes.CDLL(plugin_lib, mode=ctypes.RTLD_GLOBAL)
    if handle is None:
        raise ImportError('TensorRT-LLM Triton Plugin is unavailable')
    handle.initLibNvInferPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    handle.initLibNvInferPlugins.restype = ctypes.c_bool
    assert handle.initLibNvInferPlugins(
        None, TRT_LLM_PLUGIN_NAMESPACE.encode('utf-8'))

_load_triton_plugin_lib()

# //header end//


def fused_attention_kernel(sm_scale, num_heads, Q, K, V):
    '''
    Inputs:
    - sm_scale: float32
    - num_heads: int32
    
    - Q: tensor<float16>
    - K: tensor<float16>
    - V: tensor<float16>
    
    Outputs:
    - Out: tensor<float16>
    - L: tensor<float32>
    - M: tensor<float32>
    '''
    plg_creator = trt.get_plugin_registry().get_plugin_creator(
        'fused_attention_kernelPlugin', '0', TRT_LLM_PLUGIN_NAMESPACE)
    assert plg_creator is not None

    pfc = trt.PluginFieldCollection([
        
        trt.PluginField("sm_scale", np.array([ sm_scale ], np.float32),
                        trt.PluginFieldType.FLOAT32),
        
        trt.PluginField("num_heads", np.array([ num_heads ], np.int32),
                        trt.PluginFieldType.INT32),
        
    ])

    plugin = plg_creator.create_plugin("fused_attention_kernelPlugin", pfc)

    plug_inputs = [ Q, K, V ]
    layer = default_trtnet().add_plugin_v2(plug_inputs, plugin)

    return [
        
        _create_tensor(layer.get_output(0), layer),
        
        _create_tensor(layer.get_output(1), layer),
        
        _create_tensor(layer.get_output(2), layer),
        
    ]