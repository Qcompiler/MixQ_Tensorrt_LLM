import torch
import logging
import functools
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
from awq.utils.utils import clear_memory
from awq.utils.calib_data import get_calib_dataset
from awq.quantize.scale import apply_scale, apply_clip
from awq.modules.linear import WQLinear_GEMM, WQLinear_GEMV
from awq.utils.module import append_str_prefix, get_op_name, get_named_linears, set_op_by_name



class MixQuantizer:
    def __init__(self, f16_model, model, tokenizer, w_bit, group_size, version, 
                       calib_data, split, text_column) -> None:
        self.f16_model = f16_model
        self.model = model
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.version = version
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.modules, self.module_kwargs, self.inps = self.init_quant()
    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.awq_model.get_model_layers(self.model)
        samples = get_calib_dataset(
            data=self.calib_data, tokenizer=self.tokenizer, n_samples=n_samples, block_size=seqlen,
            split=self.split, text_column=self.text_column
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        modules[0] = modules[0].cuda()
        self.awq_model.move_embed(self.model, "cuda")
        
        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hijacked_inputs, **kwargs):
                inps.append(hijacked_inputs)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        del samples
        modules[0] = modules[0].module  # restore
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.awq_model.move_embed(self.model, "cpu")
        
        clear_memory()
        
        if "attention_mask" in layer_kwargs.keys():
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to("cuda")

        return modules, layer_kwargs, inps
