from typing import Mapping, Optional, Union

import torch

from ..._utils import torch_dtype_to_str
from ...mapping import Mapping
from ..modeling_utils import PretrainedConfig, QuantConfig


class GPTJConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of GPTJ model.
    """

    def __init__(self, *, rotary_dim: int = 64, **kwargs):
        self.rotary_dim = rotary_dim
        super().__init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.update(rotary_dim=self.rotary_dim)
        return output

    @classmethod
    def from_hugging_face(
            cls,
            hf_config_or_dir: Union[str, 'transformers.PretrainedConfig'],
            dtype: str = 'auto',
            mapping: Optional[Mapping] = None,
            quant_config: Optional[QuantConfig] = None,
            **kwargs):
        import transformers

        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config_dir = str(hf_config_or_dir)
            hf_config = transformers.AutoConfig.from_pretrained(
                hf_config_dir, trust_remote_code=True)

        if dtype == 'auto':
            dtype = getattr(hf_config, 'torch_dtype', None)
            if dtype is None:
                dtype = 'float16'
            if isinstance(dtype, torch.dtype):
                dtype = torch_dtype_to_str(dtype)
            if dtype == 'float32':
                dtype = 'float16'

        return cls(architecture=hf_config.architectures[0],
                   dtype=dtype,
                   num_hidden_layers=hf_config.num_hidden_layers,
                   num_attention_heads=hf_config.num_attention_heads,
                   hidden_size=hf_config.hidden_size,
                   norm_epsilon=hf_config.layer_norm_epsilon,
                   vocab_size=hf_config.vocab_size,
                   position_embedding_type='rope_gptj',
                   max_position_embeddings=hf_config.max_position_embeddings,
                   hidden_act='gelu',
                   rotary_dim=hf_config.rotary_dim,
                   mapping=mapping,
                   quantization=quant_config,
                   **kwargs)
