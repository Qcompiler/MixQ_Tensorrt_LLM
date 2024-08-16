# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Handles quantization plugins to correctly quantize third-party modules.

Please check out the source code of this module for examples of how plugins work and how you can
write your own one. Currently, we support plugins for

- :meth:`apex<modelopt.torch.quantization.plugins.apex>`
- :meth:`diffusers<modelopt.torch.quantization.plugins.diffusers>`
- :meth:`huggingface<modelopt.torch.quantization.plugins.huggingface>`
- :meth:`megatron<modelopt.torch.quantization.plugins.megatron>`
- :meth:`nemo<modelopt.torch.quantization.plugins.nemo>`
"""
import warnings

try:
    from .apex import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import apex plugin due to: {repr(e)}")


try:
    from .diffusers import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import diffusers plugin due to: {repr(e)}")

try:
    from .huggingface import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import huggingface plugin due to: {repr(e)}")


try:
    from .megatron import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import megatron plugin due to: {repr(e)}")

try:
    from .nemo import *
except ImportError:
    pass
except Exception as e:
    warnings.warn(f"Failed to import nemo plugin due to: {repr(e)}")
