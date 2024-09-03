/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin.h"
#include <iostream>

using namespace nvinfer1;
using nvinfer1::plugin::fused_attention_kernelPluginCreator;
using nvinfer1::plugin::fused_attention_kernelPlugin;

PluginFieldCollection fused_attention_kernelPluginCreator::mFC{};
std::vector<PluginField> fused_attention_kernelPluginCreator::mPluginAttributes;
static bool triton_kernels_loaded = false;

// constructor
fused_attention_kernelPlugin::fused_attention_kernelPlugin( float sm_scale, int32_t num_heads )
{
  this->sm_scale = sm_scale;
  this->num_heads = num_heads;
  
}


// Parameterized constructor
fused_attention_kernelPlugin::fused_attention_kernelPlugin(const void* data, size_t length)
{
    const char *d = static_cast<const char*>(data), *a = d;

    read(d, sm_scale);
    read(d, num_heads);
    
    assert(d == a + getSerializationSize());
}


nvinfer1::IPluginV2DynamicExt* fused_attention_kernelPlugin::clone() const noexcept
{
  auto* plugin = new fused_attention_kernelPlugin(sm_scale, num_heads);
  plugin->setPluginNamespace(mNamespace.c_str());
  return plugin;
}


nvinfer1::DimsExprs fused_attention_kernelPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputDims, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs outputDims;
  if (outputIndex == 0) {
    outputDims = inputDims[0];
  }
  if (outputIndex == 1) {
    outputDims.nbDims = 3;
    outputDims.d[0] = inputDims[0].d[0];
    outputDims.d[1] = inputDims[0].d[1];
    outputDims.d[2] = inputDims[0].d[2];
  }
  if (outputIndex == 2) {
    outputDims.nbDims = 3;
    outputDims.d[0] = inputDims[0].d[0];
    outputDims.d[1] = inputDims[0].d[1];
    outputDims.d[2] = inputDims[0].d[2];
  }
  return outputDims;
}

bool fused_attention_kernelPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
  PLUGIN_ASSERT(nbInputs + nbOutputs == 6);
  PLUGIN_ASSERT(0 <= pos && pos < nbInputs + nbOutputs);
  PLUGIN_ASSERT(nbInputs == 3);
  PLUGIN_ASSERT(nbOutputs == 3);

  
  if (pos == 0) {return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;}
  
  if (pos == 1) {return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;}
  
  if (pos == 2) {return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;}
  

  
  if (pos == nbInputs + 0)
    return inOut[pos].type == DataType::kHALF && inOut[pos].format == TensorFormat::kLINEAR;
  
  if (pos == nbInputs + 1)
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
  
  if (pos == nbInputs + 2)
    return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
  
}


void fused_attention_kernelPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
  
}


size_t fused_attention_kernelPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
  return 0;
}

int fused_attention_kernelPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
  // input arguments
  
  const auto* Q_buf = reinterpret_cast<const half *>(inputs[ 0 ]);
  auto Q = reinterpret_cast<CUdeviceptr>(Q_buf);
  
  const auto* K_buf = reinterpret_cast<const half *>(inputs[ 1 ]);
  auto K = reinterpret_cast<CUdeviceptr>(K_buf);
  
  const auto* V_buf = reinterpret_cast<const half *>(inputs[ 2 ]);
  auto V = reinterpret_cast<CUdeviceptr>(V_buf);
  

  // outputs
  
  auto* Out_buf = reinterpret_cast<const half *>(outputs[ 0 ]);
  auto Out = reinterpret_cast<CUdeviceptr>(Out_buf);
  
  auto* L_buf = reinterpret_cast<const float *>(outputs[ 1 ]);
  auto L = reinterpret_cast<CUdeviceptr>(L_buf);
  
  auto* M_buf = reinterpret_cast<const float *>(outputs[ 2 ]);
  auto M = reinterpret_cast<CUdeviceptr>(M_buf);
  

  // dim size arguments
  int32_t batch_size = inputDesc[0].dims.d[0];
  int32_t seq_len = inputDesc[0].dims.d[2];

  // TODO: Check result code
  fused_attention_kernel(stream, Out, L, M, Q, K, V, sm_scale, batch_size, num_heads, seq_len, 0);

  return 0;
}


nvinfer1::DataType fused_attention_kernelPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
  
  if (index == 0) {
    return DataType::kHALF;
  }
  
  if (index == 1) {
    return DataType::kFLOAT;
  }
  
  if (index == 2) {
    return DataType::kFLOAT;
  }
  
}


const char* fused_attention_kernelPlugin::getPluginType() const noexcept
{
  return "fused_attention_kernelPlugin";
}

const char* fused_attention_kernelPlugin::getPluginVersion() const noexcept
{
  return "0";
}

int fused_attention_kernelPlugin::getNbOutputs() const noexcept
{
  return 3;
}

int fused_attention_kernelPlugin::initialize() noexcept
{
  if (triton_kernels_loaded) {
      return 0;
  }
  load_fused_attention_kernel();
  triton_kernels_loaded = true;
  return 0;
}

void fused_attention_kernelPlugin::terminate() noexcept {
  if (!triton_kernels_loaded) {
      return;
  }
  unload_fused_attention_kernel();
  triton_kernels_loaded = false;
}

size_t fused_attention_kernelPlugin::getSerializationSize() const noexcept
{
  size_t ret = 0;

  ret += sizeof(float);
  ret += sizeof(int32_t);
  

  return ret;

}

void fused_attention_kernelPlugin::serialize(void* buffer) const noexcept
{

    char *d = static_cast<char*>(buffer), *a = d;

    write(d, sm_scale);
    write(d, num_heads);
    
    assert(d == a + getSerializationSize());

}

void fused_attention_kernelPlugin::destroy() noexcept
{
    // This gets called when the network containing plugin is destroyed
    delete this;
}

void fused_attention_kernelPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* fused_attention_kernelPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



fused_attention_kernelPluginCreator::fused_attention_kernelPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();

    
    mPluginAttributes.emplace_back(PluginField("sm_scale", nullptr, PluginFieldType::kFLOAT32, 0));
    
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 0));
    

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}


const char* fused_attention_kernelPluginCreator::getPluginName() const noexcept
{
    return "fused_attention_kernelPlugin";
}

const char* fused_attention_kernelPluginCreator::getPluginVersion() const noexcept
{
    return "0";
}

const PluginFieldCollection* fused_attention_kernelPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* fused_attention_kernelPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
  const PluginField* fields = fc->fields;

  // declare parameters
  
    float sm_scale;
  
    int32_t num_heads;
  

    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;
  
        if (!strcmp(attrName, "sm_scale"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
            sm_scale = static_cast<float>(*(static_cast<const float*>(fields[i].data)));
        }
  
        if (!strcmp(attrName, "num_heads"))
        {
            PLUGIN_ASSERT(fields[i].type == PluginFieldType::kINT32);
            num_heads = static_cast<int32_t>(*(static_cast<const int32_t*>(fields[i].data)));
        }
  
    }

    try
    {
        auto* obj = new fused_attention_kernelPlugin(sm_scale, num_heads);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;

}


IPluginV2* fused_attention_kernelPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call fused_attention_kernelPlugin::destroy()
    try
    {
        auto* obj = new fused_attention_kernelPlugin(serialData, serialLength);
        obj->setPluginNamespace("tensorrt_llm");
        return obj;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void fused_attention_kernelPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* fused_attention_kernelPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}