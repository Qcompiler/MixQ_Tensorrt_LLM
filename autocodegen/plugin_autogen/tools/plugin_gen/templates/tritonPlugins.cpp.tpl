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
#include "NvInfer.h"
#include "NvInferPlugin.h"

{% for header in headers %}
#include "[[ header ]]"
{% endfor %}

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <mutex>
#include <stack>
#include <unordered_set>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{

namespace
{

// This singleton ensures that each plugin is only registered once for a given
// namespace and type, and attempts of duplicate registration are ignored.
class TritonPluginCreatorRegistry
{
public:
    static TritonPluginCreatorRegistry& getInstance()
    {
        static TritonPluginCreatorRegistry instance;
        return instance;
    }

    template <typename CreatorType>
    void addPluginCreator(void* logger, const char* libNamespace)
    {
        // Make accesses to the plugin creator registry thread safe
        std::lock_guard<std::mutex> lock(mRegistryLock);

        std::string errorMsg;
        std::string verboseMsg;

        auto pluginCreator = std::make_unique<CreatorType>();
        pluginCreator->setPluginNamespace(libNamespace);

        nvinfer1::ILogger* trtLogger = static_cast<nvinfer1::ILogger*>(logger);
        std::string pluginType = std::string{pluginCreator->getPluginNamespace()}
            + "::" + std::string{pluginCreator->getPluginName()} + " version "
            + std::string{pluginCreator->getPluginVersion()};

        if (mRegistryList.find(pluginType) == mRegistryList.end())
        {
            bool status = getPluginRegistry()->registerCreator(*pluginCreator, libNamespace);
            if (status)
            {
                mRegistry.push(std::move(pluginCreator));
                mRegistryList.insert(pluginType);
                verboseMsg = "Registered plugin creator - " + pluginType;
            }
            else
            {
                errorMsg = "Could not register plugin creator -  " + pluginType;
            }
        }
        else
        {
            verboseMsg = "Plugin creator already registered - " + pluginType;
        }

        if (trtLogger)
        {
            if (!errorMsg.empty())
            {
                trtLogger->log(ILogger::Severity::kERROR, errorMsg.c_str());
            }

            if (!verboseMsg.empty())
            {
                trtLogger->log(ILogger::Severity::kVERBOSE, verboseMsg.c_str());
            }
        }
    }

    ~TritonPluginCreatorRegistry()
    {
        std::lock_guard<std::mutex> lock(mRegistryLock);

        // Release pluginCreators in LIFO order of registration.
        while (!mRegistry.empty())
        {
            mRegistry.pop();
        }
        mRegistryList.clear();
    }

private:
    TritonPluginCreatorRegistry() {}

    std::mutex mRegistryLock;
    std::stack<std::unique_ptr<IPluginCreator>> mRegistry;
    std::unordered_set<std::string> mRegistryList;

public:
    TritonPluginCreatorRegistry(TritonPluginCreatorRegistry const&) = delete;
    void operator=(TritonPluginCreatorRegistry const&) = delete;
};

template <typename CreatorType>
void initializeTritonPlugin(void* logger, const char* libNamespace)
{
    TritonPluginCreatorRegistry::getInstance().addPluginCreator<CreatorType>(logger, libNamespace);
}

} // namespace
} // namespace nvinfer1

// New Plugin APIs

extern "C"
{
    bool initLibNvInferPlugins(void* logger, const char* libNamespace)
    {
        {# create the registers for all the plugins #}
        {% for creator in plugin_creators %}
            nvinfer1::initializeTritonPlugin<nvinfer1::plugin::[[ creator ]]>(logger, libNamespace);
        {% endfor %}
        return true;
    }
} // extern "C"
