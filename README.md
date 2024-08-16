# Project For lenovo

Please clone the project
```
git clone https://github.com/Qcompiler/MixQ_Tensorrt_LLM.git
```

Please Pull the docker 

```
docker pull registry.cn-hangzhou.aliyuncs.com/dongdongchen/dongdong:v1
```

Please run the docker

```
bash -c " nvidia-smi; docker run --rm -it --ipc=host -p 6789:22 \
-v /home/chenyidong/lianxiang/lianxiangTRT/:/code/tensorrt_llm   \
-v  /mnt/octave/data/chenyidong/checkpoint:/dataset    \
-v /home/chenyidong/checkpoint:/code/checkpoint \
-v /mnt/octave/data/chenyidong/lianxiang/checkpoint:/octave/checkpoint \
               --ulimit memlock=-1 --ulimit    stack=67108864             \
                           --gpus=all       \
                       --env 'CCACHE_DIR=/code/tensorrt_llm/cpp/.ccache'            \
                            --env 'CCACHE_BASEDIR=/code/tensorrt_llm'              \
                                                    --workdir /app/tensorrt_llm     \
                                                            --hostname hpc-release \
                  --name tensorrt_llm-release-zhanghy                             \
                                                           --tmpfs /tmp:exec      \
              registry.cn-hangzhou.aliyuncs.com/dongdongchen/dongdong:v1     "

```

Please setup the env  by

cd /code/tensorrt_llm
source load.sh

Please run FP16 baseline with Llama-2-7B by

```
bash fp16.sh

```


Please run optimized model  Llama-2-7B by

```
bash mix.sh

```


## benchmark

The pytoch benchmark is shown in https://github.com/Qcompiler/MIXQ

Please run the flowing script to reproduce the benchmark of AWQ MIXQ bitsandbytes FP16

``` cd MixQ/src ```


Run:

``` bash runthroughput.sh 0 ```

For Llama-2-70B model:

<img src="figure/throughput-llama70b_revised_baseline.jpg"  align = "center"  width="600" />


For Llama-2-7B model:

<img src="figure/throughput-llama7b.jpg"  align = "center"  width="600" />
