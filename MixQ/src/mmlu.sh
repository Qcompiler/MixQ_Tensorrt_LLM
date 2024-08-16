

CMD="python"
set -ex
basepath=/dataset
#basepath=/mnt/octave/data/chenyidong/checkpoint
_dataset_path=/code/checkpoint/dataset


data_type=$2

cd examples


models=(  "Llama-2-7b"  ) 
ngpu=1
if [ ${data_type} == mix4 ] || [ ${data_type} == mix8 ]
    then 
    for model in "${models[@]}"
    do

        echo ${model}      
        bit=${data_type:3:3}
        CUDA_VISIBLE_DEVICES=$1    ${CMD} /code/tensorrt_llm/MixQ/src/examples/mmlu.py  --model_type ${data_type} \
        --hf_model_dir  ${basepath}/${model}  \
        --engine_dir /code/checkpoint/trt_enginesmix/tllm_checkpoint_${ngpu}gpu_fp16${model} --data_dir  /code/tensorrt_llm/MixQ/src/examples/data/mmlu

    done
fi


if [ ${data_type} == fp16  ] || [ ${data_type} == bitsandbytes  ]
    then 
    for model in "${models[@]}"
    do
        echo ${model}      
        export TRANSFORMERS_VERBOSITY=error
        CUDA_VISIBLE_DEVICES=$1    ${CMD} mmlu.py  --model_type ${data_type} --hf_model_dir  ${basepath}/${model}  \
        --engine_dir /code/checkpoint/trt_enginesfp16/tllm_checkpoint_${ngpu}gpu_fp16${model}  --data_dir  /code/tensorrt_llm/MixQ/src/examples/data/mmlu

    done
fi


if [ ${data_type} == awq  ]
    then 
    pip install transformers==4.35 
    for model in "${models[@]}"
    do
        echo ${model}      

        CUDA_VISIBLE_DEVICES=$1    ${CMD} mmlu.py  --model_type ${data_type} --hf_model_dir  ${basepath}/awqquant/${model}  

    done
    pip install transformers==4.38.2  
fi

cd ..