
 
set -x

model=( Llama-2-7b )
model=( Aquila2-7b )
model=( Qwen2-72B )
#model=( Llama-2-1b )
CUDA_VISIBLE_DEVICES=$1 python examples/smooth_quant_get_act.py  \
        --model-name /dataset/${model}  \
        --output-path /code/tensorrt_llm/act_scales/${model}.pt  \
        --dataset-path /code/checkpoint/val.jsonl.zst 

