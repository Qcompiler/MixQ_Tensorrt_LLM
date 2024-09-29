set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "


models=("Llama-2-7b")
models=("qwen2-7b-instruct")
# pkill -9 python
# pkill -9 bash.sh
ngpu=1
type=mix
for model in "${models[@]}"
    do

    model_dir=/dataset/${model}
    quant_dir=/octave/checkpoint/checkpoint${type}/tllm_checkpoint_${ngpu}gpu_fp16${model}
    out_dir=/octave/checkpoint/checkpoint${type}/tllm_checkpoint_${ngpu}gpu_fp16${model}

    rm -r ${quant_dir}
    rm -r ${out_dir}
    if [[ "$model" == *"qwen"* ]]; then
        CUDA_VISIBLE_DEVICES=$1  python  quantize_qwen.py --model_dir  ${model_dir} --output_dir  ${quant_dir}  --dtype float16  --load_model_on_cpu   --mix int8_mix   
    fi

    if [[ "$model" == *"Llama"* ]]; then
        CUDA_VISIBLE_DEVICES=$1  python  quantize.py --model_dir  ${model_dir} --output_dir  ${quant_dir}  --dtype float16  --load_model_on_cpu   --mix int8_mix   
    fi


    CUDA_VISIBLE_DEVICES=$1 trtllm-build --checkpoint_dir ${quant_dir} \
       --output_dir ${out_dir} --max_input_len  2048 \
           --gemm_plugin float16 

   CUDA_VISIBLE_DEVICES=$1   python  summarize.py --test_trt_llm \
                      --hf_model_dir ${model_dir} \
                      --data_type fp16 \
                      --engine_dir ${out_dir}

done 

