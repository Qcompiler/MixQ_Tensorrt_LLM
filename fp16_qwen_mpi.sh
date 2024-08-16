

set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Qwen2-72B")
type='fp16'



ngpu=5
for model in "${models[@]}"
    do

    model_dir=/dataset/${model}
    output_dir=/octave/checkpoint/checkpoint${type}/tllm_checkpoint_${ngpu}gpu_fp16${model}
    engine_dir=/octave/checkpoint/trt_engines${type}/tllm_checkpoint_${ngpu}gpu_fp16${model}

    CUDA_VISIBLE_DEVICES=$1  python  quantize_qwen.py --model_dir  ${model_dir}\
    --output_dir ${output_dir}   --dtype float16  --pp_size ${ngpu}   --load_model_on_cpu

    CUDA_VISIBLE_DEVICES=$1 trtllm-build --checkpoint_dir ${output_dir} \
            --output_dir  ${engine_dir} \
            --gemm_plugin float16 --max_batch_size 1  

    CUDA_VISIBLE_DEVICES=$1    mpirun -n ${ngpu} --allow-run-as-root python  summarize.py --test_trt_llm \
            --hf_model_dir ${model_dir} \
            --data_type fp16 \
            --engine_dir ${engine_dir}  --max_input_length 2048
done 

