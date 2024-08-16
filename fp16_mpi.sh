

set -x
CMD="srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public  "
models=("Llama-2-70b")
type='fp16'




for model in "${models[@]}"
    do

    model_dir=/dataset/${model}
    output_dir=/octave/checkpoint/checkpoint${type}/tllm_checkpoint_1gpu_fp16${model}
    engine_dir=/octave/checkpoint/trt_engines${type}/tllm_checkpoint_1gpu_fp16${model}

    CUDA_VISIBLE_DEVICES=$1   \
    python  quantize.py --model_dir  ${model_dir}\
    --output_dir ${output_dir}   --dtype float16   --pp_size 4 --load_model_on_cpu

    CUDA_VISIBLE_DEVICES=$1 trtllm-build --checkpoint_dir ${output_dir} \
            --output_dir  ${engine_dir} \
            --gemm_plugin float16

    CUDA_VISIBLE_DEVICES=$1    mpirun -n 4 --allow-run-as-root python  summarize.py --test_trt_llm \
            --hf_model_dir ${model_dir} \
            --data_type fp16 \
            --engine_dir ${engine_dir}
#     CUDA_VISIBLE_DEVICES=4,5,6,7 mpirun -n 4 --allow-run-as-root \
#     python  run.py \
#     --max_output_len 128 \
#     --max_input_length 10240 \
#     --input_file pg64317_sanitized.txt \
#     --engine_dir ${engine_dir} \
#     --tokenizer_dir  ${model_dir} 

done 


