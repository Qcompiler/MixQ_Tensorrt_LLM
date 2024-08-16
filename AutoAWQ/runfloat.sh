
for batch in 1 
    do
    for seq in   32 
        do
            for model in 7  
                do
                # floaps awq
                #CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  
                # floaps mix
                CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_type mix --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /data/chenyidong/modeldata/Llama-2-7b-hf/checkpoint --batch_size ${batch} --seq_length ${seq}  
                
                # ppl mix
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_type mix --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /data/chenyidong/modeldata/Llama-2-${model}b-hf/checkpoint 
                
                # ppl awq
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_type awq --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq

            done
        done
done