

CMD="srun -p twills -A h100 --gres=gpu:h100:1"

set -x
for batch in    512
    do
    for seq in   64  
        do
            #model_type=Aquila2
            model_type=opt
            model_type=Mistral
            model_type=gpt-j
            #model_type=falcon
            model_type=Llama-2
            data_type=awq
            #data_type=bitsandbytesfp16
            models=(  "Baichuan2-7b"  "Aquila2-7b" "Llama-2-7b" )
            
            models=(  "Llama-2-70b"  )
            for model in "${models[@]}"
                do
                echo ${model}
                # floaps awq
                #CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq  --batch_size ${batch} --seq_length ${seq}  
                # floaps mix
                #CUDA_VISIBLE_DEVICES=$1 python   benchflops.py --model_type mix --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /data/chenyidong/Llama-2-${model}b-hf-mix --batch_size ${batch} --seq_length ${seq}  
                
                # for out in 1 2 4 8 16 32 64
                # do
                #     cp /home/chenyidong/quant/AutoAWQ/awq/modules/linear_raw.py /home/chenyidong/quant/AutoAWQ/awq/modules/linear.py 
                #     sed -i s/hello/${out}/g /home/chenyidong/quant/AutoAWQ/awq/modules/linear.py
                #     CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --outliers ${out} --model_type mix --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /data/chenyidong/modeldata/Llama-2-${model}b-hf/checkpoint 

                # done
                #out=128
                #cp /home/chenyidong/quant/AutoAWQ/awq/modules/linear_raw.py /home/chenyidong/quant/AutoAWQ/awq/modules/linear.py 
                #sed -i s/hello/"inputs.abs().max()*0.3333"/g /home/chenyidong/quant/AutoAWQ/awq/modules/linear.py
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --outliers ${out} --model_type mix --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /data/chenyidong/modeldata/Llama-2-${model}b-hf/checkpoint 

                # ppl mix
                 # ppl fp16
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_type bitsandbytesfp16 --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf  --n_ctx $batch --n_batch $batch
                
                #ppl mix


                ${CMD}  sleep 3600 
                CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD}   \
                python evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                /home/dataset/llama-2/checkpoint/${model} \
                --quant_file /home/dataset/llama-2/checkpoint/awqquant/${model} \
                --n_ctx $batch --n_batch $batch  --eval_accuracy True

                # CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
                # python evalppl.py --fp_features_num 128 --model_type mix --model_path  \
                # /home/dataset/llama-2/checkpoint/${model_type}-${model}b \
                # --quant_file /home/dataset/llama-2/checkpoint/quant/${model_type}-${model}b \
                # --n_ctx $batch --n_batch $batch   --eval_throughput True 

                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python  evalppl.py --fp_features_num 128 --model_type mix --model_path /data/chenyidong/llama/Baichuan2-${model}B-Base --quant_file /data/chenyidong/llama/Baichuan2-${model}B-Base --n_ctx $batch --n_batch $batch



                # ppl quik
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py  --model_type QUIK --fp_features_num 128 --model_path /data/chenyidong/modeldata/Llama-2-${model}b-hf  --w_bits 4 --w_clip --a_bits 4 --quant_file /data/chenyidong/QUIK/Llama-2-${model}b-hf --int8_down_proj    --n_ctx $batch --n_batch $batch


                # ppl awq
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_type awq --model_path /mnt/data/zhongrx/Llama-2-${model}b-hf --quant_file /mnt/data/chenyd/Llama-2-${model}b-awq --n_ctx $batch --n_batch $batch

                # ppl 百川
                #CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 python evalppl.py --model_type bitsandbytesfp16 --model_path /data/chenyidong/llama/Baichuan2-${model}B-Base  --n_ctx $batch --n_batch $batch

            done
         
        done 
done
