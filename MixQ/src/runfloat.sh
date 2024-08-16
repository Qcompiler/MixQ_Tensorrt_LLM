

#CMD="srun  -p twills -A h100 --gres=gpu:h100:1 --export=ALL python"
CMD="python "
#CMD="srun -N 1 --pty --gres=gpu:a100:2 -p octave -A public python"
export http_proxy=127.0.0.1:8892 
export https_proxy=127.0.0.1:8892
set -x


models=(    "Llama-2-7b" )
models=(    "Aquila2-7b" )

models=(    "falcon-7b" )
models=(    "vicuna-7b" )
models=(     "Llama-2-7b" )

basepath=""
_dataset_path=/code/checkpoint/dataset
data_type=$2
for batch in    512 
    do
    for seq in   64  
        do


            # data_types=( "fp16" )
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /home/dataset/llama-2/checkpoint/${model} \
            #         --quant_file /home/dataset/llama-2/checkpoint/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True


            #     done
            # done

            if [ ${data_type} == awq ]
                then 
                for model in "${models[@]}"
                    do
                    echo ${model}
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                    python evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
                    ${basepath}/awqquant/${model} \
                    --quant_file ${basepath}/awqquant/${model} \
                    --n_ctx $batch --n_batch $batch  --_dataset_path ${_dataset_path} --eval_accuracy True
                done
            fi

            if [ ${data_type} == mix8 ]
                then 
                    bit=8
                    echo  "---------run mix 8--------"
                    for model in "${models[@]}"
                        do
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${basepath}/quant${bit}/${model} \
                        --quant_file ${basepath}/quant${bit}/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --_dataset_path ${_dataset_path} --eval_accuracy True
   
                    done
     
            fi
            if [ ${data_type} == mix4 ]
                then 
                    bit=4
                    echo  "---------run mix 4--------"
                    for model in "${models[@]}"
                        do
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                        ${CMD} evalppl.py  --model_type ${data_type} --model_path  \
                        ${basepath}/quant${bit}/${model} \
                        --quant_file ${basepath}/quant${bit}/${model} \
                        --n_ctx ${batch}  --n_batch $batch  --_dataset_path ${_dataset_path} --eval_accuracy True
   
                    done
     
            fi
            # for data_type in "${data_types[@]}"
            #     do
            #     for model in "${models[@]}"
            #         do
            #         echo ${model}          
            #         CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
            #         ${CMD} evalppl.py --fp_features_num 128 --model_type ${data_type} --model_path  \
            #         /home/dataset/quant${bit}/${model} \
            #         --quant_file  /home/dataset/quant${bit}/${model} \
            #         --n_ctx $batch --n_batch $batch  --eval_accuracy True
            #     done
            # done
         
        done 
done
