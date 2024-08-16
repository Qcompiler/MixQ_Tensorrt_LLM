

CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
CMD="python"
export http_proxy=127.0.0.1:7890 
export https_proxy=127.0.0.1:7890
set -ex
basepath=/dataset
#basepath=/mnt/octave/data/chenyidong/checkpoint
_dataset_path=/code/checkpoint/dataset


data_type=$2


for batch in   32 64 128 256 512
    do
    for seq in   512
      
        do
            
            models=(  "Llama-2-7b"  ) 
            if [ ${data_type} == mix4 ]
                then 
                    bit=4
                    echo  "---------run mix 4--------"
                    for model in "${models[@]}"
                        do
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                        ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                        ${basepath}/quant${bit}/${model} \
                        --quant_file ${basepath}/quant${bit}/${model} \
                        --batch_size ${batch} --bit ${bit} --_dataset_path ${_dataset_path}
   
                    done
     
            fi
            if [ ${data_type} == mix8 ]
                then 
                    bit=8
                    echo  "---------run mix  8--------"
                    for model in "${models[@]}"
                        do
                        echo ${model}          
                        CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892  \
                        ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                        ${basepath}/quant${bit}/${model} \
                        --quant_file ${basepath}/quant${bit}/${model} \
                        --batch_size ${batch} --bit ${bit} --_dataset_path ${_dataset_path}
   
                    done
     
            fi
            if [  ${data_type} == bitsandbytes   ]
                then 
 

                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                    ${basepath}/${model} \
                    --quant_file ${basepath}/${model} --batch_size ${batch} \
                    --_dataset_path ${_dataset_path}
                done

            fi

            if [  ${data_type} == fp16   ]
                then 
 

                for model in "${models[@]}"
                    do
                    echo ${model}          
                    CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890  \
                    ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                    ${basepath}/${model} \
                    --quant_file ${basepath}/${model} --batch_size ${batch} \
                    --_dataset_path ${_dataset_path}
                done

            fi
            if [ ${data_type} == awq ]
                then 
 
 
                for model in "${models[@]}"
                    do
                    echo ${model}
                    CUDA_VISIBLE_DEVICES=$1   ${CMD} benchflops.py  --model_type ${data_type} --model_path  \
                    ${basepath}/awqquant/${model} \
                    --quant_file ${basepath}/awqquant/${model} --batch_size ${batch} \
                    --_dataset_path ${_dataset_path}
                    
                done
            fi


         
        done 
done
