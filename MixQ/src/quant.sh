

CMD=" srun  -N 1 --pty --gres=gpu:a100:1 -p octave -A public python "
#CMD=" python" 
 
set -x

# model=65      
# CUDA_VISIBLE_DEVICES=$1   http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
# python examples/basic_quant_mix.py  \
# --model_path /home/dataset/llama-2/checkpoint/Llama-${model}b \
# --quant_file /home/dataset/llama-2/checkpoint/quant/Llama-${model}b

basepath=/mnt/octave/data/chenyidong/checkpoint
models=(  "Baichuan2-7b"  "Baichuan2-13b" "Aquila2-7b" "Llama-2-7b"  "Mistral-7b" )
models=(  "Llama-2-7b" "vicuna-7b"  "Aquila2-7b" "falcon-7b" "Baichuan2-7b")

models=(  "Llama-2-7b")

# for model in "${models[@]}"
#         do
#         echo ${model}
#         http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 ${CMD} \
#           examples/basic_quant_mix.py  \
#         --model_path /mnt/octave/data/chenyidong/checkpoint/${model} \
#         --quant_file /home/dataset/quant8/${model} --w_bit 8
# done
for bit in  4 8 
  do
  for model in "${models[@]}"
          do
          echo ${model}
          http_proxy=127.0.0.1:8892 https_proxy=127.0.0.1:8892 ${CMD} \
            examples/basic_quant_mix.py  \
          --model_path ${basepath}/${model} \
          --quant_file ${basepath}/quant${bit}/${model} --w_bit ${bit}
  done
done


