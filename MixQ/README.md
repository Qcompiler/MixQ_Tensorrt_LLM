# MixQ
Enabling High Performance [ Int8 Int8 + fp16fp16] Mix GEMM with High Accuracy for LLama

## 运行示例

``` 
bash runthroughput.sh 0 fp16
bash runthroughput.sh 0 bitsandbytes
bash runthroughput.sh 0 mix4
bash runthroughput.sh 0 mix8
```  


# Overview


Please install all necessary Python packages
```
python3 -m pip install -r requirements.txt
```

## Compile

Please compile and install the MixQ by
cloning the MixQ code to the localmachine. Make sure the nvcc version is 12.1, the gcc version is 9.4.0 or above and the Python version is 3.11. Compile the cuda kernel of MixQ by:

```
cd quantkernel
python setup.py install
```

Compile the thrid-party tools of MixQ by:

```
git clone https://github.com/NetEase-FuXi/EETQ.git
cd EETQ
python setup.py install
```

Quantize. Please quantizate the LLMs before inferencing and reproducing the result. We provide the script 
```
bash quant.sh 0
```
for quantizing the mixed-precision 4-bit and 8-bit model.

## End-to-end throughput

Please compare the perplexity of the MixQ, the FP16 and the Bitsandbytes by running the script runthroughput.sh. For evaluating the end-to-end throughput of MixQ, please run the script runthroughput.sh. To reproduce the results one by one, please run the following CMD:
```
bash runthroughput.sh 0
```
or run
```
 CUDA_VISIBLE_DEVICES=0 python benchflops.py   --model_type {type} --model_path   {path}     --quant_file  {path}        --batch_size {batch}  --bit {bit} 
```
the options of  ```{type}``` are awq, bitsandbytes, fp16, quik and mix.

For example, if the readers would evaluate the AWQ throughput, please pass the following parameter:
```
--model_type awq --model_path  /dataset/Llama-2-7b-awq    --quant_file  /dataset/Llama-2-7b    --batch_size {batch}   
```

For example, if the readers would evaluate the FP16 throughput, please pass the following parameter:
```
--model_type fp16 --model_path  /dataset/Llama-2-7b   --quant_file  /dataset/Llama-2-7b    --batch_size {batch}   
```
where the path /dataset/Llama-2-7b contains the original LLMs file.


## End-to-end perplexity. 

Please compare the perplexity of the MixQ, the FP16 and the Bitsandbytes by running the script runfloat.sh. For evaluating the end-to-end perplexity of MixQ. Please run the script runthroughput.sh. To reproduce the results one by one, please run the following CMD:

```
CUDA_VISIBLE_DEVICES=0 python evalppl.py   --model_type mix --model_path  {path}     --quant_file  {path}    --n_ctx batch --n_batch batch  --eval_accuracy True
 ```



 


# Compare with the baselines


Moreover, the readers are encouraged to compile the baselines.


Please download the QUIK by:
``` 
git clone https://github.com/IST-DASLab/QUIK
cd QUIK
python setup.py install
```


Please download the Bitsandbytes by:
```
git clone https://github.com/TimDettmers/bitsandbytes.git \&\& cd bitsandbytes/
pip install -r requirements-dev.txt
cmake -DCOMPUTE\_BACKEND=cuda -S .
make
pip install .
python setup.py install
```

Please change the dir to AutoAWQ and run the following CMD:

```
cd AutoAWQ
python setup.py install
```

Please change the dir to SmoothQuant kernel and run the following CMD:

```
cd torch-int
python setup.py install
```