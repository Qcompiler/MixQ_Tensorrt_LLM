import os
import sys
sys.path.append("/home/chenyidong/quant/AutoAWQ")
from awq import AutoAWQForCausalLM
os.environ["WORLD_SIZE"] = "1"
import argparse
import pandas as pd
import torch
from auto_gptq.utils import Perplexity
from transformers import AutoTokenizer

if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    http_proxy=127.0.0.1:7890 https_proxy=127.0.0.1:7890 CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 python examples/benchmark/perplexity.py \
        --model_name  /mnt/data/zhongrx/Llama-2-7b \
        --model_basename gptq_model-4bit-128g \
        --is_quantized
    
    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_path", type=str,   help="Model path")
    parser.add_argument("--quant_file", type=str,   help="quant_file Model path")
    
    parser.add_argument("--model_type", type=str,  default='bitsandbytesfp16')


    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default='wikitext', help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default='test', help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default='text', help="Column in the dataset containing the text.")
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="Max memory used in each GPU.")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    
    parser.add_argument("--use_safetensors", action="store_true", help="Whether to use safetensors model file")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument("--disable_exllama", action="store_true", help="Whether to use disable exllama kernel")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=args.use_fast_tokenizer, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_path = args.model_path
    quant_file = args.quant_file

    if args.model_type == 'bitsandbytesfp16':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  fp16...")
    
        n_gpus = torch.cuda.device_count()
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        max_memory=max_memory,
        )
    if args.model_type == 'bitsandbytesmix4':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  mix4...")
    
        n_gpus = torch.cuda.device_count()
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int4_threshold=6.0,
        llm_int4_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=quantization_config
        )
    if args.model_type == 'bitsandbytesmix8':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model mix bit8...")
    
        n_gpus = torch.cuda.device_count()
        max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
        max_memory = {i: max_memory for i in range(n_gpus)}
        quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map='auto',
        max_memory=max_memory,
        quantization_config=quantization_config
        )
    if args.model_type == 'awq':
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        print(f" -- Loading model  awq...")
        
        model = AutoAWQForCausalLM.from_quantized(model_path, quant_file, fuse_layers=True,mix = False)

    if args.model_type == 'mix':
        model = AutoAWQForCausalLM.from_quantized(
            model_path, quant_file, fuse_layers=True,
            mix = True,
        )         

    
    ppl = Perplexity(model, tokenizer, args.dataset_path, args.dataset_name, args.split, args.text_column)
    allppl = ppl.calculate_perplexity(args.n_ctx, args.n_batch)

    data = pd.DataFrame(allppl)
    try:
        os.mkdir("output")
    except:
        pass
    data.to_csv("output/ppl_"+str(args.n_ctx)+"_"+args.model_type+"_"+model_path.split('/')[-1]+".csv")

    


