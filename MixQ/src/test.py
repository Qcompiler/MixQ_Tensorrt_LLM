from datasets import load_dataset
import evaluate


import time
#time.sleep(3600)
import os
cache_dir="/home/chenyidong/checkpoint/dataset/"
#evaluate.load("rouge",cache_dir=cache_dir)
dataset = load_dataset("json", data_files="/code/checkpoint/val.jsonl.zst", split="train")
#evaluate.load("/home/chenyidong/checkpoint/dataset/rouge/rouge.py")


exit(0)
_split = "test"
print(cache_dir)

dataset_name = "ccdv/cnn_dailymail"
dataset_revision = "3.0.0"
dataset_input_key = 'article'
dataset_output_key = 'highlights'
dataset_split = 'test'
dataset = load_dataset(dataset_name,
                           dataset_revision,
                           cache_dir=cache_dir,
                           split=dataset_split)

# data = load_dataset(os.path.join(cache_dir,"wikitext"), name='wikitext-2-raw-v1', 
#             cache_dir=cache_dir, split=_split) 
