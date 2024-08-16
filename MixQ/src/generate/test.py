from transformers import AutoModelForCausalLM, AutoTokenizer


model="/mnt/octave/data/chenyidong/checkpoint/Llama-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", trust_remote_code=True)
inputs = tokenizer('where is the captical of China? ', return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
