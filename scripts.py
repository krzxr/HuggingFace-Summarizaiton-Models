import os
print("setting up")
cache= '/proj/krong/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
directory = "/proj/krong/PreSumm/scienceDaily/ptNew/"

print(torch.cuda.memory_allocated("cuda"))
c=2
src,tgt=[],[]
model_name = "google/pegasus-multi_news"
directory = "/proj/krong/PreSumm/scienceDaily/ptNew/"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
print("tok",torch.cuda.memory_allocated("cuda"))
a=torch.load(directory+"sd.train.1.bert.pt")
for item in a:
    if len(item)<7:
        continue
    src.append("".join(item['src_txt']))
    tgt.append(item['tgt_txt'])
src_encoding = tokenizer.prepare_seq2seq_batch(src, truncation=True, padding='longest')
print("src",torch.cuda.memory_allocated("cuda"))
input_ids = src_encoding['input_ids'][:c].to("cuda")
print("id",torch.cuda.memory_allocated("cuda"))
mask = src_encoding['attention_mask'][:c].to("cuda")

print("m",torch.cuda.memory_allocated("cuda"))
labels = tokenizer.prepare_seq2seq_batch(tgt, truncation=True, padding='longest')['input_ids'][:c].to("cuda")
print("l",torch.cuda.memory_allocated("cuda"))
model = PegasusForConditionalGeneration.from_pretrained(model_name).to("cuda")
for index,param in enumerate(model.parameters()):
    if index<674:
        param.require_grad=False
    else:
        param.require_grad=True
print("postmodel",torch.cuda.memory_allocated("cuda"))
outputs = model(input_ids,attention_mask=mask,labels=labels)

print("output",torch.cuda.memory_allocated("cuda"))
