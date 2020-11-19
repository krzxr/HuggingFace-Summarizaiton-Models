import os
import time
print("setting up")
cache= '/proj/krong/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache
from transformers import AdamW,PegasusForConditionalGeneration, PegasusTokenizer
import torch


print("starting, loading tokenizer")
model_name = "google/pegasus-multi_news"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
print("finished loading tokenizer.")
c=100
epochs = 10
device = "cpu"
directory = "/proj/krong/PreSumm/scienceDaily/ptNew/"
phrases = ['train','valid']
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
optim = AdamW(model.parameters(), lr=5e-5)
for index,param in enumerate(model.parameters()):
    if index<674:
        param.require_grad=False
    else:
        param.require_grad=True
for epoch in range(epochs):
    for phrase in phrases:
        if phrase == "train":
            model.train()
        else:
            if epoch%4!=0:
                continue
            model.eval()
        src, tgt = [],[]
        for filename in os.listdir(directory):
            if not phrase in filename:
                continue
            f = torch.load(directory+filename)
            for item in f:
                if len(item)<7:
                    continue
                src.append("".join(item['src_txt']))
                tgt.append(item['tgt_txt'])
            
        src_encoding = tokenizer.prepare_seq2seq_batch(src, truncation=True, padding='longest')
        tgt_encoding = tokenizer.prepare_seq2seq_batch(tgt, truncation=True, padding='longest')        
        for i in range(0,len(src),c):
            print("epoch",epoch,"sample ",i)
            #print(torch.cuda.memory_allocated(device))
            optim.zero_grad()
            input_ids = src_encoding['input_ids'][i:i+c].to(device)
            masks = src_encoding['attention_mask'][i:i+c].to(device)
            labels = tokenizer.prepare_seq2seq_batch(tgt, truncation=True, padding='longest')['input_ids'][i:i+c].to(device)
            print("epoch",epoch,"sample",i,"finis loading data")            
            #print(torch.cuda.memory_allocated(device))
            with torch.set_grad_enabled(phrase=="train"): 
                outputs = model(input_ids,attention_mask=masks,labels=labels)
                loss = outputs[0]
                print("epoch",epoch,"sample",i,"finish outputs")
                #print(torch.cuda.memory_allocated(device))

                if phrase=="train":
                    loss.backward()
                    optim.step()
                    print("epoch",epoch,"sample",i,"finish back prop")
                    #print(torch.cuda.memory_allocated(device))
                print(loss.detach())
                del loss
                del input_ids
                del masks
                del labels
                print("epoch",epoch,"sample",i,"finis cleaning")
                #print(torch.cuda.memory_allocated(device))
                torch.cuda.empty_cache()
        if epoch % 5==0:
            name = "model_"+str(int(time.time()))+"epoch"+str(epoch)+".pt"
            print("saving",name)
            torch.save(model,name)
name = "model_"+str(int(time.time()))+"epoch"+str(epochs)+".pt"
print("saving",name)
torch.save(model,name)
