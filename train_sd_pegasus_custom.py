import os
import time
print("setting up")
cache= 'hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache
from transformers import AdamW,PegasusForConditionalGeneration, PegasusTokenizer
import torch


print("starting, loading tokenizer")
model_name = "google/pegasus-multi_news"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
print("finished loading tokenizer.")
import random
c=1
epochs = 1
device = "cuda"
directory = "sd/"
phrases = ['train','valid']

files = [fn for fn in os.listdir(directory+'txt/') if fn[-4:] == '.txt']

random.shuffle(files)

div = 11
n = len(files)
fileTypes = { 'train': [files[:(n//div)]], 
              'valid': [files[(n//div)*i:min(n//div*(i+1), n)] for i in range(1, div)]
}

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
        curr_batch = fileTypes[phrase][random.randrange(len(fileTypes[phrase]))]
        print('starting to get data')
        for fn in curr_batch:
            txt_f = open(directory+'txt/'+fn,'r')
            abs_f = open(directory+'abs/'+fn[:-4]+'_abstract.txt','r')
            src.append(txt_f.read())
            tgt.append(abs_f.read())
            
        src_encoding = tokenizer.prepare_seq2seq_batch(src, truncation=True, padding='longest')
        tgt_encoding = tokenizer.prepare_seq2seq_batch(tgt, truncation=True, padding='longest')
        print('finished encoding data')
        print(torch.cuda.memory_allocated(device))
        for i in range(0,len(src),c):
            print("epoch",epoch,"sample ",i)
            print(torch.cuda.memory_allocated(device))
            optim.zero_grad()
            input_ids = torch.tensor(src_encoding['input_ids'][i:i+c]).to(device)
            masks = torch.tensor(src_encoding['attention_mask'][i:i+c]).to(device)
            labels = torch.tensor(tgt_encoding['input_ids'][i:i+c]).to(device)
            print("epoch",epoch,"sample",i,"finish loading data")            
            print(torch.cuda.memory_allocated(device))
            with torch.set_grad_enabled(phrase=="train"): 
                outputs = model(input_ids,attention_mask=masks,labels=labels)
                loss = outputs[0]
                print("epoch",epoch,"sample",i,"finish outputs")
                print(torch.cuda.memory_allocated(device))
                if phrase=="train":
                    loss.backward()
                    optim.step()
                    print("epoch",epoch,"sample",i,"finish back prop")
                    print(torch.cuda.memory_allocated(device))
                print(loss.detach())
                del loss
                del input_ids
                del masks
                del labels
                print("epoch",epoch,"sample",i,"finis cleaning")
                print(torch.cuda.memory_allocated(device))
                torch.cuda.empty_cache()
        if epoch % 5==0:
            name = "model_"+str(int(time.time()))+"epoch"+str(epoch)+".pt"
            print("saving",name)
            torch.save(model,name)
name = "model_"+str(int(time.time()))+"epoch"+str(epochs)+".pt"
print("saving",name)
torch.save(model,name)
