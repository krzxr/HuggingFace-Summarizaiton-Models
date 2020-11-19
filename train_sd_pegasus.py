import os
print("setting up")
cache= '/proj/krong/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache
from transformers import Trainer, TrainingArguments, PegasusForConditionalGeneration, PegasusTokenizer,T5Tokenizer, T5Model, T5ForConditionalGeneration, AutoConfig
import torch


from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
}




print("starting, loading tokenizer")
model_name = "google/pegasus-multi_news"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
print("finished loading tokenizer.")
class scienceDaily(torch.utils.data.Dataset):
    def __init__(self,types):
        src = []
        tgt = []
        directory = "/proj/krong/PreSumm/scienceDaily/ptNew/"
        for filename in os.listdir(directory):
            if not types in filename:
                continue 
            f = torch.load(directory+filename)
            for item in f:
                if len(item)<7:
                    continue
                src.append("".join(item['src_txt']))
                tgt.append(item['tgt_txt'])
                break # remove later
            print("finish loading",filename)
        src_encoding = tokenizer.prepare_seq2seq_batch(src, truncation=True, padding='longest')
        
        tgt_encoding = tokenizer.prepare_seq2seq_batch(tgt, truncation=True, padding='longest')
        self.encodings = src_encoding
        self.labels = tgt_encoding
    def __getitem__(self, idx):
        '''
        print([key for key,_ in self.encodings.items()])
        print(self.encodings.data)
        print(self.encodings.data['input_ids'][idx])
        print(self.encodings._encodings)
        '''
        print(self.encodings.data['attention_mask'])
        item = dict()
        item['input_ids']=torch.tensor(self.encodings.data['input_ids'][idx])
        item['attention_mask']=torch.tensor(self.encodings.data['attention_mask'][idx])
        item['labels'] = torch.tensor(self.labels.data['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels)
print("staring to load data")
train_dataset = scienceDaily("train")
test_dataset = scienceDaily("valid")
print("finished loading data")
#for param in model.base_model.parameters():
#    param.requires_grad = False
print("starting training")
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=10,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

print("starting, loading models")
model = PegasusForConditionalGeneration.from_pretrained(model_name)

print("finished loading models")
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset,           # evaluation dataset
    compute_metrics = compute_metrics
)
trainer.train()

print("finish training. trained parameters are in results.")
