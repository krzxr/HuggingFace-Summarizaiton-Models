import os
cache= '/proj/krong/hf_cache/'
os.environ['TRANSFORMERS_CACHE'] = cache
from transformers import PegasusForConditionalGeneration, PegasusTokenizer,T5Tokenizer, T5Model, T5ForConditionalGeneration, AutoConfig
import torch

def summarizeP(src_text,variant="xsum",device=None):
    model_name = "google/pegasus-"
    model_name += variant
    torch_device = ('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest').to(torch_device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

def summarizeT(src_text,variant="small"):
    model_name = "t5-"
    model_name += variant
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
    input_ids = tokenizer(src_text,return_tensors="pt").input_ids
    output = model.generate(input_ids)
    output = [[tokenizer._convert_id_to_token(int(output[i][j])) for j in range(len(output[i]))] for i in range(len(output))]
    tgt_text = [tokenizer.convert_tokens_to_string(output[i]) for i in range(len(output))]
    return tgt_text
