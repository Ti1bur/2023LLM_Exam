import math
import json
import glob
import collections
import random
from pathlib import Path
import pandas as pd
import numpy as np
import os
import copy
from tqdm.auto import tqdm
import pickle
import gc
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
import torch
# pip install prefetch_generator
from prefetch_generator import BackgroundGenerator
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from  transformers import AdamW, AutoTokenizer,AutoModel
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
SEED=2020
seed_everything(SEED)

# BERT_PATH = "/root/bert_path/sentence-transformers_all-MiniLM-L6-v2"
BERT_PATH = "/root/bert_path/sentence-transformer-all-mpnet-base-v2"
# BERT_PATH = "./pretrain_models/deberta-v3-xsmall"
PROMPT_LEN = 512
WIKI_LEN = 512
MAX_LEN = 512
BATCH_SIZE = 2048
DEVICE = 'cuda'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,5,6,7'
data = pd.read_parquet('./small_wiki_data_base/680w_kmeans_split_2500_k35_clear.parquet')
data = data.reset_index(drop=True)
data['context'] = data.apply(lambda row : ' '.join(str(row[x]) for x in ['title','text']),axis=1)
# data['context'] = data.apply(lambda row: row['title'] + ' '.join(list(row['categories'])) + row['text'],axis=1)
# data = data[['id','file','context']]

import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class RecallModel(nn.Module):
    def __init__(self):
        super(RecallModel, self).__init__()
        self.bert_model = AutoModel.from_pretrained(BERT_PATH)

        self.mean_pooler = MeanPooling()

    def mask_mean(self, x, mask=None):
        if mask != None:
            mask_x = x * (mask.unsqueeze(-1))
            x_sum = torch.sum(mask_x, dim=1)
            re_x = torch.div(x_sum, torch.sum(mask, dim=1).unsqueeze(-1))
        else:
            x_sum = torch.sum(x, dim=1)
            re_x = torch.div(x_sum, x.size()[1])
        return re_x

    def forward(self, input_ids):
        attention_mask = input_ids > 0
        out = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
        x = self.mean_pooler(out, attention_mask)

        # x = out[:, 0, :]
        return x
from functools import partial
from torch.utils.data import DataLoader
dataloader_class = partial(DataLoader, pin_memory=True, num_workers=32)

model= RecallModel()
from collections import OrderedDict
def load_param(model_path):
    state_dict = torch.load(model_path, map_location='cpu')
    params = OrderedDict()
    for name, param in state_dict.items():
        name = '.'.join(name.split('.')[1:])
        params[name] = param
    return params
model.load_state_dict(load_param('./save/recall_base/recall_new_data_hard_example1.bin'))
model.to(DEVICE)
model = torch.nn.parallel.DataParallel(model)

from tqdm.auto import tqdm
class LLMRecallDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
        self.data = data
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self,index):
        ids = self.data.loc[index, 'context']
        ids = self.tokenizer.encode(ids, add_special_tokens=False)
        if len(ids) > 510:
            ids = [101] + ids[:510] + [102]
        else:
            ids = [101] + ids + [102]
        return ids
    def collate_fn(self, batch):
        def sequence_padding(inputs, length=None, padding=0):
            """
            Numpy函数，将序列padding到同一长度
            """
            if length is None:
                length = max([len(x) for x in inputs])

            pad_width = [(0, 0) for _ in np.shape(inputs[0])]
            outputs = []
            for x in inputs:
                x = x[:length]
                pad_width[0] = (0, length - len(x))
                x = np.pad(x, pad_width, 'constant', constant_values=padding)
                outputs.append(x)

            return np.array(outputs, dtype='int64')

        batch_ids = torch.tensor(sequence_padding(batch), dtype=torch.long)
        
        return batch_ids

        
class DataLoaderX(torch.utils.data.DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())  

    
def get_loader(prompt,batch_size,train_mode=True,num_workers=4):
    ds_df = LLMRecallDataSet(prompt)
    # loader = DataLoaderX(ds_df, batch_size=batch_size if train_mode else batch_size//2, shuffle=train_mode, num_workers=num_workers,pin_memory=True,
    #                                      collate_fn=ds_df.collate_fn, drop_last=train_mode)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=False,collate_fn=ds_df.collate_fn,num_workers=num_workers)
    loader.num = len(ds_df)
    return loader

def debug_label():
    loader=get_loader(data.loc[:100].reset_index(drop=True),batch_size=2,train_mode=True,num_workers=0)
    model= RecallModel()
    print('models paramters:', sum(p.numel() for p in model.parameters()))
    for token_ids in loader:
        # print(token_ids)
        # print(labels)
        prob=model(token_ids)
        print(prob.shape)
        break

loader = get_loader(data, 7000, False, num_workers=32)
from torch.cuda.amp import autocast
import faiss

index = faiss.IndexFlatIP(768)
model.eval()
idx = 0
with torch.no_grad():
    for batch in tqdm(loader):
        ids = batch
        ids = ids.to(DEVICE)
        with autocast():
            output = model(ids).cpu().detach().numpy()
        faiss.normalize_L2(output)
        index.add(output)
faiss.write_index(index, './wiki_index/680w_kmeans_split_2500_k35_clear只有增量.bin')