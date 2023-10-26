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

import json
import pandas as pd
wiki = []
with open('./data/wiki_data.json', 'r',encoding='utf8') as f:
    lines = f.readlines()
    for line in lines:
        wiki.append(json.loads(line))
wiki = pd.DataFrame(wiki)
wiki['title_text'] = wiki.apply(lambda row : ' '.join(str(row[field]) for field in ['title', 'content']),axis=1)
wiki = wiki[['page_id', 'title_text']]
wiki.drop_duplicates(inplace=True)
wiki = wiki.reset_index(drop=True)
wiki['page_id'] = wiki['page_id'].apply(lambda x : int(x))

BERT_PATH = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "./save/recall/recall_epoch100.bin"
PROMPT_LEN = 512
WIKI_LEN = 512
MAX_LEN = 512
BATCH_SIZE = 128
DEVICE = 'cuda'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import os
from transformers import AutoTokenizer
import multiprocessing
def process_data(path):
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
    data = pd.read_parquet(os.path.join('./wiki_data/', path))
    res_ids = []
    res_page_id = []
    for i in range(len(data)):
        inputs = data.loc[i, 'title'] + data.loc[i, 'text']
        page_id = data.loc[i, 'id']
        inputs = tokenizer.encode(inputs, add_special_tokens=False)
        if len(inputs) > 510:
            inputs = [101] + inputs[:510] + [102]
        else:
            inputs = [101] + inputs + [102]
        res_ids.append(inputs)
        res_page_id.append(page_id)
    return res_ids, res_page_id
list_dir = os.listdir('./wiki_data')
pool = multiprocessing.Pool(processes=len(list_dir))
results = []
for idx, path in enumerate(list_dir):
    result = pool.apply_async(process_data,args=(path, ))
    results.append(result)
ids = []
page_ids = []
for result in results:
    res_id, res_page_id = result.get()
    ids.extend(res_id)
    page_ids.extend(res_page_id)

data = pd.DataFrame({'page_id':page_ids, 'ids':ids})
data.to_parquet('./data/wiki_data_tokenizer.parquet')