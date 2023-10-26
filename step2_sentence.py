from __future__ import annotations
import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf

from collections.abc import Iterable

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

from dataclasses import dataclass
from typing import Optional, Union
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader
import math
import json
import glob
import collections
import random
from pathlib import Path
import os
import copy
import pickle
import gc
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
from functools import partial
from model import EMA
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from  transformers import AdamW, AutoTokenizer,AutoModel
import torch.nn as nn
import torch.nn.functional as F
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

def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)

DATA_PATH = "../data/"
BERT_PATH = "/root/bert_path/sentence-transformers_all-MiniLM-L6-v2"
MODEL_PATH = "./save/recall/2023_recall_v1_add_text_nice_valid.pkl"
PROMPT_LEN = 512
WIKI_LEN = 512
MAX_LEN = 512
BATCH_SIZE = 200
DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = '1,2'

prompt = pd.read_csv('./data/7w8_crawl_dataset.csv')
prompt['prompt_answer'] = prompt.apply(lambda row: ' '.join(str(row[field]) for field in ['prompt', 'A', 'B', 'C', 'D', 'E']), axis=1)
# processed_wiki_text_data = process_documents(prompt.wiki_text.values, prompt.id.values)

# splits = []
# for i in tqdm(range(len(prompt))):
#     prompt.loc[i,'answer'] = prompt.loc[i, prompt.loc[i,'answer']]
#     splits.append(processed_wiki_text_data[processed_wiki_text_data['document_id'] == i]['text'].tolist())
# prompt['sentence'] = splits

# prompt = prompt[prompt['sentence'].apply(lambda x : len(x) != 0)].reset_index(drop=True)

# import pylcs
# def rouge_l(a, b):
#     if b == ' ':
#         return 0
#     lcs = pylcs.lcs(a, b)
#     r = lcs / len(a)
#     return r

# sentence = []
# s = set()
# for i in tqdm(range(len(prompt))):
#     choices = [x for x in prompt.loc[i,'sentence'] if len(x) > 0]
#     if len(choices) == 0:
#         sentence.append(' ')
#         continue
#     for item in choices:
#         s.add(item)
#     scores = []
#     for item in choices:
#         scores.append(rouge_l(prompt.loc[i,'answer'].lower(), item))
#     sentence.append(choices[scores.index(max(scores))])
# sentence_df = pd.DataFrame({'sentence_answer':list(s)})
# prompt['sentence_answer'] = sentence

prompt['sentence_answer'] = prompt['wiki_text']
sentence_df = pd.DataFrame({'sentence_answer':list(set(prompt['wiki_text'].tolist()))})

class LLMRecallDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
        self.query = []
        self.answer = []
        print('加载数据集中')
        for i in tqdm(range(len(data))):
            query = data.loc[i, 'prompt_answer']
            answer = data.loc[i, 'sentence_answer']
            query_id = self.tokenizer.encode(query, add_special_tokens=False)
            answer_id = self.tokenizer.encode(answer, add_special_tokens=False)
            if len(query_id) > 510:
                query_id = [101] + query_id[:510] + [102]
            else:
                query_id = [101] + query_id + [102]
            if len(answer_id) > 510:
                answer_id = [101] + answer_id[:510] + [102]
            else:
                answer_id = [101] + answer_id + [102]
            self.query.append(query_id)
            self.answer.append(answer_id)
    def __len__(self):
        return len(self.query) 
    
    def __getitem__(self,index):
        return self.query[index], self.answer[index]
    
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
        batch_query, batch_answer = [], []
        
        for item in batch:
            query, answer = item
            batch_query.append(query)
            batch_answer.append(answer)
        batch_query = torch.tensor(sequence_padding(batch_query), dtype=torch.long)
        batch_answer = torch.tensor(sequence_padding(batch_answer), dtype=torch.long)
        
        return batch_query, batch_answer

        
class DataLoaderX(torch.utils.data.DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())  

    
def get_loader(prompt,batch_size,train_mode=True,num_workers=4):
    ds_df = LLMRecallDataSet(prompt)
    # loader = DataLoaderX(ds_df, batch_size=batch_size if train_mode else batch_size // 2, shuffle=train_mode, num_workers=num_workers,
    #                      pin_memory=True,
    #                      collate_fn=ds_df.collate_fn, drop_last=train_mode)
    dataloader_class = partial(DataLoader, pin_memory=True)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=train_mode, collate_fn=ds_df.collate_fn, num_workers=num_workers)
    return loader

def debug_loader(prompt, batch_size):
    loader=get_loader(prompt,batch_size,train_mode=True,num_workers=0)
    for token_ids,labels in loader:
        print(token_ids)
        print(labels)
        break
    return loader

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

def debug_label():
    loader=get_loader(prompt,batch_size=2,train_mode=True,num_workers=0)
    model= RecallModel()
    print('models paramters:', sum(p.numel() for p in model.parameters()))
    for token_ids,labels in loader:
        # print(token_ids)
        # print(labels)
        prob=model(token_ids)
        print(prob)
        break
    
def SimCSE_loss(topic_pred,content_pred,tau=0.05):
    similarities = F.cosine_similarity(topic_pred.unsqueeze(1), content_pred.unsqueeze(0), dim=2) # B,B
    y_true = torch.arange(0,topic_pred.size(0)).to(DEVICE)
    # similarities = similarities - torch.eye(pred.shape[0]) * 1e12
    similarities = similarities / tau
    loss=F.cross_entropy(similarities, y_true)
    return torch.mean(loss)
from torch.cuda.amp import autocast, GradScaler
import faiss
def valid(model):
    model.eval()
    index = faiss.IndexFlatIP(384)
    prompt_embed = []
    with torch.no_grad():
        for batch in val_loader:
            topic_inputs, content_inputs = (_.to(DEVICE) for _ in batch)
            with autocast():
                topic_pred = model(topic_inputs).cpu().numpy()
            faiss.normalize_L2(topic_pred)
            prompt_embed.append(topic_pred)
        for batch in tqdm(sentence_loader):
            topic_inputs, content_inputs = (_.to(DEVICE) for _ in batch)
            with autocast():
                content_pred = model(content_inputs).cpu().numpy()
            faiss.normalize_L2(content_pred)
            index.add(content_pred)
    prompt_embed = np.concatenate(prompt_embed, axis=0)
    search_score, search_index = index.search(prompt_embed, 1)
    cnt = 0
    for i in range(len(val)):
        label = val.loc[i, 'sentence_answer']
        pred = sentence_df.loc[search_index[i][0],'sentence_answer']
        if label == pred:
            cnt += 1
        
    return cnt / len(val)

def trainer(train_dataloader,val_dataloader,model, epochs, fold=0,
            accumulation_steps=1, early_stop_epochs=5, device='cpu'):
    ########早停
    no_improve_epochs = 0

    ########优化器 学习率
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    crf_p = [n for n, p in param_optimizer if str(n).find('crf') != -1]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and n not in crf_p], 'weight_decay': 1e-6},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and n not in crf_p], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in crf_p], 'lr': 2e-3, 'weight_decay': 0.8},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)

    scaler = GradScaler()
    criterion = nn.BCEWithLogitsLoss()

    train_len = len(train_dataloader)

    ema_inst = EMA(model, 0.95)
    ema_inst.register()

    best_score = 0
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        bar = tqdm(train_dataloader)
        for i, inputs in enumerate(bar):
            with autocast():
                topic_inputs, content_inputs = (_.to(device) for _ in inputs)
                # print(topic_inputs.size())
                # print(content_inputs.size())
                topic_pred = model(topic_inputs)
                content_pred = model(content_inputs)
                # print(topic_pred.size())
                # print(content_pred.size())
                loss = SimCSE_loss(topic_pred, content_pred)
            scaler.scale(loss).backward()
            losses.append(loss.item())
            if (i + 1) % accumulation_steps == 0 or (i + 1) == train_len:
                scaler.step(optimizer)
                if ema_inst:
                    ema_inst.update()
                optimizer.zero_grad()
                scaler.update()
            bar.set_postfix(loss_mean=np.array(losses).mean(), epoch=epoch)

        if ema_inst:
            ema_inst.apply_shadow()
        score = valid(model)
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f'./save/recall_sentence/recall_{fold}_best_11.bin')
        print(f'score:{score} best_score:{best_score}')
        if ema_inst:
            ema_inst.restore()

            
            
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=20)
sentence_df['prompt_answer'] = 'A'
sentence_loader = get_loader(sentence_df, batch_size=BATCH_SIZE,
                         train_mode=False,
                         num_workers=12)
for fold, (train_idx, val_idx) in enumerate(gkf.split(prompt['prompt_answer'].tolist(), prompt['sentence_answer'].tolist(), prompt['page_id'].tolist())):
    train = prompt.loc[train_idx].reset_index(drop=True)
    val = prompt.loc[val_idx].reset_index(drop=True)
    train_loader=get_loader(train,
                          batch_size=BATCH_SIZE,
                          train_mode=True,
                          num_workers=12)
    val_loader=get_loader(val, batch_size=BATCH_SIZE,
                         train_mode=False,
                         num_workers=12)
    model= RecallModel().to(DEVICE)
    model = torch.nn.parallel.DataParallel(model)
    trainer(train_loader,val_loader,model,
                epochs=10,
                fold = fold,
                accumulation_steps=1,
                early_stop_epochs=5, device=DEVICE)
    break