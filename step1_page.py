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
from model import EMA
import faiss
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold,KFold,GroupKFold
import torch
# pip install prefetch_generator
from prefetch_generator import BackgroundGenerator
from functools import partial

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

DATA_PATH = "./data/"
BERT_PATH = "/root/bert_path/sentence-transformer-all-mpnet-base-v2"
# BERT_PATH = "deberta-v3-xsmall"
# BERT_PATH = "deberta-v3-base"
PROMPT_LEN = 512
WIKI_LEN = 512
MAX_LEN = 512
BATCH_SIZE = 200
DEVICE = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
use_hard_sample = False
K = 2
import logging

logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
# prompt = pd.read_parquet('./data/recall_data.parquet')
# wiki = pd.read_parquet('./data/recall_wiki.parquet')
# data = pd.read_csv('./data/新8w数据.csv')
# data = pd.read_csv('./data/新微调数据.csv')
data = pd.read_csv('./data/7w8_crawl_dataset.csv')

from copy import deepcopy
class LLMRecallDataSet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        query = self.data.loc[index,'prompt_answer']
        answer = self.data.loc[index, 'title_text']
        if use_hard_sample:
            hards = self.data.loc[index, 'hard']
        query_id = self.tokenizer.encode(query, add_special_tokens=False)
        answer_id = self.tokenizer.encode(answer, add_special_tokens=False)
        if use_hard_sample:
            hard_ids = []
            for hard in hards:
                hard_id = self.tokenizer.encode(hard, add_special_tokens=False)
                if len(hard_id) > 510:
                    hard_id = [101] + hard_id[:510] + [102]
                else:
                    hard_id = [101] + hard_id + [102]
                hard_ids.append(hard_id)
        if len(query_id) > 510:
            query_id = [101] + query_id[:510] + [102]
        else:
            query_id = [101] + query_id + [102]
        if len(answer_id) > 510:
            answer_id = [101] + answer_id[:510] + [102]
        else:
            answer_id = [101] + answer_id + [102]
        if use_hard_sample:
            return query_id, answer_id, hard_ids
        return query_id, answer_id

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
        batch_hards = [[] for i in range(K)]
        for item in batch:
            if use_hard_sample:
                query, answer, hards = item
            else:
                query, answer = item
            batch_query.append(query)
            batch_answer.append(answer)
            if use_hard_sample:
                for i in range(K):
                    batch_hards[i].append(hards[i])
        batch_query = torch.tensor(sequence_padding(batch_query), dtype=torch.long)
        batch_answer = torch.tensor(sequence_padding(batch_answer), dtype=torch.long)
        if use_hard_sample:
            for i in range(K):
                batch_hards[i] = torch.tensor(sequence_padding(batch_hards[i]), dtype=torch.long)
            return batch_query, batch_answer, batch_hards
        return batch_query, batch_answer


class DataLoaderX(torch.utils.data.DataLoader):
    '''
        replace DataLoader with PrefetchDataLoader
    '''

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def get_loader(prompt, batch_size, train_mode=True, num_workers=4):
    ds_df = LLMRecallDataSet(prompt)
    # loader = DataLoaderX(ds_df, batch_size=batch_size if train_mode else batch_size // 2, shuffle=train_mode, num_workers=num_workers,
    #                      pin_memory=True,
    #                      collate_fn=ds_df.collate_fn, drop_last=train_mode)
    dataloader_class = partial(DataLoader, pin_memory=True)
    loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=train_mode, collate_fn=ds_df.collate_fn, num_workers=num_workers)
    return loader


def debug_loader(prompt, batch_size):
    loader = get_loader(prompt, batch_size, train_mode=True, num_workers=0)
    for token_ids, labels in loader:
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
    loader = get_loader(prompt, batch_size=2, train_mode=True, num_workers=0)
    model = RecallModel()
    print('models paramters:', sum(p.numel() for p in model.parameters()))
    for token_ids, labels in loader:
        # print(token_ids)
        # print(labels)
        prob = model(token_ids)
        print(prob)
        break


def SimCSE_loss(topic_pred, content_pred, hards=None, tau=0.05):
    if use_hard_sample:
        similarities = F.cosine_similarity(topic_pred.unsqueeze(1), content_pred.unsqueeze(0), dim=2)  # B,B
        hard_sims = []
        for i in range(K):
            hard_sim = F.cosine_similarity(topic_pred, hards[i])
            hard_sim = hard_sim.unsqueeze(dim=1)
            hard_sims.append(hard_sim)
        hard_sims = torch.concat(hard_sims, axis=1)
        similarities = torch.concat([similarities, hard_sims],axis=1)
        y_true = torch.arange(0, topic_pred.size(0)).to(DEVICE)
        # similarities = similarities - torch.eye(pred.shape[0]) * 1e12
        similarities = similarities / tau
        loss = F.cross_entropy(similarities, y_true)
    else:
        similarities = F.cosine_similarity(topic_pred.unsqueeze(1), content_pred.unsqueeze(0), dim=2)  # B,B
        y_true = torch.arange(0, topic_pred.size(0)).to(DEVICE)
        # similarities = similarities - torch.eye(pred.shape[0]) * 1e12
        similarities = similarities / tau
        loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


from torch.cuda.amp import autocast, GradScaler

def valid(model):
    model.eval()
    index = faiss.IndexFlatIP(768)
    prompt_embed = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            if use_hard_sample:
                topic_inputs, content_inputs, _ = batch
            else:
                topic_inputs, content_inputs = batch
            topic_inputs = topic_inputs.to(DEVICE)
            content_inputs = content_inputs.to(DEVICE)
            with autocast():
                topic_pred = model(topic_inputs).cpu().numpy()
            faiss.normalize_L2(topic_pred)
            prompt_embed.append(topic_pred)
        for batch in tqdm(wiki_loader):
            if use_hard_sample:
                topic_inputs, content_inputs, _ = batch
            else:
                topic_inputs, content_inputs = batch
            topic_inputs = topic_inputs.to(DEVICE)
            content_inputs = content_inputs.to(DEVICE)
            with autocast():
                content_pred = model(content_inputs).cpu().numpy()
            faiss.normalize_L2(content_pred)
            index.add(content_pred)
    prompt_embed = np.concatenate(prompt_embed, axis=0)
    search_score, search_index = index.search(prompt_embed, 1)
    cnt = 0
    for i in range(len(val)):
        label = val.loc[i, 'title_text']
        pred = wiki.loc[search_index[i][0],'title_text']
        if label == pred:
            cnt += 1
        
    return cnt / len(val)
        

def trainer(train_dataloader, val_dataloader, model, epochs, fold,
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


    ema_inst = EMA(model, 0.95)
    ema_inst.register()
    best_score = 0
    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        # train = train.sample(frac=1.0).reset_index(drop=True)
        # print(train.loc[0,'prompt_answer'])
        # sub = train.drop_duplicates(subset='title_text',keep='first').reset_index(drop=True)
        # train_dataloader=get_loader(sub,
        #               batch_size=BATCH_SIZE,
        #               train_mode=True,
        #               num_workers=6)
        bar = tqdm(train_dataloader)
        for i, inputs in enumerate(bar):
            with autocast():
                if use_hard_sample:
                    topic_inputs, content_inputs, hards = inputs
                    topic_inputs = topic_inputs.to(device)
                    content_inputs = content_inputs.to(device)
                    topic_pred = model(topic_inputs)
                    content_pred = model(content_inputs)
                    hard_preds = []
                    for i in range(K):
                        hard = hards[i].to(device)
                        hard_pred = model(hard)
                        hard_preds.append(hard_pred)
                    loss = SimCSE_loss(topic_pred, content_pred, hard_preds)
                else:
                    topic_inputs, content_inputs = inputs
                    topic_inputs = topic_inputs.to(device)
                    content_inputs = content_inputs.to(device)
                    topic_pred = model(topic_inputs)
                    content_pred = model(content_inputs)
                    loss = SimCSE_loss(topic_pred, content_pred)
            scaler.scale(loss).backward()
            losses.append(loss.item())
            scaler.step(optimizer)
            if ema_inst:
                ema_inst.update()
            optimizer.zero_grad()
            scaler.update()
            bar.set_postfix(loss_mean=np.array(losses).mean(), epoch=epoch)
        if ema_inst:
            ema_inst.apply_shadow()
        if epoch % 1 == 0:
            score = valid(model)
            if score > best_score:
                torch.save(model.state_dict(), f'./save/recall_base/recall_new_data_hard_example{K}.bin')
                best_score = score
            print(f"epoch {epoch} score {score} best_score {best_score}")
        
        if ema_inst:
            ema_inst.restore()

def get_hard_data():
    model = RecallModel().to(DEVICE)
    model = torch.nn.parallel.DataParallel(model)
    model.load_state_dict(torch.load('./save/recall_base/recall_new_data_best.bin',map_location='cpu'))
    model.eval()
    train_dataloader=get_loader(train,
                      batch_size=BATCH_SIZE,
                      train_mode=False,
                      num_workers=6)
    index = faiss.IndexFlatIP(768)
    prompt_embed = []
    with torch.no_grad():
        for batch in tqdm(train_dataloader):
            topic_inputs, content_inputs,_ = batch
            topic_inputs = topic_inputs.to(DEVICE)
            content_inputs = content_inputs.to(DEVICE)
            with autocast():
                topic_pred = model(topic_inputs).cpu().numpy()
            faiss.normalize_L2(topic_pred)
            prompt_embed.append(topic_pred)
        for batch in tqdm(wiki_loader):
            topic_inputs, content_inputs,_ = batch
            topic_inputs = topic_inputs.to(DEVICE)
            content_inputs = content_inputs.to(DEVICE)
            with autocast():
                content_pred = model(content_inputs).cpu().numpy()
            faiss.normalize_L2(content_pred)
            index.add(content_pred)
    prompt_embed = np.concatenate(prompt_embed, axis=0)
    search_score, search_index = index.search(prompt_embed, K + 1)
    hard_data = []
    for i in range(len(train)):
        tmp = []
        for j in range(K + 1):
            text = wiki.loc[search_index[i][j], 'title_text']
            if text != train.loc[i,'title_text']:
                tmp.append(text)
        if len(tmp) > K:
            tmp = tmp[:-1]
        hard_data.append(tmp)

    return hard_data
            
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=30)
data['prompt_answer'] = data.apply(lambda row: ' '.join(str(row[field]) for field in ['prompt', 'A', 'B', 'C', 'D', 'E']), axis=1)
data['title_text'] = data.apply(lambda row : ' '.join(str(row[field]) for field in ['page_title', 'section','wiki_text']), axis=1)
# data['title_text'] = data.apply(lambda row : ' '.join(str(row[field]) for field in ['page_title','wiki_text']), axis=1)

# data = data[['prompt_answer','title_text']]
wiki = data.copy()
wiki = wiki.drop_duplicates(subset='title_text').reset_index(drop=True)
if use_hard_sample:
    wiki['hard'] = [['A' for i in range(K)] for j in range(len(wiki))]
wiki_loader=get_loader(wiki, batch_size=BATCH_SIZE,
                         train_mode=False,
                         num_workers=16)


for fold, (train_idx, val_idx) in enumerate(gkf.split(data['prompt_answer'].tolist(), data['title_text'].tolist(), data['title_text'].tolist())):
    train = data.loc[train_idx].reset_index(drop=True)
    val = data.loc[val_idx].reset_index(drop=True)

    if use_hard_sample:
        train['hard'] = [['A' for i in range(K)] for j in range(len(train))]
        train['hard'] = get_hard_data()
        val['hard'] = [['A' for i in range(K)] for j in range(len(val))]
    train_dataloader=get_loader(train,
                            batch_size=BATCH_SIZE,
                            train_mode=True,
                            num_workers=16)
    val_loader=get_loader(val, batch_size=BATCH_SIZE,
                         train_mode=False,
                         num_workers=16)
    model= RecallModel().to(DEVICE)
    model = torch.nn.parallel.DataParallel(model)
    trainer(train_dataloader,val_loader,model,
                epochs=20,
                fold = fold,
                accumulation_steps=1,
                early_stop_epochs=5, device=DEVICE)
    break
