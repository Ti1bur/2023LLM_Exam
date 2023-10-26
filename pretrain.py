from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")
from data_helper import BertDataSet_MLM
from dataclasses import dataclass
from typing import Optional, Union
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader
import random
from model import Model, EMA, FGM
from torch.utils.data import DataLoader,random_split
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
from sklearn.model_selection import StratifiedKFold, train_test_split
from config import parse_args
import warnings
import torch.nn.functional as F
from utils import *
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from functools import partial
import pdb

warnings.filterwarnings("ignore")

def comput_metrix(logits, labels, off=-100):
    # masked_index = (labels[0] == -100).nonzero().item()
    masked_index = (labels != off).nonzero(as_tuple=True)
    logits = logits[masked_index]
    labels = labels[masked_index]

    y_pred = torch.argmax(logits, dim=-1)
    y_pred = y_pred.view(size=(-1,))
    y_true = labels.view(size=(-1,)).float()
    corr = torch.eq(y_pred, y_true)
    acc = torch.sum(corr.float()) / y_true.shape[0]
    return acc
def train(args):
    data = pd.read_parquet('./data/pretrain_data.parquet')
    data = data[['total']]
    total_len = len(data)
    train, val = data.loc[:(total_len - 10000)].reset_index(drop=True), data.loc[(total_len-10000):].reset_index(drop=True)
    del data
    _ = gc.collect()
    libc.malloc_trim(0)
    print('加载数据完成！')
    dataset = BertDataSet_MLM(args, train)
    val = BertDataSet_MLM(args, val)
    dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    train_dataloader = dataloader_class(dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=dataset.collate)
    val_dataloader = dataloader_class(val, batch_size=args.train_batch_size, shuffle=False, collate_fn=val.collate)
 
    model = AutoModelForMaskedLM.from_pretrained(args.pretrain_model_path)
    model = model.to(args.device)
    model = torch.nn.parallel.DataParallel(model)
    
    num_total_steps = args.num_epoch * len(train_dataloader)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, num_total_steps)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    best_score = 0
    step = 0
    for epoch in range(1, args.num_epoch + 1):
        model.train()
        with tqdm(train_dataloader) as pBar:
            pBar.set_description(desc=f'[Epoch: {epoch} training]')
            for batch in pBar:
                step += 1
                with autocast():
                    for k in batch.keys():
                        batch[k] = batch[k].to(args.device)
                    output = model(**batch)
                    loss = output.loss
                    loss = loss.mean()
                scaler.scale(loss).backward()


                scaler.step(optimizer)
                optimizer.zero_grad()
                scheduler.step()
                scaler.update()
                pBar.set_postfix(loss=loss.item())
                if (step + 1) % 2000 == 0:
                    model.eval()
                    acc_all = []
                    with torch.no_grad():
                        for batch in val_dataloader:
                            with autocast():
                                for k in batch.keys():
                                    batch[k] = batch[k].to(args.device)
                                logits = model(**batch).logits
                            acc = comput_metrix(output.logits, batch['labels']).item()
                            acc_all.append(acc)
                    print(f"STEP:{step} acc:{np.mean(acc_all)}")
                    if np.mean(acc_all) > best_score:
                        best_score = np.mean(acc_all)
                        torch.save(model.state_dict(), f'./save/pretrain/pretrain.bin')
                    torch.save(model.state_dict(), f'./save/pretrain/step{step+1}.bin')
                    model.train()

if __name__ == '__main__':
    args = parse_args()
    args.num_epoch = 1
    args.train_batch_size = 96
    args.valid_batch_size = 96
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    setup_seed(seed=args.seed)
    setup_device(args)
    warnings.filterwarnings('ignore')
    logger = setup_logging(args)
    os.makedirs(args.output_model_path, exist_ok=True)
    logger.info("Training/evaluation parameters: %s", args)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)

    train(args)