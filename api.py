import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm

from collections.abc import Iterable

from collections import OrderedDict
import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

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
from transformers import AutoTokenizer, AutoConfig, AutoModelForMultipleChoice
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
options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}

def preprocess(example):
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs
    # so we'll copy our question 5 times before tokenizing
    first_sentence = [example['prompt']] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(str(example[option]))
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True, max_length=512)

    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch

def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss
def map3(y_true, y_pred):
    m = (y_true.reshape((-1,1)) == y_pred)
    return np.mean(np.where(m.any(axis=1), m.float().argmax(axis=1)+1, np.inf)**(-1))

def compute_metrics(predictions, labels):
    predictions_sorted = np.argsort(-predictions, axis=1)[:, :3]
    return map3(labels, predictions_sorted)

def get_data(context_path, data_path):
    data = pd.read_csv(context_path)
    trt = pd.read_csv(data_path)
    if 'crawl' in data_path:
        data = data[data['type'] == '4w8'].reset_index(drop=True)
    else:
        data = data[data['type'] == '6w'].reset_index(drop=True)
    # data['concat'] = data.apply(lambda x : str(x['prompt'])+str(x['A'])+str(x['B'])+str(x['C'])+str(x['D'])+str(x['E']),axis=1)
    # data = data.drop_duplicates(subset=['concat'])
    data = data.reset_index(drop=True)
    data.index = list(range(len(data)))
    data['id'] = list(range(len(data)))
    data['prompt'] = data['context'].apply(lambda x : x[:1750]) + ' #### ' + data['prompt']

    return data

def load_param(model):
    state_dict = torch.load('./save/pretrain/step50000.bin',map_location='cpu')
    params = OrderedDict()
    for name, param in state_dict.items():
        if 'module.' in name and 'deberta' in name:
            name = name[7:]
            params[name] = param
    model.load_state_dict(params, strict=False)
    return model
    

def train(args):
    # trn1 = get_data('./data/10w_bm25_20_20_merge.csv', './data/crawl_new_dataset.csv')
    # trn2 = get_data('./data/10w_bm25_20_20_merge.csv', './data/all_12_with_context2.csv')
    # trn = pd.concat([trn1, trn2],axis=0).reset_index(drop=True)
    val = pd.read_parquet('./data/model19_val0.8969.parquet')
    val['id'] = list(range(len(val)))
    val = val.reset_index(drop=True)
    # val['prompt'] = val['context'].apply(lambda x : x[:1750]) + ' #### ' + val['prompt']
    
    # trn = pd.read_csv('./data/RACE_with_context_original.csv')
    # trn['prompt'] = trn['context'].apply(lambda x : x[:1750]) + ' #### ' + trn['prompt']
    # val = trn.sample(2000)
    # trn = trn.drop(val.index).reset_index(drop=True)
    # val = val.reset_index(drop=True)
    # trn['id'] = list(range(len(trn)))
    # val['id'] = list(range(len(val)))
    
    trn = pd.read_parquet('./data/13w_recall_Top1000_Top20_sentence.parquet')
    trn['id'] = list(range(len(trn)))
    # val = trn.sample(2000)
    # trn = trn.drop(val.index).reset_index(drop=True)
    # val = val.reset_index(drop=True)
    trn['context'] = trn['context'].apply(lambda x : ' '.join(x))
    trn['prompt'] = trn['context'].apply(lambda x : x[:1750]) + ' #### ' + trn['prompt']
    # val['prompt'] = val['context'].apply(lambda x : ' '.join(x)[:1750]) + ' #### ' + val['prompt']

    # val.index = list(range(len(val)))
    # val['id'] = list(range(len(val)))
    
    train_dataset = Dataset.from_pandas(trn[['id', 'prompt', 'A','B','C','D','E','answer']].drop(columns=['id'])).map(preprocess,remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E','answer'])
    val_dataset = Dataset.from_pandas(val[['id', 'prompt', 'A','B','C','D','E','answer']].drop(columns=['id'])).map(preprocess,remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E','answer'])
    
    # train_dataset = Dataset.from_pandas(trn[['id', 'prompt', 'A','B','C','D','answer']].drop(columns=['id'])).map(preprocess,remove_columns=['prompt', 'A', 'B', 'C', 'D','answer'])
    # val_dataset = Dataset.from_pandas(val[['id', 'prompt', 'A','B','C','D','answer']].drop(columns=['id'])).map(preprocess,remove_columns=['prompt', 'A', 'B', 'C', 'D','answer'])
    
    if "__index_level_0__" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns(["__index_level_0__"])
    if "__index_level_0__" in val_dataset.column_names:
        val_dataset = val_dataset.remove_columns(["__index_level_0__"])
    
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    dataloader_class = partial(DataLoader, pin_memory=True)
    train_dataloader = dataloader_class(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator, num_workers=12)
    val_dataloader = dataloader_class(val_dataset, batch_size=args.valid_batch_size, shuffle=False, collate_fn=data_collator, num_workers=0)
 
    model = AutoModelForMultipleChoice.from_pretrained(args.pretrain_model_path)
    # model = load_param(model)
    model = model.to(args.device)
    model = torch.nn.parallel.DataParallel(model)
    
    num_total_steps = args.num_epoch * len(train_dataloader)
    num_total_steps = num_total_steps // args.gradient_step
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, num_total_steps)
    scaler = torch.cuda.amp.GradScaler()
    autocast = torch.cuda.amp.autocast
    step = 0
    best_score = 0
    if args.use_EMA:
        ema = EMA(model, args.ema_decay)
    if args.use_FGM:
        fgm = FGM(model, emb_name='word_embeddings',epsilon=args.fgm_eps)
    trick_step = len(train_dataloader)
    ok = False
    gradient_step = args.gradient_step
    for epoch in range(1, 3):
        model.train()
        if epoch >= 2 and args.use_EMA and not ok:
            print('ema start!')
            ema.register()
            ok = True
        with tqdm(train_dataloader) as pBar:
            pBar.set_description(desc=f'[Epoch: {epoch} training]')
            for batch in pBar:
                step += 1
                # if step > 1.6 * trick_step:
                #     break
                model.train()
                with autocast():
                    for k in batch.keys():
                        batch[k] = batch[k].to(args.device)
                    if args.use_rdrop and epoch >= 2:
                        output_a = model(**batch)
                        loss_a, logits_a = output_a.loss, output_a.logits
                        output_b = model(**batch)
                        loss_b, logits_b = output_b.loss, output_b.logits
                        loss = (loss_a + loss_b).mean() / 2 + compute_kl_loss(logits_a, logits_b) * 0.1
                    else:
                        loss = model(**batch).loss
                        loss = loss.mean()
                loss = loss / gradient_step
                scaler.scale(loss).backward()
                

                if args.use_FGM and ok:
                    fgm.attack()
                    with autocast():
                        loss_adv = model(**batch).loss
                        loss_adv = loss_adv.mean()
                    loss_adv = loss_adv / gradient_step
                    scaler.scale(loss_adv).backward()
                    fgm.restore()
                
                if step % gradient_step == 0:
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scheduler.step()
                    scaler.update()
                    if args.use_EMA and ok:
                        ema.update()
                pBar.set_postfix(loss=loss.item() * gradient_step)
                if step % 400 == 0:
                    model.eval()
                    labels = []
                    preds = []
                    if args.use_EMA and ok:
                        ema.apply_shadow()
                    with torch.no_grad():
                        for batch in val_dataloader:
                            for k in batch.keys():
                                batch[k] = batch[k].to(args.device)
                            with autocast():
                                logit = model(**batch).logits
                            preds.append(logit.cpu().detach())
                            labels.append(batch['labels'].cpu().detach())
                    preds = torch.cat(preds)
                    labels = torch.cat(labels)
                    score = compute_metrics(preds, labels)
                    if score > best_score:
                        torch.save(model.state_dict(), f'./save/13w_ema_sentence.bin')
                        best_score = score
                    # torch.save(model.state_dict(), f'./save/15w_true_data/20w_ema_{step}.bin')
                    logger.info(f'[STEP : {step}] score:{score}  best_score:{best_score}')
                    if args.use_EMA and ok:
                        ema.restore()

if __name__ == '__main__':
    args = parse_args()
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