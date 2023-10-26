from __future__ import annotations
import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
from collections.abc import Iterable
import blingfire as bf

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer
from torch.cuda.amp import autocast
import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")

from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from torch.utils.data import DataLoader



DEVICE = 'cuda'
MAX_LENGTH = 512
BATCH_SIZE = 16
BERT_PATH = "/root/bert_path/sentence-transformer-all-mpnet-base-v2"
WIKI_PATH = "./wiki_data"
wiki_files = os.listdir(WIKI_PATH)
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
import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
# class MeanPooling(nn.Module):
#     def __init__(self):
#         super(MeanPooling, self).__init__()

#     def forward(self, last_hidden_state, attention_mask):
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
#         sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
#         sum_mask = input_mask_expanded.sum(1)
#         sum_mask = torch.clamp(sum_mask, min=1e-9)
#         mean_embeddings = sum_embeddings / sum_mask
#         return mean_embeddings

# class RecallModel(nn.Module):
#     def __init__(self):
#         super(RecallModel, self).__init__()
#         self.bert_model = AutoModel.from_pretrained(BERT_PATH)
#         self.mean_pooler = MeanPooling()

#     def mask_mean(self, x, mask=None):
#         if mask != None:
#             mask_x = x * (mask.unsqueeze(-1))
#             x_sum = torch.sum(mask_x, dim=1)
#             re_x = torch.div(x_sum, torch.sum(mask, dim=1).unsqueeze(-1))
#         else:
#             x_sum = torch.sum(x, dim=1)
#             re_x = torch.div(x_sum, x.size()[1])
#         return re_x

#     def forward(self, input_ids):
#         attention_mask = input_ids > 0
#         out = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state
#         x = self.mean_pooler(out, attention_mask)

#         # x = out[:, 0, :]
#         return x

# part = 1
# trn1 = pd.read_csv("./data/all_12_with_context2.csv")
# trn2 = pd.read_csv('./data/crawl_new_dataset.csv')
# trn2 = pd.read_csv('./data/7w8_crawl_dataset.csv')
# trn = pd.read_csv('./tmp/recall_val.csv')
# trn1['type'] = '6w'
# trn2['type'] = '7w8'
# trn3['type'] = '7k'
# trn = pd.concat([trn1,trn2, trn3],axis=0).reset_index(drop=True)
# subs = np.array_split(trn, 3)
# trn = subs[part].reset_index(drop=True)
# trn = pd.read_csv('./data/新8w数据.csv')
# trn['prompt_answer'] = trn.apply(lambda row : ' '.join(str(row[x]) for x in ['prompt', 'A', 'B', 'C', 'D', 'E']),axis=1)
# # # tmp = pd.read_csv('./data/recall_train.csv')
# # # del trn1, trn2
# from functools import partial
# from torch.utils.data import DataLoader
# dataloader_class = partial(DataLoader, pin_memory=True, num_workers=4)
# model= RecallModel()
# from collections import OrderedDict
# def load_param(model_path):
#     state_dict = torch.load(model_path, map_location='cpu')
#     params = OrderedDict()
#     for name, param in state_dict.items():
#         name = '.'.join(name.split('.')[1:])
#         params[name] = param
#     return params
# model.load_state_dict(load_param('./save/recall_base/recall_new_data_hard_example1.bin'))
# model.to(DEVICE)
# model = torch.nn.parallel.DataParallel(model)
# model.eval()

# from tqdm.auto import tqdm
# class LLMRecallDataSet(torch.utils.data.Dataset):
#     def __init__(self, data, col):
#         self.tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, use_fast=True)
#         self.data = data
#         self.col = col
#     def __len__(self):
#         return len(self.data) 
    
#     def __getitem__(self,index):
#         inputs = self.data.loc[index, self.col]
#         inputs = self.tokenizer.encode(inputs, add_special_tokens=False)
#         if len(inputs) > 510:
#             inputs = [101] + inputs[:510] + [102]
#         else:
#             inputs = [101] + inputs + [102]
#         return inputs
    
#     def collate_fn(self, batch):
#         def sequence_padding(inputs, length=None, padding=0):
#             """
#             Numpy函数，将序列padding到同一长度
#             """
#             if length is None:
#                 length = max([len(x) for x in inputs])

#             pad_width = [(0, 0) for _ in np.shape(inputs[0])]
#             outputs = []
#             for x in inputs:
#                 x = x[:length]
#                 pad_width[0] = (0, length - len(x))
#                 x = np.pad(x, pad_width, 'constant', constant_values=padding)
#                 outputs.append(x)

#             return np.array(outputs, dtype='int64')
#         batch_ids = torch.tensor(sequence_padding(batch), dtype=torch.long)
        
#         return batch_ids

        
# class DataLoaderX(torch.utils.data.DataLoader):
#     '''
#         replace DataLoader with PrefetchDataLoader
#     '''
#     def __iter__(self):
#         return BackgroundGenerator(super().__iter__())  

    
# def get_loader(prompt,col,batch_size,train_mode=True,num_workers=4):
#     ds_df = LLMRecallDataSet(prompt,col)
#     # loader = DataLoaderX(ds_df, batch_size=batch_size if train_mode else batch_size//2, shuffle=train_mode, num_workers=num_workers,pin_memory=True,
#     #                                      collate_fn=ds_df.collate_fn, drop_last=train_mode)
#     loader = dataloader_class(ds_df, batch_size=batch_size, shuffle=False,collate_fn=ds_df.collate_fn, num_workers=num_workers)
#     loader.num = len(ds_df)
#     return loader
# from prefetch_generator import BackgroundGenerator
# loader = get_loader(trn, 'prompt_answer',50, False)
# prompt_embeddings = []
# with torch.no_grad():
#     for batch in tqdm(loader):
#         batch = batch.to(DEVICE)
#         with autocast():
#             output = model(batch).cpu().detach().numpy()
#         faiss.normalize_L2(output)
#         prompt_embeddings.append(output)
# prompt_embeddings = np.concatenate(prompt_embeddings, axis=0)
# sentence_index = read_index("./wiki_index/small_new_data_wiki_data_hard_example.bin")

# search_index = []
# subarrays = np.array_split(prompt_embeddings, 100)
# for item in tqdm(subarrays):
#     _, index = sentence_index.search(item, 5)
#     search_index.append(index)
# search_index = np.concatenate(search_index, axis=0)
# df = pd.read_parquet('./small_wiki_data/data.parquet')

# contexts = []
# for i in range(len(trn)):
#     index = list(search_index[i])
#     context = []
#     for j in range(len(index)):
#         context.append(df.loc[index[j], 'text'])
#     contexts.append(context)
# trn['context'] = contexts
# trn.to_csv('./data/新8w_with_Top5_recall.csv',index=False)
# del sentence_index, model
# torch.cuda.empty_cache()
# tokenize_text = []
# import pickle as pkl
# for i in tqdm(range(61)):
#     with open(f'./tmp/token_small/{i}.pkl','rb') as f:
#             tokenize_text.extend(pkl.load(f))


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

def word_process(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    return words
stop_words = set(stopwords.words('english'))


def query_process(q):
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalnum()]
    keywords = [word for word in words if word not in stop_words]
    return " ".join(keywords)

def recall_doc(q, bm25):
    q = word_process(q)
    res_ids = [item[1] for item in bm25.top_k_sentence(q,  k=5)]
    return res_ids

# from fastbm25 import fastbm25
# ids = []
# for i in tqdm(range(len(trn))):
#     idxs = list(search_index[i])
#     token_corpus = [tokenize_text[item] for item in idxs]
#     bm25 = fastbm25(token_corpus)
#     res_ids = recall_doc(trn.loc[i, 'prompt_answer'], bm25)
#     tmp = [idxs[item] for item in res_ids]
#     ids.append(tmp)
# trn['Top5'] = ids
# import pickle as pkl
# with open(f'./data/13w_recall_small_wiki_Top5.pkl', 'wb') as f:
#     pkl.dump(trn, f)
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
# data = pd.read_parquet('./data/10w_bm25_top1000_top20.parquet')
# subs = np.array_split(data, 20)
# part = 19
# trn = subs[part].reset_index(drop=True)
# trn['top1000_to_top20_ids'] = trn['top1000_to_top20_ids'].apply(lambda x : x[:3])
# search_index = trn['top1000_to_top20_ids'].tolist()
# df = pd.read_parquet('./wiki_data/my_index.parquet')
# wikipedia_file_data = []
# for i, idx in tqdm(enumerate(search_index), total=len(search_index)):
#     scr_idx = idx
#     _df = df.loc[scr_idx].copy()
#     _df['prompt_id'] = i
#     wikipedia_file_data.append(_df)
# wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
# wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

# ## Get the full text data
# wiki_text_data = []

# for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
#     _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
#     _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

#     _df_temp = _df[_df['id'].isin(_id)].copy()
#     wiki_text_data.append(_df_temp)
# wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
# _ = gc.collect()


# processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)
# # processed_wiki_text_data = pd.read_pickle('./tmp/processed_wiki_text_data.pickle')

# model = SentenceTransformer('/root/bert_path/sentence-transformers_all-MiniLM-L6-v2', device='cuda')
# model.max_seq_length = MAX_LENGTH
# model = model.half()
# ## Get embeddings of the wiki text data
# wiki_data_embeddings = model.encode(processed_wiki_text_data.text,
#                                     batch_size=1000,
#                                     device=DEVICE,
#                                     show_progress_bar=True,
#                                     convert_to_tensor=True,
#                                     normalize_embeddings=True)#.half()
# wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()

# ## Combine all answers
# trn['answer_all'] = trn.apply(lambda x: " ".join([str(x['A']), str(x['B']), str(x['C']), str(x['D']), str(x['E'])]), axis=1)


# ## Search using the prompt and answers to guide the search
# trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']

# question_embeddings = model.encode(trn.prompt_answer_stem.values, batch_size=200, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
# question_embeddings = question_embeddings.detach().cpu().numpy()
# del model
# torch.cuda.empty_cache()
# ## Parameter to determine how many relevant sentences to include
# NUM_SENTENCES_INCLUDE = 5

# ## List containing just Context
# contexts = []

# for r in tqdm(trn.itertuples(), total=len(trn)):

#     prompt_id = r.Index

#     prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

#     if prompt_indices.shape[0] > 0:
#         prompt_index = faiss.IndexFlatIP(wiki_data_embeddings.shape[1])
#         prompt_index.add(wiki_data_embeddings[prompt_indices])

#         context = []
        
#         ## Get the top matches
#         ss, ii = prompt_index.search(question_embeddings[prompt_id][None,:], NUM_SENTENCES_INCLUDE)
#         for _i in ii[0]:
#             context.append(processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i])
        
#     contexts.append(context)
# trn['context'] = contexts
# trn.to_pickle(f'./tmp/10w_top1000_top3_sentence_part{part}.pkl')

part = 99

data = pd.read_parquet(f'./tmp/680w_token/data_{part}.parquet')
text = data['text'].tolist()
token_text = [word_process(item) for item in tqdm(text)]
import pickle as pkl
with open(f'./tmp/680w_token/token/token_{part}.pkl', 'wb') as f:
    pkl.dump(token_text, f)