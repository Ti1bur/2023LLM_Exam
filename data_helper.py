import pdb

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from scipy.stats import bernoulli
import random

def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
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



class BertDataSet_MLM(Dataset):
    def __init__(self, args, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path)
        self.max_input_len = args.max_seq_len
        self.mask_token = self.tokenizer.mask_token_id
        self.special_token = self.tokenizer.all_special_ids
        self.probability_list = bernoulli.rvs(p=0.2, size=len(data))  # 20%概率进行替换

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data.loc[index, 'total']
        # item = self.data[index]
        return self.encode(item)

    def ngram_mask(self, text_ids):
        text_length = len(text_ids)
        input_ids, output_ids = [], []
        ratio = np.random.choice([0.45, 0.35, 0.25, 0.15], p=[0.2, 0.4, 0.3, 0.1])  # 随机选取mask比例
        replaced_probability_list = bernoulli.rvs(p=ratio, size=text_length)  # 元素为0/1的列表
        replaced_list = bernoulli.rvs(p=0, size=text_length)  # 1表示需要mask，0表示不需要mask, 初始化时全0
        idx = 0
        while idx < text_length:
            if (replaced_probability_list[idx] == 1) and text_ids[idx] not in self.special_token:
                ngram = np.random.choice([1, 2, 3, 4], p=[0.5, 0.3, 0.1, 0.1])  # 若要mask，进行x_gram mask的概率
                L = idx
                R = idx + ngram
                # 1 表示用 【MASK】符号进行mask，2表示随机替换，3表示不进行替换
                mask_partten = np.random.choice([1, 2, 3], p=[0.8, 0.1, 0.1])
                replaced_list[L: R] = mask_partten
                idx = R
                if idx < text_length:
                    replaced_probability_list[R] = 0  # 防止连续mask
            idx += 1

        for r, i in zip(replaced_list, text_ids):
            if r == 1:
                # 使用【mask】替换
                input_ids.append(self.mask_token)
                output_ids.append(i)  # mask预测自己
            elif r == 2:
                # 随机的一个词预测自己，随机词从训练集词表中取，有小概率抽到自己
                input_ids.append(random.choice(text_ids))
                output_ids.append(i)
            elif r == 3:
                # 不进行替换
                input_ids.append(i)
                output_ids.append(i)
            else:
                # 没有被选中要被mask
                input_ids.append(i)
                output_ids.append(-100)

        return input_ids, output_ids


    def encode(self, item):
        text = item
        text_ids = self.tokenizer.encode(text, truncation=True,max_length=600)[1:-1]
  
        input_ids, labels = self.ngram_mask(text_ids)  # ngram mask
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        labels = [-100] + labels + [-100]

        attention_mask = [1] * len(input_ids)

        return input_ids, attention_mask, labels

    @staticmethod
    def collate(batch):
        batch_input_ids, batch_attention_mask = [], []
        batch_labels = []

        for item in batch:
            input_ids, attention_mask, label = item
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(label)

        batch_input_ids = torch.tensor(sequence_padding(batch_input_ids)).long()
        batch_attention_mask = torch.tensor(sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(sequence_padding(batch_labels,padding=-100)).long()

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels
        }