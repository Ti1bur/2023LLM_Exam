import pdb
import random
import torch
import os
import numpy as np
import logging
from transformers import get_cosine_schedule_with_warmup, AdamW, get_linear_schedule_with_warmup, set_seed





def setup_seed(seed):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    random.seed(seed)
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

# def seed_everything(seed=42):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
    
def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.device = 'cpu'
    args.n_gpu = torch.cuda.device_count()



def setup_logging(args):
    # 创建一个logger
    # 实例化一个logger对象
    logger = logging.getLogger('logger')
    # 设置初始显示级别
    logger.setLevel(logging.DEBUG)
    # timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
    # 创建一个文件句柄
    fh = logging.FileHandler(args.logger_file_path)
    fh.setLevel(logging.DEBUG)
    # 创建一个流句柄
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将相应的handler添加在logger对象中
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger



def build_optimizer_and_scheduler(args, model, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']

    param_optimizer = []

    for name, param in model.named_parameters():
        param_optimizer.append((name, param))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.learning_rate},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * num_total_steps, num_training_steps=num_total_steps * 1.05
    )


    return optimizer, scheduler