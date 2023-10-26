import pandas as pd
import os
from tqdm.auto import tqdm
import pickle as pkl
idx = -1
for item in range(60):
    print(item)
    with open(f'./tmp/corpus_{item}.pkl','rb') as f:
        token = pkl.load(f)
    chunk = 1000000
    for i in tqdm(token):
        idx += 1
        fold = idx // chunk
        if not os.path.exists(f'./token/{fold}'):
            os.mkdir(f'./token/{fold}')
        with open(f'./token/{fold}/{idx}.pkl','wb') as f:
            pkl.dump(i, f)