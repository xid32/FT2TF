import os
import tqdm
import numpy as np
import torch
import os
from args import Args
from transformers import pipeline


args = Args()
splits = ['train', 'test']
with torch.no_grad():
    for split in splits:
        video_data_root = args.video_data_root
        data_root = args.data_root
        if split == "pretrain":
            video_data_root = video_data_root.replace("main", "pretrain")
            data_root = data_root.replace("main", "pretrain")
        textlist = []
        with open('./project/filelists/{}.txt'.format(split)) as f_:
            f = list(f_)
            for line in tqdm.tqdm(f, total=len(f)):
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                save_path = os.path.join(video_data_root, line + "_gpt.txt")
                if os.path.exists(save_path):
                    os.remove(save_path)


