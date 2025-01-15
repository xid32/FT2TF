from sentence_transformers import SentenceTransformer
from os.path import join, basename
from tqdm import tqdm
from glob import glob
import os
from args_lrs3 import Args
import numpy as np


def get_image_list(data_root, video_data_root, split):
    filelist = []
    linelist = []
    textlist = []
    with open('lrs3_filelists_checked/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            if not os.path.exists(os.path.join(video_data_root, line+".txt")):
                continue
            filelist.append(os.path.join(data_root, line).replace("val", ""))
            linelist.append(line)
            with open(os.path.join(video_data_root, line+".txt"), 'r') as fline:
                text = str(fline.readline()).strip()
                if "Text:" not in text:
                    textlist.append("")
                else:
                    textlist.append(text.replace("Text:", ""))
    return filelist, linelist, textlist


def get_frame_id(frame):
    return int(basename(frame).split('.')[0])


args = Args()
text_encoder = SentenceTransformer("tae898/emoberta-base").cuda()
for split in ['train', 'test']:
    all_videos, linelist, textlist = get_image_list(args.data_root, args.video_data_root, split)
    for idx, vidname in tqdm(enumerate(all_videos), total=len(all_videos)):
        encoded_texts = text_encoder.encode(textlist[idx], convert_to_numpy=True)
        np.save(os.path.join(args.video_data_root, linelist[idx] + "_roberta.npy"), encoded_texts)
