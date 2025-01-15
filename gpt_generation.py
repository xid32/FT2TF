import tqdm
import numpy as np
import torch
import os
from args import Args
from transformers import pipeline

model = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')
args = Args()
max_length = 15
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
                    continue
                with open(os.path.join(video_data_root, line + ".txt"), 'r') as fline:
                    text = str(fline.readline()).strip()
                    save_file = open(save_path, 'w', encoding='utf-8')
                    if "Text:" not in text:
                        save_file.write("")
                    else:
                        text = text.replace("Text:", "")
                        text_ = text.split(" ")
                        text_f_ = text_[:int(len(text_) / 2) + 1]

                        text_f = ""
                        for word in text_f_:
                            text_f = text_f + word + " "
                        text_f = text_f[:-1]

                        model_output = model(text_f, do_sample=True, min_length=max_length)[0]["generated_text"]
                        save_file.write("Text:" + model_output)

                    save_file.close()