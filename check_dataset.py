from os.path import join, basename
from tqdm import tqdm
from glob import glob
import os
from args import Args
import numpy as np


def get_image_list(data_root, video_data_root, split):
    if split == "pretrain":
        video_data_root = video_data_root.replace("main", "pretrain")
        data_root = data_root.replace("main", "pretrain")
    filelist = []
    linelist = []
    textlist = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: line = line.split()[0]
            if not os.path.exists(os.path.join(video_data_root, line+".txt")):
                continue
            filelist.append(os.path.join(data_root, line))
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
all_num = []
# for split in ['pretrain']:
for split in ['test', 'train', 'val', 'pretrain']:
    all_videos, linelist, textlist = get_image_list(args.data_root, args.video_data_root, split)
    new_f = open('{}.txt'.format(split), 'w', encoding='utf-8')
    print(len(all_videos))
    for idx, vidname in tqdm(enumerate(all_videos), total=len(all_videos)):
        img_names_ = list(glob(join(vidname, '*.jpg')))
        img_names = sorted(img_names_, key=get_frame_id)
        if len(img_names) < 30 or len(img_names) >= 36:
            continue
        if textlist[idx] == "":
            continue
        new_f.write(f'{linelist[idx]}\n')
        all_num.append(len(img_names))
    new_f.close()
all_num = np.array(all_num)
np.save("all_num.npy", all_num)
print(all_num.max(), all_num.min(), all_num.mean())
