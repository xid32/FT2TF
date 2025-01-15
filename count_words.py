import os
from args import Args
import numpy as np
import matplotlib.pyplot as plt



def get_image_list(data_root, video_data_root, split):
    if split == "pretrain":
        video_data_root = video_data_root.replace("main", "pretrain")
        data_root = os.path.join(data_root, "pretrain")
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

args = Args()
all_num = []
for split in ['train', 'test', 'val']:
    all_videos, linelist, textlist = get_image_list(args.data_root, args.video_data_root, split)
    for text in textlist:
        text = text.strip()
        text = text.split(" ")
        all_num.append(len(text))
all_num = np.array(all_num)
np.save("words_num.npy", all_num)
print(all_num.max(), all_num.min(), all_num.mean())

def plot_histogram(data_list):
    plt.hist(data_list, bins=max(data_list) - min(data_list) + 1, align='left', rwidth=0.8)
    plt.title('Histogram of Sentence Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("word_count.png")

plot_histogram(all_num)

