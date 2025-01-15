import os
import tqdm
from args import Args


path = "./project/filelists/val.txt"
args = Args()
video_data_root = args.video_data_root
data_root = args.data_root
no_count = 0
with open(path) as f_:
    f = list(f_)
    for line in tqdm.tqdm(f, total=len(f)):
        line = line.strip()
        if ' ' in line: line = line.split()[0]
        with open(os.path.join(video_data_root, line + ".txt"), 'r') as fline:
            text = str(fline.readline()).strip()
            if not os.path.exists(os.path.join(video_data_root, line + "_all.npy")) or not os.path.exists(os.path.join(video_data_root, line + "_last.npy")):
                no_count += 1
print(f"{no_count}/{len(f)}")



