import numpy as np
import os
from args import Args


def get_image_list(data_root, video_data_root, split, dataset="lrs2"):
    filelist = []
    textlist = []
    linelist = []

    if split == "pretrain":
        video_data_root = video_data_root.replace("main", "pretrain")
        data_root = data_root.replace("main", "pretrain")

    if dataset == "lrs2":
        with open('/jumbo/jinlab/XD/face_expression/project/filelists/{}.txt'.format(split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(os.path.join(data_root, line))
                linelist.append(line)
                with open(os.path.join(video_data_root, line+".txt"), 'r') as fline:
                    text = str(fline.readline()).strip()
                    if "Text:" not in text:
                        textlist.append("")
                    else:
                        textlist.append(text.replace("Text:", ""))
    else:
        with open('/jumbo/jinlab/XD/face_expression/project/lrs3_filelists_checked/{}.txt'.format(split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(os.path.join(data_root, line).replace("val", ""))
                linelist.append(line)
                with open(os.path.join(video_data_root, line+".txt"), 'r') as fline:
                    text = str(fline.readline()).strip()
                    if "Text:" not in text:
                        textlist.append("")
                    else:
                        textlist.append(text.replace("Text:", ""))
    return filelist, linelist, textlist


def cal_cosine_distance(array1, array2):
    dot_product = np.dot(array1, array2)
    norm_array1 = np.linalg.norm(array1)
    norm_array2 = np.linalg.norm(array2)
    cosine_distance = 1.0 - (dot_product / (norm_array1 * norm_array2))
    return cosine_distance




def check_distance(linelist, arg):
    all_cd = []
    video_path = arg.video_data_root
    for file_i in linelist:
        file_i = os.path.join(video_path, file_i)
        roberta_path = file_i + "_roberta.npy"
        gpt_roberta_path = file_i + "_roberta_gpt_txt.npy"
        if not os.path.exists(roberta_path) or not os.path.exists(gpt_roberta_path):
            continue
        roberta = np.load(roberta_path)
        gpt_roberta_path = np.load(gpt_roberta_path)
        cd = cal_cosine_distance(roberta, gpt_roberta_path)
        all_cd.append(cd)
    all_num = len(all_cd)
    all_cd = np.array(all_cd)
    return np.sum(all_cd) / all_num



if __name__ == '__main__':
    arg = Args()
    filelist, linelist, textlist = get_image_list(arg.data_root, arg.video_data_root, "test", arg.dataset)
    print(check_distance(linelist, arg))

