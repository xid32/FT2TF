from os.path import dirname, join, basename, isfile
from torch.utils import data as data_utils

from glob import glob

import os, cv2
from dataset_utils.hparams import hparams, get_image_list
from dataset_utils.transforms import *
import math


xWINDOW = 15
yWINDOW = 15
log1e5 = math.log(1e-5)
class Dataset(data_utils.Dataset):
    def __init__(self, args, split, mode="train"):
        self.all_videos, self.linelist, self.all_texts = get_image_list(args.data_root, args.video_data_root, split)
        self.args = args
        self.mode = mode

        if split == "pretrain":
            self.video_data_root = self.args.video_data_root.replace("main", "pretrain")
            self.data_root = self.args.data_root.replace("main", "pretrain")
        else:
            self.video_data_root = self.args.video_data_root
            self.data_root = self.args.data_root
            
            

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window_x(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + xWINDOW):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def get_window_y(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, start_id + yWINDOW):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def read_window(self, window_fnames, size=hparams.img_size):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (size, size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def prepare_window(self, window):
        # 3 x T x H x W
        x = (np.asarray(window) / 255.)
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            if self.all_texts[idx] == "":
                print("IF1:", self.all_videos[idx])
                continue
            vidname = self.all_videos[idx]
            img_names_ = list(glob(join(vidname, '*.jpg')))
            img_names = sorted(img_names_, key=self.get_frame_id)
            if len(img_names) < 30:
                print("IF2:", self.all_videos[idx])
                continue

            img_name_x = img_names[0]
            x_window_fnames = self.get_window_x(img_name_x)
            img_name_y = img_names[xWINDOW]
            y_window_fnames = self.get_window_y(img_name_y)

            if x_window_fnames is None or y_window_fnames is None:
                # print("IF3:", self.all_videos[idx])
                continue

            x_window = self.read_window(x_window_fnames)
            if x_window is None:
                # print("IF4:", self.all_videos[idx])
                continue

            y_window = self.read_window(y_window_fnames)
            if y_window is None:
                # print("IF5:", self.all_videos[idx])
                continue

            x_window = self.prepare_window(x_window)
            y_window = self.prepare_window(y_window)

            x = torch.FloatTensor(x_window)
            y = torch.FloatTensor(y_window)

            text_embedding = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_all.npy"))
            text_embedding_f = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_first.npy"))
            text_embedding_l = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_last.npy"))

            return x, y, text_embedding, text_embedding_f, text_embedding_l


# x: [b, 3, 5, 224, 224]
# indiv_mels: [b, 5, 1, 80, 16]
# mel: [b, 1, 80, 16]
# y: [b, 3, 5, 224, 224]

