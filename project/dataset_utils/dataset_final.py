from os.path import dirname, join, basename, isfile

import numpy as np
from torch.utils import data as data_utils

from glob import glob

import os, cv2
from dataset_utils.hparams import hparams, get_image_list
from dataset_utils.transforms import *
import math
from dataset_utils import audio
from sentence_transformers import SentenceTransformer
from args import Args


arg = Args()
xWINDOW = arg.input_frames
yWINDOW = arg.output_frames
log1e5 = math.log(1e-5)
syncnet_mel_step_size = 16
class Dataset(data_utils.Dataset):
    def __init__(self, args, split, mode="train", text_source="GT"):
        self.all_videos, self.linelist, self.all_texts = get_image_list(args.data_root, args.video_data_root, split, args.dataset)
        self.args = args
        self.mode = mode
        self.video_data_root = self.args.video_data_root
        self.data_root = self.args.data_root
        self.text_source = text_source
        # self.text_encoder = SentenceTransformer("tae898/emoberta-base").cuda()

        if split == "pretrain":
            self.video_data_root = self.args.video_data_root.replace("main", "pretrain")
            self.data_root = self.args.data_root.replace("main", "pretrain")

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window_x(self, start_frame):
        start_id = self.get_frame_id(start_frame) + xWINDOW
        final_id = start_id + xWINDOW
        vidname = dirname(start_frame)

        window_fnames = []
        for frame_id in range(start_id, final_id):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames.append(frame)
        return window_fnames

    def get_window_y(self, start_frame):
        start_id = self.get_frame_id(start_frame)
        vidname = dirname(start_frame)

        window_fnames_ = []
        window_fnames = []
        for frame_id in range(start_id, start_id + self.args.input_frames):
            frame = join(vidname, '{}.jpg'.format(frame_id))
            if not isfile(frame):
                return None
            window_fnames_.append(frame)
        if self.mode == "test":
            for a_i in range(int(self.args.output_frames // self.args.input_frames)):
                window_fnames += window_fnames_
        else:
            window_fnames = window_fnames_
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

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)  # 0-indexing ---> 1-indexing
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))

        end_idx = start_idx + syncnet_mel_step_size * 3

        return spec[start_idx: end_idx, :]

    def get_line_list(self):
        return self.linelist

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        while 1:
            idx = random.randint(0, len(self.all_videos) - 1)
            if self.all_texts[idx] == "":
                # print("IF1:", self.all_videos[idx])
                continue
            vidname = self.all_videos[idx]
            img_names_ = list(glob(join(vidname, '*.jpg')))
            img_names = sorted(img_names_, key=self.get_frame_id)
            if len(img_names) < 30 or len(img_names) >= 36:
                # print("IF2:", self.all_videos[idx])
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

            # text_embedding = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_all.npy"))
            text_embedding_f = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_first.npy"))[:, :self.args.input_frames, :]
            # text_embedding_l = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_last.npy"))


            try:
                wavpath = join(vidname, "audio.wav")
                wav = audio.load_wav(wavpath, hparams.sample_rate)

                orig_mel = audio.melspectrogram(wav).T
            except Exception as e:
                continue

            mel = self.crop_audio_window(orig_mel.copy(), xWINDOW)
            mel = torch.FloatTensor(mel.T).unsqueeze(0)

            if self.text_source == "GPT":
                encoded_texts = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_roberta_gpt_txt.npy"))
            else:
                encoded_texts = np.load(os.path.join(self.video_data_root, self.linelist[idx] + "_roberta.npy"))


            pidx = torch.from_numpy(np.array([idx]))

            return x, y, mel, text_embedding_f, encoded_texts, pidx


if __name__ == '__main__':
    args = Args()
    print("Init Dataset...")
    train_dataset = Dataset(args, 'train')
    print("Init DataLoader...")
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    print(len(train_dataset), len(train_data_loader))
    for step, (x, y, mel, text_embedding_f, encoded_texts) in enumerate(train_data_loader):
        print(x.shape, y.shape, mel.shape, text_embedding_f.shape, encoded_texts.shape)
        break


# x: [64, 3, 15, 96, 96]
# y: [64, 3, 15, 96, 96]
# mel: [64, 48, 80]
# text_embedding_f: [64, 1, 15, 2560]
# encoded_texts: [64, 768]
