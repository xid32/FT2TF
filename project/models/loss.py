from dataset_utils.dataset import Dataset
from args import Args
from utils.loadckpt import load_checkpoint
from os.path import dirname, join, basename, isfile
from tqdm import tqdm

from torch.nn import functional as F

import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from models.contrastiveLoss import ContrastiveLoss
import numpy as np
from .conv import Conv2d
from glob import glob

import os, random, cv2, argparse
from dataset_utils.hparams import hparams, get_image_list
from models.model import LLCFaceSy
from utils.train_save import save_sample_images, save_checkpoint, eval_model, eval_sample
import wandb
import copy
import warnings

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



args = Args()


class FaceSyncNet(nn.Module):
    def __init__(self):
        super(FaceSyncNet, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),

            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)

    def forward(self, audio_sequences, face_sequences): # audio_sequences := (B, dim, T)
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)


        return audio_embedding, face_embedding



logloss = nn.BCELoss()
def cosine_loss(a, v, y):
    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)

    return loss

device = torch.device("cuda" if args.use_cuda else "cpu")
syncnet = FaceSyncNet().to(device)
load_checkpoint("./ckpts/syncnet.pth", syncnet, None, reset_optimizer=True,
                overwrite_global_states=False)
for p in syncnet.parameters():
    p.requires_grad = False

if args.multi_gpu:
    model = torch.nn.DataParallel(syncnet, device_ids=args.device_ids)

recon_loss = nn.L1Loss()
def get_sync_loss(mel_all, vg):
    vg = vg[:, :, :, vg.size(3)//2:]
    closses = []
    for v_i in range(3):
        g = torch.cat([vg[:, :, i] for i in range(int(5*v_i), int(5*(v_i+1)))], dim=1)
        # B, 3 * T, H//2, W
        mel = mel_all[:, :, :, int(16*v_i):int(16*(v_i+1))]
        a, v = syncnet(mel, g)
        y = torch.ones(g.size(0), 1).float().to(device)
        closses.append(cosine_loss(a, v, y))
    closs = closses[0]
    for i in range(1, 3):
        closs += closses[i]
    return closs




