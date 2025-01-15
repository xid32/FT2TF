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


global_step = 0
global_epoch = 0


def train(device, model, train_data_loader, test_data_loader, tdataset, optimizer, args,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None, mode=1):
    global global_step, global_epoch
    wandb.init(project="CrossRunner", entity='xingjiandiao', name="Per")
    resumed_step = global_step
    L1s_loss = nn.SmoothL1Loss()
    CL_loss = ContrastiveLoss()

    num_stop = 0
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_l1s_loss, running_cl_loss = 0., 0.
        # prog_bar = tqdm(enumerate(train_data_loader))
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for step, (x, y, text_embedding, text_embedding_f, text_embedding_l) in prog_bar:
            model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            y = y.to(device)
            text_embedding = text_embedding.to(device).squeeze()
            text_embedding_f = text_embedding_f.to(device).squeeze()
            text_embedding_l = text_embedding_l.to(device).squeeze()


            if mode == 1:
                video_decoded, video_embedding, query_output = model(x, text_embedding_f, text_embedding_l)
            else:
                video_decoded, video_embedding, query_output = model(x, text_embedding, text_embedding_l)

            if mode == 1:
                clloss = CL_loss(video_embedding, query_output, text_embedding_f)
                loss = clloss
                running_cl_loss += clloss.item()
            elif mode == 2:
                l1loss = L1s_loss(video_decoded, y)
                loss = l1loss
                running_l1s_loss += l1loss
            loss.backward()
            if args.multi_gpu:
                optimizer.module.step()
            else:
                optimizer.step()

            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch, hparams, global_step)
                average_ssim, average_psnr = eval_sample(
                    video_decoded,
                    y,
                    global_step,
                    checkpoint_dir,
                )

            prog_bar.set_description(
                f'CL Loss: {running_cl_loss / (step + 1)}, L1-S V: {running_l1s_loss / (step + 1)}')

            wandb.log({
                "CL Loss": running_cl_loss / (step + 1),
                "L1-S V": running_l1s_loss / (step + 1),
            })

        global_epoch += 1
    wandb.finish()


def main():
    print("Start Training!")
    args = Args()
    print("Init Dataset...")
    train_dataset = Dataset(args, 'train')
    test_dataset = Dataset(args, 'test')
    print("Init DataLoader...")
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers, pin_memory=args.pin_memory)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # Model
    model = LLCFaceSy()
    model = model.to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=hparams.initial_learning_rate)

    if args.checkpoint_path is not None:
        model = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=True)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=args.device_ids)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    print("Training...")
    train(device, model, train_data_loader, test_data_loader, train_dataset, optimizer, args,
          checkpoint_dir=args.checkpoint_dir,
          checkpoint_interval=args.checkpoint_interval,
          nepochs=args.epochs,
          mode=2
          )


if __name__ == '__main__':
    main()
