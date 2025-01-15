from dataset_utils.dataset_final import Dataset
from args import Args
from utils.loadckpt import load_checkpoint
from tqdm import tqdm
from torch.nn import functional as F
import torch
from torch import nn
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from models.loss import get_sync_loss
import os
from dataset_utils.hparams import hparams
from models.model_final import LLCFaceSy
from utils.train_save import save_checkpoint, eval_sample
import wandb
import warnings
from models.discriminator import LLCFaceSy_discriminator

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


global_step = 0
global_epoch = 0


def train(device, model, disc_model, train_data_loader, test_data_loader, optimizer, disc_optimizer, args,
          checkpoint_dir=None, checkpoint_interval=None, nepochs=None):
    global global_step, global_epoch
    os.environ["WANDB_API_KEY"] = '570e501be3352158879e72a4935daf15aa00fcd0'
    wandb.login(key='570e501be3352158879e72a4935daf15aa00fcd0')
    wandb.init(project="CrossRunner", entity='1303868777', name="Per")
    resumed_step = global_step
    L1_loss = nn.L1Loss()


    num_stop = 0
    while global_epoch < nepochs:
        print('Starting Epoch: {}'.format(global_epoch))
        running_l1_loss, running_sync_loss, disc_loss, running_disc_p_loss, running_disc_real_loss, running_disc_fake_loss = 0., 0., 0., 0., 0., 0.
        # prog_bar = tqdm(enumerate(train_data_loader))
        prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for step, (x, y, mel, text_embedding_f, encoded_texts) in prog_bar:
            model.train()
            disc_model.train()
            optimizer.zero_grad()

            # Move data to CUDA device
            x = x.to(device)
            y = y.to(device)
            text_embedding_f = text_embedding_f.to(device).squeeze()
            mel = mel.to(device)
            encoded_texts = encoded_texts.to(device)

            ### Train generator now. Remove ALL grads.
            optimizer.zero_grad()
            disc_optimizer.zero_grad()

            video_decoded = model(x, encoded_texts, text_embedding_f)

            if args.syncnet_wt > 0.:
                sync_loss = get_sync_loss(mel, video_decoded)
            else:
                sync_loss = 0.

            if args.disc_model_p_loss_wt > 0.:
                p_loss = disc_model.forward(video_decoded, p_flag=True)
            else:
                p_loss = 0.

            l1loss = L1_loss(video_decoded, y)

            loss = args.syncnet_wt * sync_loss + args.disc_model_p_loss_wt * p_loss + \
                   (1. - args.syncnet_wt - args.disc_model_p_loss_wt) * l1loss

            loss.backward()
            optimizer.step()

            ### Remove all gradients before Training disc
            disc_optimizer.zero_grad()

            pred = disc_model(y)
            disc_real_loss = F.binary_cross_entropy(pred, torch.ones((len(pred), 1)).to(device))
            disc_real_loss.backward()

            pred = disc_model(video_decoded.detach())
            disc_fake_loss = F.binary_cross_entropy(pred, torch.zeros((len(pred), 1)).to(device))
            disc_fake_loss.backward()

            disc_optimizer.step()



            global_step += 1
            cur_session_steps = global_step - resumed_step

            if global_step == 1 or global_step % checkpoint_interval == 0:
                save_checkpoint(
                    model, optimizer, global_step, checkpoint_dir, global_epoch, hparams, global_step, "ED")
                save_checkpoint(
                    disc_model, disc_optimizer, global_step, checkpoint_dir, global_epoch, hparams, global_step, "disc")
                average_ssim, average_psnr = eval_sample(
                    video_decoded,
                    y,
                    global_step,
                    checkpoint_dir,
                )

            running_disc_real_loss += disc_real_loss.item()
            running_disc_fake_loss += disc_fake_loss.item()
            running_l1_loss += l1loss.item()
            if args.syncnet_wt > 0.:
                running_sync_loss += sync_loss.item()
            else:
                running_sync_loss += 0.

            if args.disc_model_p_loss_wt > 0.:
                running_disc_p_loss += p_loss.item()
            else:
                running_disc_p_loss += 0.

            prog_bar.set_description(
                f'L1 Loss: {running_l1_loss / (step + 1)}, sync Loss: {running_sync_loss / (step + 1)}, D Loss: {running_disc_p_loss / (step + 1)}, D Real: {running_disc_real_loss / (step + 1)}, D Fake: {running_disc_fake_loss / (step + 1)}')

            wandb.log({
                "L1 Loss": running_l1_loss / (step + 1),
                "sync Loss": running_sync_loss / (step + 1),
                "D Loss": running_disc_p_loss / (step + 1),
                "D Real": running_disc_real_loss / (step + 1),
                "D Fake": running_disc_fake_loss / (step + 1)
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
    disc_model = LLCFaceSy_discriminator().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('total discriminator trainable params {}'.format(sum(p.numel() for p in disc_model.parameters() if p.requires_grad)))

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=hparams.initial_learning_rate)

    disc_optimizer = optim.Adam([p for p in disc_model.parameters() if p.requires_grad],
                           lr=hparams.disc_initial_learning_rate, betas=(0.5, 0.999))

    if args.checkpoint_path is not None:
        model = load_checkpoint(args.checkpoint_path, model, optimizer, reset_optimizer=True)
    load_checkpoint("./ckpts/discriminator_pretrained.pth", disc_model, disc_optimizer,
                    reset_optimizer=True, overwrite_global_states=False)
    if args.disc_checkpoint_path is not None:
        load_checkpoint(args.disc_checkpoint_path, disc_model, disc_optimizer,
                        reset_optimizer=False, overwrite_global_states=False)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        optimizer = torch.nn.DataParallel(optimizer, device_ids=args.device_ids)
        disc_model = torch.nn.DataParallel(disc_model, device_ids=args.device_ids)
        disc_optimizer = torch.nn.DataParallel(disc_optimizer, device_ids=args.device_ids)


    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    print("Training...")
    train(device, model, disc_model, train_data_loader, test_data_loader, optimizer, disc_optimizer, args,
          checkpoint_dir=args.checkpoint_dir,
          checkpoint_interval=args.checkpoint_interval,
          nepochs=args.epochs,
          )


if __name__ == '__main__':
    main()
