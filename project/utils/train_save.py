from os.path import dirname, join, basename, isfile
import os, cv2, numpy as np
import torch, torch.nn as nn
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import ToPILImage
from einops import rearrange
import math
from torchmetrics.functional import structural_similarity_index_measure as SSIM
import librosa
from scipy.io import wavfile
from dataset_utils.hparams import hparams
from pesq import pesq
from pystoi.stoi import stoi
from glob import glob
import soundfile as sf


def save_sample_images(g, gt, au, aut, global_step, checkpoint_dir, save_v=True, save_a=False):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    if save_v:
        g = ((g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) + 1) / 2 * 255.).astype(np.uint8)
        gt = ((gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) + 1) / 2 * 255.).astype(np.uint8)
        collage = np.concatenate((g, gt), axis=1)
        for batch_idx, c in enumerate(collage):
            for t in range(len(c)):
                cv2.imwrite('{}/v_{}_{}.jpg'.format(folder, batch_idx, t), c[t])
    if save_a:
        au = au.detach().cpu().numpy()
        aut = aut.detach().cpu().numpy()
        au_co = np.concatenate((au, aut), axis=1)
        for batch_idx, c in enumerate(au_co):
            for t in range(len(c)):
                mel_spectrogram = np.log(c[t] + 1e-10)
                plt.figure()
                plt.imshow(np.squeeze(mel_spectrogram), cmap='inferno', origin='lower', aspect='auto')
                plt.colorbar(format='%+2.0f dB')
                plt.ylabel('mel-spectrogram channels')
                plt.savefig('{}/a_{}_{}.png'.format(folder, batch_idx, t))


def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, hparams, global_step, mode=1):

    checkpoint_path = join(
        checkpoint_dir, "mode_{}_checkpoint_step{:09d}.pth".format(mode, global_step))
    optimizer_state = optimizer.state_dict() if hparams.save_optimizer_state else None
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def eval_model(test_data_loader, global_step, device, model, checkpoint_dir, args):
    eval_steps = global_step
    print('Evaluating for {} steps'.format(eval_steps))
    rec_losses, perc_losses, sh_losses = [], [], []
    step = 0
    celoss = nn.CrossEntropyLoss()
    while 1:
        for x, indiv_mels, mel, gt in test_data_loader:
            step += 1
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = gt.to(device)
            indiv_mels = indiv_mels.to(device)
            mel = mel.to(device)

            v_sh, a_sh, v_out, a_out = model(indiv_mels, x)

            rec_loss = compute_reconstruction_loss(args, v_out, gt)
            perc_loss = compute_perceptual_loss(args, a_out, indiv_mels)
            sh_loss = celoss(v_sh, a_sh)

            rec_losses.append(rec_loss.item())
            perc_losses.append(perc_loss.item())
            sh_losses.append(sh_loss)

            if step > eval_steps:
                averaged_rec_loss = sum(rec_losses) / len(perc_losses)
                averaged_perc_loss = sum(perc_losses) / len(perc_losses)
                averaged_sh_loss = sum(sh_losses) / len(sh_losses)

                print('Reconstruction: {}, Perceptual loss: {}, Cross Entropy'.format(averaged_rec_loss, averaged_perc_loss, averaged_sh_loss))

                return averaged_rec_loss + averaged_perc_loss + averaged_sh_loss

def calculate_average_ssim_psnr(imgs_ori, imgs_gen, dim=2):
    num_images = imgs_ori.shape[dim]
    ssim_sum = 0
    psnr_sum = 0

    for i in range(num_images):
        if dim == 2:
            img_ori = imgs_ori[:, :, i, :, :]
            img_gen = imgs_gen[:, :, i, :, :]
        else:
            img_ori = imgs_ori[:, i, :, :, :]
            img_gen = imgs_gen[:, i, :, :, :]
        # Calculate SSIM value using torchvision's SSIM metric
        ssim_val = SSIM(img_gen, img_ori, data_range=1.0)
        ssim_sum += ssim_val.item()

        # Calculate PSNR value
        mse = torch.mean((img_ori - img_gen) ** 2)
        psnr_val = 10 * math.log10(1.0 / mse.item())
        psnr_sum += psnr_val

    # Calculate averages
    average_ssim = ssim_sum / num_images
    average_psnr = psnr_sum / num_images

    return average_ssim, average_psnr



def eval_sample(g, gt, global_step, checkpoint_dir):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    g_e = torch.from_numpy(g.detach().cpu().numpy().transpose(0, 2, 1, 3, 4))
    gt_e = torch.from_numpy(gt.detach().cpu().numpy().transpose(0, 2, 1, 3, 4))
    average_ssim, average_psnr = calculate_average_ssim_psnr(gt_e, g_e, dim=1)
    with open(f'{folder}/sample_results.txt', 'w') as f:
        f.write(f'SSIM : {average_ssim}\n')
        f.write(f'PSNR : {average_psnr}\n')

    g = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    for batch_idx, c in enumerate(g):
        for t in range(len(c)):
            cv2.imwrite('{}/v_{}_{}_g.jpg'.format(folder, batch_idx, t), c[t])
    for batch_idx, c in enumerate(gt):
        for t in range(len(c)):
            cv2.imwrite('{}/v_{}_{}_gt.jpg'.format(folder, batch_idx, t), c[t])

    return average_ssim, average_psnr


def vis_sample_images(device, img, bool_masked_pos, outputs, args, checkpoint_dir, global_step):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    # save original video
    mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
    std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
    ori_img = img * std + mean  # in [0, 1]
    imgs = []
    for b_i in range(ori_img.size(0)):
        for t_i in range(ori_img.size(2)):
            imgs.append(ToPILImage()(ori_img[b_i, :, t_i, :, :].cpu()))
    for id, im in enumerate(imgs):
        im.save(f"{folder}/ori_img{id}.jpg")

    img_squeeze = rearrange(ori_img, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', p0=2, p1=args.patch_size,
                            p2=args.patch_size)
    img_norm = (img_squeeze - img_squeeze.mean(dim=-2, keepdim=True)) / (
                img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    for b_i in range(img_patch.size(0)):
        img_patch[b_i:b_i+1][bool_masked_pos[b_i:b_i+1]] = outputs[b_i:b_i+1]

    # make mask
    mask = torch.ones_like(img_patch)
    mask[bool_masked_pos] = 0
    mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
    mask = rearrange(mask, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2) ', p0=2, p1=args.patch_size,
                     p2=args.patch_size, h=14, w=14)

    # save reconstruction video
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    # Notice: To visualize the reconstruction video, we add the predict and the original mean and var of each patch.
    rec_img = rec_img * (img_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6) + img_squeeze.mean(dim=-2,
                                                                                                                keepdim=True)
    rec_img = rearrange(rec_img, 'b (t h w) (p0 p1 p2) c -> b c (t p0) (h p1) (w p2)', p0=2, p1=args.patch_size,
                        p2=args.patch_size, h=14, w=14)
    imgs = []
    for b_i in range(ori_img.size(0)):
        for t_i in range(ori_img.size(2)):
            imgs.append(ToPILImage()(rec_img[b_i, :, t_i, :, :].cpu().clamp(0, 0.996)))
    for id, im in enumerate(imgs):
        im.save(f"{folder}/rec_img{id}.jpg")

    # save masked video
    img_mask = rec_img * mask
    imgs = []
    for b_i in range(ori_img.size(0)):
        for t_i in range(ori_img.size(2)):
            imgs.append(ToPILImage()(img_mask[b_i, :, t_i, :, :].cpu()))
    for id, im in enumerate(imgs):
        im.save(f"{folder}/mask_img{id}.jpg")
    rec_img = rec_img.clamp(0, 0.996)
    average_ssim, average_psnr = calculate_average_ssim_psnr(ori_img, rec_img)
    return average_ssim, average_psnr


def griffin_lim(magnitude, n_fft, hop_length, num_iterations=100):
    phase = np.exp(2j * np.pi * np.random.rand(*magnitude.shape))
    complex_spec = magnitude * phase
    signal = librosa.istft(complex_spec, hop_length=hop_length)

    for i in range(num_iterations):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))
        complex_spec = magnitude * phase
        signal = librosa.istft(complex_spec, hop_length=hop_length)

    return signal


def audio_vis_eval(au, aut, checkpoint_dir, global_step):
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    au = au.detach().cpu().numpy()
    aut = aut.detach().cpu().numpy()
    wav_list_gt = []
    wav_list_rec = []
    for batch_idx, c in enumerate(au):
        mel_spectrogram = np.log(c + 1e-10)
        plt.figure()
        plt.imshow(np.squeeze(mel_spectrogram), cmap='inferno', origin='lower', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.ylabel('mel-spectrogram channels')
        plt.savefig('{}/audio_gt_{}.png'.format(folder, batch_idx))

        magnitude = np.exp(c)
        reconstructed_audio = griffin_lim(magnitude, 130, 200, num_iterations=100)
        output_wavpath = '{}/audio_gt_{}.wav'.format(folder, batch_idx)
        wavfile.write(output_wavpath, hparams.sample_rate, reconstructed_audio.astype(np.int16))
        wav_list_gt.append(output_wavpath)

        mel_spectrogram = np.log(aut[batch_idx]+ 1e-10)
        plt.figure()
        plt.imshow(np.squeeze(mel_spectrogram), cmap='inferno', origin='lower', aspect='auto')
        plt.colorbar(format='%+2.0f dB')
        plt.ylabel('mel-spectrogram channels')
        plt.savefig('{}/audio_gt_{}.png'.format(folder, batch_idx))

        magnitude = np.exp(aut[batch_idx])
        reconstructed_audio = griffin_lim(magnitude, 130, 200, num_iterations=100)
        output_wavpath = '{}/audio_rec_{}.wav'.format(folder, batch_idx)
        wavfile.write(output_wavpath, hparams.sample_rate, reconstructed_audio.astype(np.int16))
        wav_list_rec.append(output_wavpath)

    total_pesq = 0
    total_stoi = 0
    total_estoi = 0

    for f_i, filename in enumerate(wav_list_gt):
        _, deg = wavfile.read(wav_list_rec[f_i])
        _, ref = wavfile.read(filename)
        if len(ref.shape) > 1: ref = np.mean(ref, axis=1)  # raise ValueError('Audio should be a mono band')
        if len(ref) > len(deg):
            x = ref[0: deg.shape[0]]
        elif len(deg) > len(ref):
            deg = deg[: ref.shape[0]]
            x = ref
        else:
            x = ref
        x = np.concatenate((x, x))
        deg = np.concatenate((deg, deg))
        total_pesq += pesq(hparams.sample_rate, x, deg, 'nb')
        total_stoi += stoi(x, deg, hparams.sample_rate, extended=False)
        total_estoi += stoi(x, deg, hparams.sample_rate, extended=True)

    return total_pesq / len(wav_list_gt), total_stoi / len(wav_list_gt), total_estoi / len(wav_list_gt)



