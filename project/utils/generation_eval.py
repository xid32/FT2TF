import torch,sys
from utils.LPIPS.util import util
from utils.LPIPS.models import pretrained_networks as pn
from utils.LPIPS.models import dist_model as dm
from tqdm import tqdm
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
import torchvision.transforms as transforms
from torchvision import models
from args import Args
from PIL import Image
from utils.train_save import calculate_average_ssim_psnr
import os
import dlib
import cv2


args = Args()

def eval_lpips(batch_data_g, batch_data_gt, lpips_model):
    batch_data_g = batch_data_g * 2 - 1
    batch_data_gt = batch_data_gt * 2 - 1
    dist = lpips_model.forward(batch_data_g, batch_data_gt)
    return np.mean(dist)


def eval_fid(batch_data_g, batch_data_gt, fid_model):
    # Ensure both sets of images have the same shape [batch_size, 3, 96, 96]
    assert batch_data_g.shape == batch_data_gt.shape, "Input tensor shapes do not match"

    g = (batch_data_g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (batch_data_gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)


    # Prepare the images for Inception V3 input
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def preprocess_images(images):
        processed_images = torch.zeros((len(images), 3, 299, 299))
        for i in range(len(images)):
            processed_images[i] = preprocess(Image.fromarray(images[i]))
        processed_images = processed_images.to("cuda")
        return processed_images

    # Preprocess both real and fake images
    real_images = preprocess_images(g)
    fake_images = preprocess_images(gt)

    # Extract features from the Inception V3 model
    with torch.no_grad():
        real_features = fid_model(real_images).squeeze().cpu().numpy()
        fake_features = fid_model(fake_images).squeeze().cpu().numpy()

    # Calculate mean and covariance for real and fake features
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # Calculate Frechet Distance
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real.dot(sigma_fake))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = np.sum(diff ** 2) / 9.0 + np.trace(sigma_real + sigma_fake - 2 * covmean) / 9.0

    return fd


def load_pretrained_arcface_model():
    # Download and load the pretrained Arcface model
    model = models.resnet50(pretrained=True).cuda()
    return model


def eval_csim(batch_data_g, batch_data_gt, csim_model):
    # Ensure both sets of images have the same shape [batch_size, 3, 96, 96]
    assert batch_data_g.shape == batch_data_gt.shape, "Input tensor shapes do not match"

    g = (batch_data_g.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    gt = (batch_data_gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)


    def preprocess_images(images):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        processed_images = torch.zeros((len(images), 3, 224, 224))
        for i in range(len(images)):
            processed_images[i] = transform(Image.fromarray(images[i]))
        processed_images = processed_images.to("cuda")

        return processed_images

    def calculate_csim(image1, image2, arcface_model):
        # Forward pass the Arcface model to obtain feature vectors
        with torch.no_grad():
            image1_embedding = arcface_model(image1.unsqueeze(0))
            image2_embedding = arcface_model(image2.unsqueeze(0))

        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(image1_embedding, image2_embedding)

        return similarity.item()


    def calculate_batch_csim(images1, images2, arcface_model):
        batch_size = images1.shape[0]
        csim_values = np.zeros(batch_size)

        for i in range(batch_size):
            image1 = images1[i]
            image2 = images2[i]
            csim_values[i] = calculate_csim(image1, image2, arcface_model)

        return csim_values

    g = preprocess_images(g)
    gt = preprocess_images(gt)

    csim_values = calculate_batch_csim(g, gt, csim_model)
    return np.mean(csim_values)


def eval_ssim_psnr(batch_data_g, batch_data_gt):
    g_e = torch.from_numpy(batch_data_g.detach().cpu().numpy().transpose(0, 2, 1, 3, 4))
    gt_e = torch.from_numpy(batch_data_gt.detach().cpu().numpy().transpose(0, 2, 1, 3, 4))
    average_ssim, average_psnr = calculate_average_ssim_psnr(gt_e, g_e, dim=1)
    return average_ssim, average_psnr



def eval_lip_lmd(batch_g_images, batch_gt_images, detector, predictor):
    batch_g_images = (batch_g_images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)
    batch_gt_images = (batch_gt_images.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255.).astype(np.uint8)

    lmds = []
    for g_image, gt_image in zip(batch_g_images, batch_gt_images):
        g_dets = detector(g_image, 1)
        gt_dets = detector(gt_image, 1)

        if len(g_dets) > 0 and len(gt_dets) > 0:
            g_shape = predictor(g_image, g_dets[0])
            gt_shape = predictor(gt_image, gt_dets[0])
            if len(g_shape.parts()) == 68 and len(gt_shape.parts()) == 68:
                # Extract lip landmarks (assume index 48 to 67 in the shape)
                lip_landmarks_g = []
                lip_landmarks_gt = []
                for g_i in g_shape.parts()[48:68]:
                    lip_landmarks_g.append([g_i.x, g_i.y])
                for gt_i in gt_shape.parts()[48:68]:
                    lip_landmarks_gt.append([gt_i.x, gt_i.y])
                lip_landmarks_g = np.array(lip_landmarks_g)
                lip_landmarks_gt = np.array(lip_landmarks_gt)

                # Calculate the lip ROI rectangle
                lip_roi_width = np.max(lip_landmarks_g[:, 0]) - np.min(lip_landmarks_g[:, 0])
                lip_roi_height = np.max(lip_landmarks_g[:, 1]) - np.min(lip_landmarks_g[:, 1])

                # Normalize lip landmarks
                lip_landmarks_g[:, 0] = (lip_landmarks_g[:, 0] - np.min(lip_landmarks_g[:, 0])) / lip_roi_width
                lip_landmarks_g[:, 1] = (lip_landmarks_g[:, 1] - np.min(
                    lip_landmarks_g[:, 1])) / lip_roi_height

                # Calculate the lip ROI rectangle
                lip_roi_width = np.max(lip_landmarks_gt[:, 0]) - np.min(lip_landmarks_gt[:, 0])
                lip_roi_height = np.max(lip_landmarks_gt[:, 1]) - np.min(lip_landmarks_gt[:, 1])

                # Normalize lip landmarks
                lip_landmarks_gt[:, 0] = (lip_landmarks_gt[:, 0] - np.min(
                    lip_landmarks_gt[:, 0])) / lip_roi_width
                lip_landmarks_gt[:, 1] = (lip_landmarks_gt[:, 1] - np.min(
                    lip_landmarks_gt[:, 1])) / lip_roi_height

                liplmd_values = np.linalg.norm(lip_landmarks_g - lip_landmarks_gt, ord=2) / 20

                lmds.append(liplmd_values)

    if len(lmds) == 0:
        return 0
    lmds = np.array(lmds)
    # Compute the mean LipLMD value
    mean_liplmd = np.mean(lmds)
    return mean_liplmd


def eval_test(test_dataloader, model, checkpoint_dir, global_step, line_list, save_test=True):
    device = torch.device("cuda" if args.use_cuda else "cpu")
    model.eval()

    # lpips model
    lpips_model = dm.DistModel()
    lpips_model.initialize(model='net-lin', net='alex', use_gpu=True)


    # fid model
    # Create an Inception V3 model with pretrained weights
    fid_model = inception_v3(pretrained=True, transform_input=False).to("cuda").eval()

    # csim model
    csim_model = load_pretrained_arcface_model()
    csim_model.eval()

    # liplmd model
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./ckpts/shape_predictor_68_face_landmarks.dat")




    ssim = []
    psnr = []
    lpips = []
    fid = []
    csim = []
    liplmd = []
    prog_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    with torch.no_grad():
        for step, (x, y, mel, text_embedding_f, encoded_texts, pidx) in prog_bar:
            model.eval()

            # Move data to CUDA device
            x = x.to(device)
            gt = y.to(device)
            text_embedding_f = text_embedding_f.to(device).squeeze()
            encoded_texts = encoded_texts.to(device)

            pidx = pidx.detach().cpu().numpy()

            g_ = model(x, encoded_texts, text_embedding_f)
            circle_time = args.output_frames // args.input_frames
            if circle_time == 1:
                g = g_
            else:
                c_g = [g_]
                for a_i in range(circle_time-1):
                    c_g.append(model(c_g[-1], encoded_texts, text_embedding_f))
                g = g_
                for a_i in range(1, circle_time):
                    g = torch.cat([g, c_g[a_i]], dim=2)

            ssim_, psnr_ = eval_ssim_psnr(g, gt)
            ssim.append(ssim_)
            psnr.append(psnr_)


            if save_test:
                g_save = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
                gt_save = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
                folder = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step), "test_generation")
                if not os.path.exists(folder):
                    os.makedirs(folder)
                for batch_idx, c in enumerate(g_save):
                    line_name = line_list[int(pidx[int(batch_idx)])].replace("/", "_")
                    for t in range(len(c)):
                        cv2.imwrite('{}/{}_{}_g.jpg'.format(folder, line_name, t), c[t])
                for batch_idx, c in enumerate(gt_save):
                    for t in range(len(c)):
                        cv2.imwrite('{}/{}_{}_gt.jpg'.format(folder, line_name, t), c[t])


            g = torch.cat([g[:, :, i] for i in range(g.size(2))], dim=0)
            gt = torch.cat([gt[:, :, i] for i in range(gt.size(2))], dim=0)

            lpips.append(eval_lpips(g, gt, lpips_model))
            fid.append(eval_fid(g, gt, fid_model))
            csim.append(eval_csim(g, gt, csim_model))
            liplmd.append(eval_lip_lmd(g, gt, detector, predictor))


    ssim = np.array(ssim).mean()
    psnr = np.array(psnr).mean()
    lpips = np.array(lpips).mean()
    fid = np.array(fid).mean()
    csim = np.array(csim).mean()
    liplmd = np.array(liplmd).mean()

    folder = os.path.join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    with open(f'{folder}/test_results.txt', 'w') as f:
        f.write(f'SSIM : {ssim}\n')
        f.write(f'PSNR : {psnr}\n')
        f.write(f'LPIPS : {lpips}\n')
        f.write(f'FID : {fid}\n')
        f.write(f'CSIM : {csim}\n')
        f.write(f'LipLMD : {liplmd}\n')

    return ssim, psnr, lpips, fid, csim, liplmd






