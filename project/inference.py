from args_inf import Args
from utils.loadckpt import load_checkpoint
import torch
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from sentence_transformers import SentenceTransformer
import os
from dataset_utils.hparams import hparams
from models.model_final import LLCFaceSy
import warnings
from transformers import AutoTokenizer, GPTNeoModel

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
args = Args()
xWINDOW = args.inf_len


def sort_paths_by_number(paths):
    def get_number_from_path(path):
        start = path.rfind('\\') + 1
        end = path.rfind('.jpg')
        number = path[start:end]
        return int(number)

    sorted_paths = sorted(paths, key=get_number_from_path)
    return sorted_paths


def get_window_x(start_frame, imgs_path):
    start_id = start_frame + xWINDOW
    final_id = start_id + xWINDOW

    window_fnames = []
    for frame_id in range(start_id, final_id):
        frame = os.path.join(imgs_path, '{}.jpg'.format(frame_id))
        window_fnames.append(frame)
    return window_fnames


def get_window_y(start_frame, imgs_path):
    start_id = start_frame
    window_fnames = []
    for frame_id in range(start_id, start_id + xWINDOW):
        frame = os.path.join(imgs_path, '{}.jpg'.format(frame_id))
        window_fnames.append(frame)
    return window_fnames



def read_window(window_fnames, size=hparams.img_size):
    window = []
    for fname in window_fnames:
        img = cv2.imread(fname)
        if img is None:
            return None
        img = cv2.resize(img, (size, size))
        window.append(img)
    return window


def prepare_window(window):
    # 3 x T x H x W
    x = (np.asarray(window) / 255.)
    x = np.transpose(x, (3, 0, 1, 2))

    return x



def inference(device, model, sample_path, save_path, args, use_exit_roberta=True, use_exit_gpt=True):
    imgs_path = os.path.join(sample_path, "imgs")
    imgs_paths = []
    for ii in os.listdir(imgs_path):
        imgs_paths.append(os.path.join(imgs_path, ii))
    imgs_paths = sort_paths_by_number(imgs_paths)
    with open(os.path.join(sample_path, "sample.txt"), 'r', encoding="utf-8") as f:
        text = list(f)[0].strip()
    if use_exit_gpt:
        gpt = np.load(os.path.join(sample_path, "gpt.npy"))
    else:
        max_length = args.inf_len
        gpt_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
        text = text.replace("Text:", "")
        text_ = text.split(" ")
        text_f_ = text_[:int(len(text_) / 2) + 1]
        text_f = ""
        for word in text_f_:
            text_f = text_f + word + " "
        text_f = text_f[:-1]
        input_ids = torch.tensor(gpt_tokenizer.encode(
            text_f,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True)).unsqueeze(0)
        gpt_out = gpt_model(input_ids)
        last_hidden_state = gpt_out.last_hidden_state
        gpt = last_hidden_state.cpu().detach().numpy()
        del gpt_model
        del gpt_tokenizer
    if use_exit_roberta:
        roberta = np.load(os.path.join(sample_path, "roberta.npy"))
    else:
        text_encoder = SentenceTransformer("tae898/emoberta-base").to(device)
        roberta = text_encoder.encode(text, convert_to_numpy=True)
        text_encoder = text_encoder.cpu()
        del text_encoder
    img_name_x = 0
    img_name_y = args.inf_len
    x_window_fnames = get_window_x(img_name_x, imgs_path)
    x_window = read_window(x_window_fnames)
    x_window = prepare_window(x_window)
    y_window_fnames = get_window_y(img_name_y, imgs_path)
    y_window = read_window(y_window_fnames)
    y_window = prepare_window(y_window)

    x = torch.FloatTensor(x_window).unsqueeze(0).to(device)
    gt = torch.FloatTensor(y_window).unsqueeze(0).to(device)
    gpt = torch.FloatTensor(gpt).squeeze().unsqueeze(0).to(device)
    roberta = torch.FloatTensor(roberta).unsqueeze(0).to(device)
    g = model(x, roberta, gpt)

    g_save = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    gt_save = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    folder_g = os.path.join(save_path, "results", "g")
    folder_gt = os.path.join(save_path, "results", "gt")
    if not os.path.exists(folder_g):
        os.makedirs(folder_g)
    if not os.path.exists(folder_gt):
        os.makedirs(folder_gt)
    for batch_idx, c in enumerate(g_save):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_g.jpg'.format(folder_g, t), c[t])
    for batch_idx, c in enumerate(gt_save):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_gt.jpg'.format(folder_gt, t), c[t])
    return


def main():
    args = Args()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    # Model
    model = LLCFaceSy()
    model = model.to(device)
    model = load_checkpoint(args.checkpoint_path, model, None, reset_optimizer=True, overwrite_global_states=True)
    inference(device, model, args.sample_path, args.save_path, args, use_exit_roberta=args.use_exit_roberta, use_exit_gpt=args.use_exit_gpt)


if __name__ == '__main__':
    main()
