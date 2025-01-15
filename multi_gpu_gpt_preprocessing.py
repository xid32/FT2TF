import tqdm
import numpy as np
import torch
import os
from args import Args
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoModel
import torch.distributed as dist
import torch.multiprocessing as mp


args = Args()
max_length = 15
splits = ['pretrain', 'train', 'test', 'val']

def process_text_data(line, video_data_root, model, tokenizer, max_length, device):
    line = line.strip()
    if ' ' in line: line = line.split()[0]
    with open(os.path.join(video_data_root, line + ".txt"), 'r') as fline:
        text = str(fline.readline()).strip()
        if "Text:" not in text:
            np.save(os.path.join(video_data_root, line + "_all.npy"), np.zeros([1, 1]))
            np.save(os.path.join(video_data_root, line + "_first.npy"), np.zeros([1, 1]))
            np.save(os.path.join(video_data_root, line + "_last.npy"), np.zeros([1, 1]))
        else:
            try:
                text = text.replace("Text:", "")
                text_ = text.split(" ")
                text_f_ = text_[:int(len(text_) / 2) + 1]
                text_l_ = text_[int(len(text_) / 2) + 1:]

                text_f = ""
                for word in text_f_:
                    text_f = text_f + word + " "
                text_f = text_f[:-1]

                text_l = ""
                for word in text_l_:
                    text_l = text_l + word + " "
                text_l = text_l[:-1]

                input_ids = torch.tensor(tokenizer.encode(
                    text,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True)).unsqueeze(0).to(device)

                model_output = model(input_ids)
                last_hidden_state = model_output.last_hidden_state
                sentence_feature = last_hidden_state.cpu().detach().numpy()
                np.save(os.path.join(video_data_root, line + "_all.npy"), sentence_feature)

                input_ids_f = torch.tensor(tokenizer.encode(
                    text_f,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True)).unsqueeze(0).to(device)

                input_ids_l = torch.tensor(tokenizer.encode(
                    text_l,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=True)).unsqueeze(0).to(device)

                model_output_f = model(input_ids_f)
                model_output_l = model(input_ids_l)

                sentence_feature_f = model_output_f.last_hidden_state.cpu().detach().numpy()
                sentence_feature_l = model_output_l.last_hidden_state.cpu().detach().numpy()

                np.save(os.path.join(video_data_root, line + "_first.npy"), sentence_feature_f)
                np.save(os.path.join(video_data_root, line + "_last.npy"), sentence_feature_l)

            except:
                np.save(os.path.join(video_data_root, line + "_all.npy"), np.zeros([1, 1]))
                np.save(os.path.join(video_data_root, line + "_first.npy"), np.zeros([1, 1]))
                np.save(os.path.join(video_data_root, line + "_last.npy"), np.zeros([1, 1]))

def main(local_rank):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
    tokenizer.pad_token = tokenizer.eos_token
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)

    for split in splits:
        video_data_root = args.video_data_root
        if split == "pretrain":
            video_data_root = video_data_root.replace("main", "pretrain")
        with open('filelists/{}.txt'.format(split)) as f_:
            f = list(f_)
            num_lines = len(f)
            batch_size = num_lines // torch.cuda.device_count()
            start_idx = batch_size * local_rank
            end_idx = start_idx + batch_size if local_rank < torch.cuda.device_count() - 1 else num_lines

            for line in tqdm.tqdm(f[start_idx:end_idx], total=end_idx - start_idx):
                process_text_data(line, video_data_root, model, tokenizer, max_length, device)

if __name__ == "__main__":
    # Use multiple GPUs with multiprocessing
    mp.spawn(main, nprocs=torch.cuda.device_count(), join=True)
