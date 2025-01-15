import tqdm
import numpy as np
import torch
import os
from args import Args
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoModel

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = model.to("cuda")
args = Args()
model_name = "EleutherAI/gpt-neo-2.7B"
max_length = 15
tokenizer.pad_token = tokenizer.eos_token
# splits = ['pretrain'] 
splits = ['train', 'test', 'val']
with torch.no_grad():
    for split in splits:
        video_data_root = args.video_data_root
        data_root = args.data_root
        if split == "pretrain":
            video_data_root = video_data_root.replace("main", "pretrain")
            data_root = data_root.replace("main", "pretrain")
        textlist = []
        with open('./project/filelists/{}.txt'.format(split)) as f_:
            f = list(f_)
            for line in tqdm.tqdm(f, total=len(f)):
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
                                add_special_tokens=True)).unsqueeze(0).to("cuda")
                            model_output = model(input_ids)
                            last_hidden_state = model_output.last_hidden_state
                            sentence_feature = last_hidden_state.cpu().detach().numpy()
                            np.save(os.path.join(video_data_root, line + "_all.npy"), sentence_feature)

                            input_ids = torch.tensor(tokenizer.encode(
                                text_f,
                                max_length=max_length,
                                truncation=True,
                                padding="max_length",
                                add_special_tokens=True)).unsqueeze(0).to("cuda")
                            model_output = model(input_ids)
                            last_hidden_state = model_output.last_hidden_state
                            sentence_feature = last_hidden_state.cpu().detach().numpy()
                            np.save(os.path.join(video_data_root, line + "_first.npy"), sentence_feature)

                            input_ids = torch.tensor(tokenizer.encode(
                                text_l,
                                max_length=max_length,
                                truncation=True,
                                padding="max_length",
                                add_special_tokens=True)).unsqueeze(0).to("cuda")
                            model_output = model(input_ids)
                            last_hidden_state = model_output.last_hidden_state
                            sentence_feature = last_hidden_state.cpu().detach().numpy()
                            np.save(os.path.join(video_data_root, line + "_last.npy"), sentence_feature)
                        except:
                            np.save(os.path.join(video_data_root, line + "_all.npy"), np.zeros([1, 1]))
                            np.save(os.path.join(video_data_root, line + "_first.npy"), np.zeros([1, 1]))
                            np.save(os.path.join(video_data_root, line + "_last.npy"), np.zeros([1, 1]))
