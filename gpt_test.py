import tqdm
import numpy as np
import torch
import os
from args import Args
from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoModel

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-2.7B")


args = Args()
model_name = "EleutherAI/gpt-neo-2.7B"
tokenizer.pad_token = tokenizer.eos_token

splits = ['pretrain', 'train', 'test', 'val']
with torch.no_grad():
    with open(os.path.join(args.video_data_root, "5535415699068794046/00001"+".txt"), 'r') as fline:
        text = str(fline.readline()).strip()
        text = text.replace("Text:", "")
        text_ = text.split(" ")
        text_f_ = text_[:int(len(text_)/2)+1]
        text_l_ = text_[int(len(text_)/2)+1:]

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
            max_length=100,
            truncation=True,
            padding="max_length",
            add_special_tokens=True)).unsqueeze(0)
        model_output = model(input_ids)
        last_hidden_state = model_output.last_hidden_state
        sentence_feature = last_hidden_state.cpu().detach().numpy()
        np.save("gpt_test_all.npy", sentence_feature)

        input_ids = torch.tensor(tokenizer.encode(text_f, add_special_tokens=True)).unsqueeze(0)
        model_output = model(input_ids)
        last_hidden_state = model_output.last_hidden_state
        sentence_feature = last_hidden_state.cpu().detach().numpy()
        np.save("gpt_test_f.npy", sentence_feature)

        input_ids = torch.tensor(tokenizer.encode(text_l, add_special_tokens=True)).unsqueeze(0)
        model_output = model(input_ids)
        last_hidden_state = model_output.last_hidden_state
        sentence_feature = last_hidden_state.cpu().detach().numpy()
        np.save("gpt_test_l.npy", sentence_feature)