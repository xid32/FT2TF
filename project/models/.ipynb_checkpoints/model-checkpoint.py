import torch
from torch import nn
from torch.nn import functional as F
import math
from models.conv import Conv2dTranspose, Conv2d
from models.qformer import Blip2QFormerModel
import transformers
from typing import Callable, List, Optional


class LLCFaceSy(nn.Module):
    def __init__(self):
        super(LLCFaceSy, self).__init__()
        # video part
        self.video_encoder = nn.ModuleList([
            nn.Sequential(Conv2d(3, 16, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 48,48
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 24,24
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 12,12
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 6,6
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # 1, 1
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)), ])

        self.video_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96,96

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())


        self.load_ckpt()
        self.frozen_model()

        self.qformer_config = transformers.Blip2QFormerConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0,
            position_embedding_type="absolute",
            classifier_dropout=None,
            cross_attention_frequency=2,
            encoder_hidden_size=768,
        )
        self.qformer = Blip2QFormerModel(self.qformer_config)
        self.config = transformers.Blip2Config(num_query_tokens=15)
        self.config.qformer_config = self.qformer_config
        self.query_tokens = nn.Parameter(torch.zeros(1, self.config.num_query_tokens, self.config.qformer_config.hidden_size))

        self.text_f_projection = nn.Linear(2560, self.qformer_config.hidden_size)

        self.video_projection = nn.Linear(self.qformer_config.hidden_size, 512)

    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/wav2lip.pth")
        s = checkpoint["state_dict"]
        video_encoder_dict = {}
        video_decoder_dict = {}
        video_output_dict = {}
        for k, v in s.items():
            k = k.replace('module.', '')
            if 'face_encoder_blocks.' in k:
                if k == 'face_encoder_blocks.0.0.conv_block.0.weight':
                    v = v[:, :3, :, :]
                video_encoder_dict[k.replace('face_encoder_blocks.', '')] = v
            elif 'face_decoder_blocks.' in k:
                video_decoder_dict[k.replace('face_decoder_blocks.', '')] = v
            elif 'output_block.' in k:
                video_output_dict[k.replace('output_block.', '')] = v
        self.video_encoder.load_state_dict(video_encoder_dict)
        self.video_decoder_blocks.load_state_dict(video_decoder_dict)
        self.output_block.load_state_dict(video_output_dict)

    def frozen_model(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = False


    def forward(self, video, text_f, text_l,  return_dict: Optional[bool] = None,):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        B = video.size(0)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)
        feats = []
        x = video
        for f in self.video_encoder:
            x = f(x)
            feats.append(x)
        video_embedding = feats[-1]

        text_f_embeds = self.text_f_projection(text_f)
        text_f_attention_mask = torch.ones(text_f_embeds.size()[:-1], dtype=torch.long, device=text_f_embeds.device)

        query_tokens = self.query_tokens.expand(text_f_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=text_f_embeds,
            encoder_attention_mask=text_f_attention_mask,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        text_f_2_video = self.video_projection(query_output)

        x = torch.cat([text_f_2_video[:, i] for i in range(text_f_2_video.size(1))], dim=0).unsqueeze(-1).unsqueeze(-1)
        query_output = x
        for f in self.video_decoder_blocks:
            x = f(x)
            x = torch.cat((x, feats[-1]), dim=1)
            feats.pop()

        video_decoded_ = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(video_decoded_, B, dim=0)  # [(B, C, H, W)]
            video_decoded = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            video_decoded = video_decoded_

        return video_decoded, video_embedding, query_output


if __name__ == '__main__':
    a = torch.normal(0, 1, [3, 3, 15, 96, 96])
    b = torch.normal(0, 1, [3, 15, 2560])
    c = torch.normal(0, 1, [3, 15, 2560])
    model = LLCFaceSy()
    video_decoded, video_embedding, query_output = model(a, b, c)
    print(video_decoded.shape)
    print(video_embedding.shape)
    print(query_output.shape)


