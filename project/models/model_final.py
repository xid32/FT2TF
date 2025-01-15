import torch
from torch import nn
from torch.nn import functional as F
from models.conv import Conv2dTranspose, Conv2d
from models.vit import ViT_encoder


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

        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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

    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)
        feats = []
        x = video.clone()
        for f in self.video_encoder:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        video_embedding = feats[-1].clone()

        roberta_encoding = self.roberta_mapping(roberta_embedding)
        roberta_encoding = F.normalize(roberta_encoding)
        roberta_encoding = roberta_encoding.repeat(S, 1).unsqueeze(1)
        video_roberta_encoding = F.normalize(video_embedding.squeeze())
        video_roberta_encoding = video_roberta_encoding.unsqueeze(1)

        cross_roberta = self.video_from_roberta(roberta_encoding, video_roberta_encoding, video_roberta_encoding)[
            0].squeeze().unsqueeze(-1).unsqueeze(-1)
        feats[-1] += cross_roberta

        if input_dim_size > 4:
            text_f = torch.cat([text_f[:, i, :] for i in range(text_f.size(1))], dim=0)
        text_embeds = self.text_f_projection(text_f)
        text_embeds = F.normalize(text_embeds)
        text_embeds = text_embeds.unsqueeze(1)
        cross_gpt = self.video_from_gpt(text_embeds, video_roberta_encoding, video_roberta_encoding)[0].squeeze()

        x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
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

        return video_decoded


class LLCFaceSy_ablation_global(nn.Module):
    def __init__(self):
        super(LLCFaceSy_ablation_global, self).__init__()
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

        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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

    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)
        feats = []
        x = video.clone()
        for f in self.video_encoder:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        video_embedding = feats[-1].clone().squeeze().unsqueeze(1)

        if input_dim_size > 4:
            text_f = torch.cat([text_f[:, i, :] for i in range(text_f.size(1))], dim=0)
        text_embeds = self.text_f_projection(text_f)
        text_embeds = F.normalize(text_embeds)
        text_embeds = text_embeds.unsqueeze(1)
        cross_gpt = self.video_from_gpt(text_embeds, video_embedding, video_embedding)[0].squeeze()

        x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
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

        return video_decoded


class LLCFaceSy_ablation_local(nn.Module):
    def __init__(self):
        super(LLCFaceSy_ablation_local, self).__init__()
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

        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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

    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)
        feats = []
        x = video.clone()
        for f in self.video_encoder:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        video_embedding = feats[-1].clone()

        roberta_encoding = self.roberta_mapping(roberta_embedding)
        roberta_encoding = F.normalize(roberta_encoding)
        roberta_encoding = roberta_encoding.repeat(S, 1).unsqueeze(1)
        video_roberta_encoding = F.normalize(video_embedding.squeeze())
        video_roberta_encoding = video_roberta_encoding.unsqueeze(1)

        cross_roberta = self.video_from_roberta(roberta_encoding, video_roberta_encoding, video_roberta_encoding)[
            0].squeeze().unsqueeze(-1).unsqueeze(-1)
        feats[-1] += cross_roberta

        x = cross_roberta
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

        return video_decoded


class LLCFaceSy_ablation_att(nn.Module):
    def __init__(self):
        super(LLCFaceSy_ablation_att, self).__init__()
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

        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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

    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)
        feats = []
        x = video.clone()
        for f in self.video_encoder:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        video_embedding = feats[-1].clone()

        x = video_embedding
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

        return video_decoded


class LLCFaceSy_vit_small(nn.Module):
    def __init__(self):
        super(LLCFaceSy_vit_small, self).__init__()
        # video part
        self.video_encoder = ViT_encoder(
            image_size=96,
            patch_size=16,
            dim=512,
            depth=4,
            heads=4,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.skip_part = nn.ModuleList([
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
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

        ])


        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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


    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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
        self.skip_part.load_state_dict(video_encoder_dict)
        self.video_decoder_blocks.load_state_dict(video_decoder_dict)
        self.output_block.load_state_dict(video_output_dict)

    def frozen_model(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = False

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)


        feats = []
        x = video.clone()
        for f in self.skip_part:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        feats.pop()
        x = video.clone()
        x = self.video_encoder(x.clone().cuda()).unsqueeze(-1).unsqueeze(-1)
        feats.append(x.clone())
        video_embedding = feats[-1].clone()

        roberta_encoding = self.roberta_mapping(roberta_embedding)
        roberta_encoding = F.normalize(roberta_encoding)
        roberta_encoding = roberta_encoding.repeat(S, 1).unsqueeze(1)
        video_roberta_encoding = F.normalize(video_embedding.squeeze())
        video_roberta_encoding = video_roberta_encoding.unsqueeze(1)

        cross_roberta = self.video_from_roberta(roberta_encoding, video_roberta_encoding, video_roberta_encoding)[
            0].squeeze().unsqueeze(-1).unsqueeze(-1)
        feats[-1] += cross_roberta

        if input_dim_size > 4:
            text_f = torch.cat([text_f[:, i, :] for i in range(text_f.size(1))], dim=0)
        text_embeds = self.text_f_projection(text_f)
        text_embeds = F.normalize(text_embeds)
        text_embeds = text_embeds.unsqueeze(1)
        cross_gpt = self.video_from_gpt(text_embeds, video_roberta_encoding, video_roberta_encoding)[0].squeeze()

        # x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
        # for f in self.video_decoder_blocks:
        #     x = f(x)
        #     if len(feats) != 0:
        #         x = torch.cat((x, feats[-1]), dim=1)
        #         feats.pop()
        # video_decoded_ = self.output_block(x)

        x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
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

        return video_decoded


class LLCFaceSy_vit_large(nn.Module):
    def __init__(self):
        super(LLCFaceSy_vit_large, self).__init__()
        # video part
        self.video_encoder = ViT_encoder(
            image_size=96,
            patch_size=16,
            dim=512,
            depth=64,
            heads=4,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.skip_part = nn.ModuleList([
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
                          Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),

        ])


        self.roberta_mapping = nn.Linear(768, 512)
        self.video_from_roberta = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

        self.text_f_projection = nn.Linear(2560, 512)
        self.video_from_gpt = nn.MultiheadAttention(512, 4, dropout=0.1, batch_first=True)

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


    def load_ckpt(self):
        checkpoint = torch.load("./ckpts/pretrained_for_encoder_decoder.pth")
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
        self.skip_part.load_state_dict(video_encoder_dict)
        self.video_decoder_blocks.load_state_dict(video_decoder_dict)
        self.output_block.load_state_dict(video_output_dict)

    def frozen_model(self):
        for p in self.video_encoder.parameters():
            p.requires_grad = False

    # x: [64, 3, 15, 96, 96]
    # y: [64, 3, 15, 96, 96]
    # mel: [64, 48, 80]
    # text_embedding_f: [64, 1, 15, 2560]
    # encoded_texts: [64, 768]

    def forward(self, video, roberta_embedding, text_f):
        B = video.size(0)
        S = video.size(2)
        input_dim_size = len(video.size())
        if input_dim_size > 4:
            video = torch.cat([video[:, :, i] for i in range(video.size(2))], dim=0)


        feats = []
        x = video.clone()
        for f in self.skip_part:
            x = f(x.clone().cuda())
            feats.append(x.clone())
        feats.pop()
        x = video.clone()
        x = self.video_encoder(x.clone().cuda()).unsqueeze(-1).unsqueeze(-1)
        feats.append(x.clone())
        video_embedding = feats[-1].clone()

        roberta_encoding = self.roberta_mapping(roberta_embedding)
        roberta_encoding = F.normalize(roberta_encoding)
        roberta_encoding = roberta_encoding.repeat(S, 1).unsqueeze(1)
        video_roberta_encoding = F.normalize(video_embedding.squeeze())
        video_roberta_encoding = video_roberta_encoding.unsqueeze(1)

        cross_roberta = self.video_from_roberta(roberta_encoding, video_roberta_encoding, video_roberta_encoding)[
            0].squeeze().unsqueeze(-1).unsqueeze(-1)
        feats[-1] += cross_roberta

        if input_dim_size > 4:
            text_f = torch.cat([text_f[:, i, :] for i in range(text_f.size(1))], dim=0)
        text_embeds = self.text_f_projection(text_f)
        text_embeds = F.normalize(text_embeds)
        text_embeds = text_embeds.unsqueeze(1)
        cross_gpt = self.video_from_gpt(text_embeds, video_roberta_encoding, video_roberta_encoding)[0].squeeze()

        # x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
        # for f in self.video_decoder_blocks:
        #     x = f(x)
        #     if len(feats) != 0:
        #         x = torch.cat((x, feats[-1]), dim=1)
        #         feats.pop()
        # video_decoded_ = self.output_block(x)

        x = cross_gpt.unsqueeze(-1).unsqueeze(-1)
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

        return video_decoded


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    video = torch.normal(0, 1, [2, 3, 5, 96, 96]).cuda()
    roberta_embedding = torch.normal(0, 1, [2, 768]).cuda()
    text_f = torch.normal(0, 1, [2, 5, 2560]).cuda()

    # with torch.autograd.detect_anomaly():
    model = LLCFaceSy_vit_large().cuda()
    res = model(video, roberta_embedding, text_f)
    print(res.shape)
    # [64, 3, 15, 96, 96]
    l1 = nn.L1Loss()
    loss = l1(res, res)
    loss.backward()
