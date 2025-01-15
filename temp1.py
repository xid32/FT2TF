import torch.nn as nn
import torch
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
a = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
b = torch.normal(0, 1, [2, 3, 256, 256])

c = a(b)



