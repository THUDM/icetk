# -*- encoding: utf-8 -*-
'''
@File    :   image_tokenizer.py
@Time    :   2021/12/20 14:19:49
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random

import torch
import torch.nn.functional as F
from torchvision import transforms 

from .vqvae import load_default_HVQVAE, load_ckpt

class ImageTokenizer(object):
    def __init__(self,
                model_path,
                device='cuda',
                fp16=True):
        model = load_default_HVQVAE()
        model = load_ckpt(model, model_path)
        model = model.to(device)
        model.eval()
        
        self.tr_normalize = transforms.Normalize(
            [0.79093, 0.76271, 0.75340], 
            [0.30379, 0.32279, 0.32800]
            )

        self.model = model
        self.device = device
        self.fp16 = fp16
        self.num_tokens = model.quantize.n_embed
        
        if fp16:
            model = model.half()

    def __len__(self):
        return self.num_tokens

    def encode(self, image_torch, l=1):
        '''Convert a batch of img to code
        Args:
            model: The tokenizer model.
            img: [b, c, h, w]
        '''
        if len(image_torch.shape) == 3:
            image_torch = image_torch.unsqueeze(0)
        img = self.tr_normalize(image_torch).to(self.device)
        if self.fp16:
            img = img.half()
        with torch.no_grad():
            quant, diff, id = self.model.single_encode(img, l)
        return id.view(img.shape[0], -1)

    def decode(self, codes, l=1):
        '''Convert a batch of code to imgs
        Args:
            codes : [b, h, w] or [b, h*w] or [h*w] LongTensor / list
        '''
        if isinstance(codes, list):
            codes = torch.tensor(codes, dtype=torch.long, device=self.device)
        if len(codes.shape) == 1:   
            codes = codes.unsqueeze(0)
        if len(codes.shape) == 2:
            s = int(math.sqrt(len(codes.view(-1))) + 1e-5)
            codes = codes.view(codes.shape[0], s, s)
        with torch.no_grad():
            out = self.model.single_decode_code(codes, l)
            out = out * torch.tensor([0.30379, 0.32279, 0.32800], device=out.device).view(1, -1, 1, 1) + torch.tensor([0.79093, 0.76271, 0.75340], device=out.device).view(1, -1, 1, 1)
        return out