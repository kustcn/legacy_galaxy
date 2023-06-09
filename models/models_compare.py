
import torchvision.models as models

from functools import partial
""" Layer/Module Helpers

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from itertools import repeat
# from torch._six import container_abcs
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


import torch
import torch.nn as nn

import timm.models.vision_transformer
import torchvision.models as models
from torch import nn
from torchvision.models import efficientnet_b3,efficientnet_b0,efficientnet_b5,alexnet,resnet50,resnet18
#%%


class vgg(nn.Module):
    def __init__(self,num_classes=7):
        super(vgg,self).__init__()
        # 定义编码器
        vgg_model = models.vgg16(pretrained=False)
        num_features = vgg_model.classifier[6].in_features

        vgg_model.classifier[6] = nn.Linear(num_features, 7)
        self.feature=vgg_model



    def forward(self,x):
        
        out=self.feature(x)

        return out

# %%




        
class effientnet_b5(nn.Module):
    def __init__(self):
        super(effientnet_b5, self).__init__()
        
        self.efficientnet_b5 = efficientnet_b5(pretrained=False)
     #    self.net = nn.Sequential(*list(self.effientnet_b5.children())[:-1])
        infeature=self.efficientnet_b5.classifier[1].in_features
        self.efficientnet_b5.classifier[1] = nn.Linear(infeature, 7)

        
    def forward(self, x):
     #    out1 = self.net(x)
        out = self.efficientnet_b5(x)
        return out
    

class alex(nn.Module):
    def __init__(self):
        super(alex, self).__init__()
        
        self.alex = alexnet(pretrained=False)
     #    self.net = nn.Sequential(*list(self.effientnet_b5.children())[:-1])
        infeature=self.alex.classifier[6].in_features
        self.alex.classifier[6] = nn.Linear(infeature, 7)

        
    def forward(self, x):
     #    out1 = self.net(x)
        out = self.alex(x)
        return out
    
class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        
        self.re50 = resnet50(pretrained=False)
     #    self.net = nn.Sequential(*list(self.effientnet_b5.children())[:-1])
        infeature=self.re50.fc.in_features
        self.re50.fc = nn.Linear(infeature, 7)

        
    def forward(self, x):
     #    out1 = self.net(x)
        out = self.re50(x)
        return out
    

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        
        self.re18 = resnet18(pretrained=False)
     #    self.net = nn.Sequential(*list(self.effientnet_b5.children())[:-1])
        infeature=self.re18.fc.in_features
        self.re18.fc = nn.Linear(infeature, 7)

        
    def forward(self, x):
     #    out1 = self.net(x)
        out = self.re18(x)
        return out
    
import torch
from torchvision.models import vit_b_16

# 导入 Vision Transformer (ViT) 模型

# 打印模型结构
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple







class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=112, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.patch_embed= PatchEmbed(
                img_size=112, patch_size=8, in_chans=3, embed_dim=768)
        num_patches = self.patch_embed.num_patches
        self.head = nn.Linear(768, 7) 

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

