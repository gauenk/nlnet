

import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    """ copied from https://github.com/rwightman/pytorch-image-models/blob/d7b55a9429f3d56a991e604cbc2e9fdf1901612f/timm/models/layers/norm.py#L26 """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self,vid: torch.Tensor) -> torch.Tensor:
        B,T = vid.shape[:2]
        vid = rearrange(vid,'b t c h w -> (b t) c h w ')
        vid = F.layer_norm(vid.permute(0, 2, 3, 1), self.normalized_shape,
                           self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
        vid = rearrange(vid,'(b t) c h w -> b t c h w ',b=B)
        vid = vid.contiguous()
        return vid
