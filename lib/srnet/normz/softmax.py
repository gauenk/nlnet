import torch as th
import torch.nn as nn
from einops import rearrange


def init(cfg):
    return SoftmaxNormalize(cfg.k_n,cfg.scale,cfg.normz_drop_rate)

class SoftmaxNormalize():

    def __init__(self,k,scale,drop_rate=0.):
        self.k = k
        self.scale = scale
        self.drop_rate = drop_rate
        self.norm = nn.Softmax(dim=-1)
        self.drop = nn.Dropout(drop_rate)

    def __call__(self,dists):

        # -- limiting --
        dists = dists[...,:self.k].contiguous()

        # -- scale --
        dists = self.scale * dists

        # -- normalize --
        dists = self.norm(dists)

        # -- drop-rate --
        dists = self.drop(dists)

        # -- contiguous --
        dists = dists.contiguous()

        return dists
