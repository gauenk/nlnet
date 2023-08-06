"""

Processing a stack of non-local patches

"""

# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from copy import deepcopy as dcopy

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
from stnls.pytorch.nn import NonLocalAttention

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- misc --
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -- "backbone"
from .res import ResBlockList

class NonLocalStack(nn.Module):

    def __init__(self, stack_cfg):
        super().__init__()
        stack_cfg
        if nres > 0:
            res = []
            for _ in range(nres):
                res.append(ResBlock(conv, n_feats, kernel_size))
            self.res = nn.Sequential(*res)
        else:
            self.res = nn.Identity()

        # -- init attn fxns --
        # self.k_stack = 4
        self.search = stnls.search.init(search_cfg)
        self.normz = stnls.normz.init(normz_cfg)
        self.agg = stnls.reducer.init(agg_cfg)

        # -- "backbone" --
        self.feat_extract = RSTBWithInputConv(in_channels=self.k*n_feats,
                                              kernel_size=(1, 3, 3),
                                              groups=1,
                                              num_blocks=2,
                                              dim=n_feats,depth=2,
                                              num_heads=4,window_size=[1, 8, 8],
                                              mlp_ratio=2.,qkv_bias=True,qk_scale=None,
                                              use_checkpoint_attn=[False],
                                              use_checkpoint_ffn=[False]
        )

    def forward(self,vid,state):

        # -- search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)

        # -- weighted stack --
        stacked = self.run_stacking(v_vid,dists,inds)

        # -- propagate through network --
        vid = vid + self.feat_extract(stacked)

        return vid

class NonLocalResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.PReLU(), res_scale=1):
        super().__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res

