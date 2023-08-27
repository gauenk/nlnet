# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from copy import deepcopy as dcopy

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
# from .nl_attn_vid import NonLocalAttentionVideo
from stnls.pytorch.nn import NonLocalAttention

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
from . import attn_mods
from .shared import get_norm_layer
from .mlps import init_mlp
# from .sk_conv import SKUnit
from .res import ResBlockList
from .misc import LayerNorm2d

class BlockV5(nn.Module):

    def __init__(self, btype, blocklist, block):
        super().__init__()

        # -- unpack vars --
        self.type = btype
        self.blocklist = blocklist
        self.dim = blocklist.embed_dim * blocklist.nheads
        self.mlp_ratio = blocklist.mlp_ratio
        self.block_mlp = blocklist.block_mlp
        self.drop_mlp_rate = blocklist.drop_rate_mlp
        self.drop_path_rate = blocklist.drop_rate_path
        norm_layer = get_norm_layer(blocklist.norm_layer)
        mult = 2 if self.type == "dec" else 1

        # -- modify embed_dim --
        block.attn.embed_dim *= mult
        edim = block.attn.embed_dim * blocklist.nheads
        self.edim = edim

        # -- norm layers --
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()
        self.norm1 = LayerNorm2d(edim)
        self.norm2 = LayerNorm2d(edim)

        # -- init non-local attn --
        attn = dcopy(block.attn)
        search = block.search
        normz = block.normz
        agg = block.agg
        self.attn = NonLocalAttention(attn,search,normz,agg)

        # # -- init proj --
        # self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
        #                           nn.Linear(2*attn.embed_dim,attn.embed_dim),
        #                           Rearrange('n d h w c -> n d c h w'))

        # -- init non-linearity --
        dprate = blocklist.drop_rate_path
        ksize = block.res.res_ksize
        nres = block.res.nres_per_block
        bn = block.res.res_bn
        self.res = ResBlockList(nres, edim, ksize, bn)
        self.drop_path = DropPath(dprate) if dprate > 0. else nn.Identity()

        # -- init mlp --
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(edim,2*edim,edim),
                                 Rearrange('n d h w c -> n d c h w'))


    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, vid, flows=None, state=None):

        # -=-=-=-=-=-=-=-=-=-=-=-=-
        #       Init/Unpack
        # -=-=-=-=-=-=-=-=-=-=-=-=-

        # -- create shortcut --
        B,T,C,H,W = vid.shape
        shortcut = vid

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    Non-Local Attn Layer
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = self.norm1(vid)
        vid = self.attn(vid, flows=flows, state=state)
        # vid = self.proj(vid) # back to input dim

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #   Non-Linearity & Residual
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = shortcut + self.drop_path(vid)
        vid = self.norm2(vid)
        # vid = self.res(vid)
        vid = vid + self.drop_path(self.mlp(self.res(vid)))

        return vid

    def flops(self,H,W):
        flops = 0
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        # flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops

class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

