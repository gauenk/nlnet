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
from stnls.nn import NonLocalAttentionStack

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
from . import attn_mods
from .shared import get_norm_layer
from .mlps import init_mlp
# from .sk_conv import SKUnit
from .res import ResBlockList
from .misc import LayerNorm2d
from .rstb import RSTBWithInputConv
from .chnls_attn import ChannelAttention

class BlockV11(nn.Module):

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
        self.attn = NonLocalAttentionStack(attn,search,normz,agg)

        # -- init non-linearity --
        dprate = blocklist.drop_rate_path
        ksize = block.res.res_ksize
        nres = block.res.nres_per_block
        bn = block.res.res_bn
        stg_depth = block.res.stg_depth
        stg_nheads = block.res.stg_nheads
        stg_ngroups = block.res.stg_ngroups
        self.channel_attn_0 = ChannelAttention(edim)
        self.channel_attn_1 = ChannelAttention(edim)
        self.res = RSTBWithInputConv(edim, ksize, nres, dim=edim,
                                     depth=stg_depth,num_heads=stg_nheads,
                                     groups=stg_ngroups)
        self.drop_path = DropPath(dprate) if dprate > 0. else nn.Identity()


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
        # print("[in] vid.shape: ",vid.shape)
        vid = self.attn(vid, flows=flows, state=state)
        # print("[out] vid.shape: ",vid.shape)
        # print("[attn] delta: ",th.mean((shortcut-vid)**2).item())
        vid = self.channel_attn_0(vid)
        # print("[chnl_attn] delta: ",th.mean((shortcut-vid)**2).item())

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #   Non-Linearity & Residual
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = shortcut + self.drop_path(vid)
        # print("[shortcut] delta: ",th.mean((shortcut-vid)**2).item())
        vid = self.norm2(vid)
        vid = self.res(vid)
        vid = self.channel_attn_1(vid)

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

