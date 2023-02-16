
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
from .mlps import FastLeFF,LeFF,Mlp
from .nl_attn import NonLocalAttention

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
# from . import attn_mods
# from dev_basics.utils import clean_code

# -- local --
from .blocks import get_block_version
from .res import ResBlockList

# @clean_code.add_methods_from(bench_mods)
class BlockList(nn.Module):
    def __init__(self, btype, blocklist, blocks):
        super().__init__()
        num_res = 0
        BlockLayer = get_block_version(blocklist.block_version)
        self.blocklist = blocklist

        # -- blocks --
        self.blocks = nn.ModuleList(
            [BlockLayer(btype,blocklist,blocks[d])
             for d in range(blocklist.depth)])

        # -- residual blocks --
        mult = 2 if btype == "dec" else 1
        nres = blocklist.num_res
        n_feats = blocklist.embed_dim * blocklist.nheads * mult
        ksize = blocklist.res_ksize
        self.nres = nres
        if nres > 0:
            self.res = ResBlockList(blocklist.num_res,n_feats,ksize)
        else:
            self.res = []

    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, vid, flows=None, state=None):

        # -- residual blocks --
        if self.nres > 0:
            vid = self.res(vid)

        # -- non-local blocks --
        state_b = [state[0],None]
        for blk in self.blocks:
            vid = blk(vid,flows,state_b)
            state_b = [state_b[1],None]
        state[1] = state_b[0]

        return vid

    def flops(self,h,w):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(h,w)
        return flops

    @property
    def times(self):
        times = ExpTimerList()
        for blk in self.blocks:
            times.update_times(blk.attn.timer)
            blk.attn.reset_times()
        return times

    def reset_times(self):
        for blk in self.blocks:
            blk.attn.reset_times()
