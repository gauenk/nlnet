
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
from .rstb import RSTBWithInputConv


# @clean_code.add_methods_from(bench_mods)
class BlockList(nn.Module):
    def __init__(self, btype, blocklist, blocks):
        super().__init__()
        BlockLayer = get_block_version(blocklist.block_version)
        self.blocklist = blocklist


        # -- residual blocks --
        # nls_stack = blocklist.use_nls_stack
        self.blist_cat = blocklist.blist_cat
        mult = 2 if btype == "dec" else 1
        nres = blocklist.num_res
        n_feats = blocklist.embed_dim * blocklist.nheads * mult
        # n_feats = blocklist.embed_dim * mult
        ksize = blocklist.res_ksize
        append_noise = blocklist.append_noise and blocklist.enc_dec == "enc"
        self.nres = nres
        self.res = ResBlockList(blocklist.num_res,n_feats,ksize,
                                blocklist.res_bn,append_noise)
        nblocks = blocklist.depth+1

        # -- feature dimension --
        # _edim = blocks[-1].attn.embed_dim * mult
        # edim = _edim * blocklist.nheads * nblocks
        # print(edim,_edim,n_feats)

        # -- blocks --
        for d in range(blocklist.depth): blocks[d].bnum = d
        self.blocks = nn.ModuleList(
            [BlockLayer(btype,blocklist,blocks[d])
             for d in range(blocklist.depth)])

        # -- output --
        nblock_mult = nblocks if self.blist_cat else 1
        edim = n_feats * nblock_mult
        # print(n_feats,nblocks,edim)
        dprate = blocklist.drop_rate_path
        ksize = blocks[-1].res.res_ksize
        nres = blocks[-1].res.nres_per_block
        bn = blocks[-1].res.res_bn
        stg_nblocks = blocks[-1].res.stg_nblocks
        stg_depth = blocks[-1].res.stg_depth
        stg_nheads = blocks[-1].res.stg_nheads
        stg_ngroups = blocks[-1].res.stg_ngroups
        self.out = RSTBWithInputConv(edim, ksize, nres, dim=edim,
                                     depth=stg_depth,num_heads=stg_nheads,
                                     groups=stg_ngroups)
        self.conv_proj = nn.Sequential(
            nn.Conv3d(edim,n_feats,kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, ivid, flows=None, state=None):

        # -- residual blocks --
        vid = self.res(ivid)

        # -- cats --
        if self.blist_cat: ftrs = [ivid]
        else: ftrs = []

        # -- non-local blocks --
        for blk in self.blocks:
            vid = blk(vid,flows,state)
            if self.blist_cat: ftrs.append(vid)
        if self.blist_cat: ftrs = th.cat(ftrs,-3)
        else: ftrs = vid
        vid = self.out(ftrs)
        vid = self.conv_proj(vid.transpose(1,2)).transpose(1,2)

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
