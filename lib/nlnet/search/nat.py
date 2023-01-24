import natten
import torch as th
import torch.nn as nn
from einops import rearrange

def init(cfg):
    F = cfg.nftrs_per_head * cfg.nheads
    return NLSearch(F, cfg.nheads, ps=cfg.ps)

def init_from_cfg(cfg):
    return init(cfg)

class NLSearch():
    def __init__(self, nftrs, nheads, k=7, ps=7):
        self.nftrs = nftrs
        self.nheads = nheads
        self.k = k
        self.ps = ps
        self.ws = ps
        self.dil = 1
        self.nheads = nheads
        print(self.nftrs,self.nheads)
        self.nat_search = natten.NeighborhoodAttention2D(nftrs,nheads,
                                                         ps,self.dil).to("cuda:0")

    def __call__(self,vid,**kwargs):
        B,T,C,H,W = vid.shape
        vid = rearrange(vid,'b t c h w -> (b t) h w c')
        attn = self.nat_search(vid)
        inds = th.zeros(1)
        return attn,inds

    def flops(self,B,C,H,W):
        ps = self.ps
        _C = C//self.nheads
        nflops_per_search = 2*(ps*ps*_C)
        nsearch_per_pix = ps*ps
        nflops_per_pix = nsearch_per_pix * nflops_per_search
        npix = B*self.nheads*H*W
        nflops = nflops_per_pix * npix
        return nflops

    def radius(self,*args):
        return self.ws
