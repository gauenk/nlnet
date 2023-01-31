import natten
import torch as th
import torch.nn as nn
from einops import rearrange

def init(cfg):
    F = cfg.nftrs_per_head * cfg.nheads
    return NATSearch(F, cfg.nheads, ps=cfg.ps)

def init_from_cfg(cfg):
    return init(cfg)

class NATSearch(nn.Module):

    def __init__(self, nftrs, nheads, k=7, ps=7):
        super().__init__()
        self.nftrs = nftrs
        self.nheads = nheads
        self.k = k
        self.ps = ps
        self.ws = ps
        self.dil = 1
        self.nheads = nheads
        self.nat_search = natten.NeighborhoodAttention2D(nftrs,nheads,
                                                         ps,self.dil).to("cuda:0")

    def forward(self,vid,vid1,fflow,bflow):
        B,T,C,H,W = vid.shape
        vid = rearrange(vid,'b t c h w -> (b t) h w c')
        attn = self.nat_search(vid)
        inds = th.zeros(1)
        return attn,inds

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        def wrap(vid0,vid1):
            return self.forward(vid0)
        return wrap

    def flops(self,B,HD,T,C,H,W):
        return 0
        # ps = self.ps
        # _C = C//self.nheads
        # nflops_per_search = 2*(ps*ps*_C)
        # nsearch_per_pix = ps*ps
        # nflops_per_pix = nsearch_per_pix * nflops_per_search
        # npix = B*self.nheads*H*W
        # nflops = nflops_per_pix * npix
        # return nflops

    def radius(self,*args):
        return self.ws
