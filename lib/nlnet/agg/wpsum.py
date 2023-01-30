import torch as th
import torch.nn as nn
from einops import rearrange
import dnls

def init(cfg):

    # -- unpack params --
    ps      = cfg.ps
    pt      = cfg.pt
    dil     = cfg.dilation
    exact = cfg.exact
    reflect_bounds = cfg.reflect_bounds

    # -- init --
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=0, exact=exact)

    return WpSumAgg(cfg.k_a,wpsum)

class WpSumAgg(nn.Module):

    def __init__(self,k,wpsum):
        super().__init__()
        self.k = k
        self.wpsum = wpsum

    def __call__(self,vid,dists,inds):

        # -- limiting --
        dists = dists[...,:self.k].contiguous()
        inds = inds[...,:self.k,:].contiguous()

        # -- aggregate --
        patches = self.wpsum(vid,dists,inds)

        # -- reshape --
        ps = patches.shape[-1]
        ntotal = dists.shape[-2]
        shape_str = 'b h (o n) c ph pw -> (b o ph pw) n (h c)'
        patches = rearrange(patches,shape_str,o=ntotal)

        return patches