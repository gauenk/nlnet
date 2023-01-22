import torch as th
from einops import rearrange
import dnls

def init(cfg):

    # -- unpack params --
    ps      = cfg.ps
    pt      = cfg.pt
    dil     = cfg.dil
    exact = cfg.exact
    reflect_bounds = cfg.reflect_bounds

    # -- init --
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=0, exact=exact)

    return WpSumAgg(cfg.k_a,wpsum)

class WpSumAgg():

    def __init__(self,k_a,wpsum_fxn):
        self.k_a = self.k_a

    def __call__(self,vid,dists,inds):

        # -- limiting --
        dists = dists[...,:self.k_a].contiguous()
        inds = inds[...,:self.k_a].contiguous()

        # -- aggregate --
        patches = self.wpsum(vid,dists,inds)

        # -- reshape --
        ps = patches.shape[-1]
        shape_str = 'b h (o n) c ph pw -> (b o ph pw) n (h c)'
        patches = rearrange(patches,shape_str,o=ntotal)

        return patches
