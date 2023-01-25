import torch as th
from torch.nn.functional import fold,unfold
from einops import rearrange,repeat


def iunfold(vid,ps,stride):
    patches = unfold(vid,(ps,ps),stride=stride).transpose(-2, -1)
    return patches

def init(cfg):
    return NLSearch(cfg.k,cfg.ps,cfg.nheads,cfg.stride0,cfg.stride1)

def init_from_cfg(cfg):
    return init(cfg)

class NLSearch(nn.Module):

    def __init__(self,k=7, ps=7, nheads=1, stride0=1, stride1=1):
        self.k = k
        self.ps = ps
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1

    def forward(self,vid):
        nheads = self.nheads
        B,T,C,H,W = vid.shape
        vid = rearrange(vid,'b t (H c) h w -> (b t H) c h w',H=nheads)
        patches = iunfold(vid,self.ps,self.stride0)
        q = patches
        k = iunfold(vid,self.ps,self.stride1)
        v = k
        attn = th.matmul(q,k.transpose(-2, -1))
        attn = rearrange(attn,'(b t H) d0 d1 -> b t H d0 d1',b=B,H=nheads)
        inds = th.zeros((1))
        return attn,inds

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        def wrap(vid0,vid1):
            return self.forward(vid0)
        return wrap

    def flops(self,B,C,H,W):

        # -- init --
        num = B * self.nheads
        ps = self.ps
        dim = ps*ps*(C//self.nheads)
        nH0 = (H-1)//self.stride0+1
        nW0 = (W-1)//self.stride0+1
        nH1 = (H-1)//self.stride1+1
        nW1 = (W-1)//self.stride1+1
        N0 = nH0*nW0
        N1 = nH1*nW1
        nflops = num * N0 * N1 * (dim + dim)
        return nflops

    def radius(self,H,W):
        return max(H,W)
