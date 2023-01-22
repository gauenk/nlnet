
import torch as th
from einops import rearrange,repeat


def unfold(x,ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws*ws,C)
    return windows

def fold(windows,ws,B,H,W):
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def windows_to_qkv(windows,nheads):
    B_,N,C = windows.shape
    q = windows.reshape(B_, N, 1, nheads, C // nheads).permute(2, 0, 3, 1, 4)
    # q = rearrange(windows,'l n (c h) -> l h n c',h=nheads)
    return q,q,q

def init(cfg):
    return NLSearch(cfg.k,cfg.ps,cfg.nheads)

def init_from_cfg(cfg):
    return init(cfg)

class NLSearch():

    def __init__(self,k=7, ps=7, nheads=1):
        self.k = k
        self.ps = ps
        self.ws = 8
        self.nheads = nheads

    def __call__(self,vid,**kwargs):
        B,T,C,H,W = vid.shape
        vid = vid
        vid = rearrange(vid,'b t c h w -> (b t) h w c')
        windows = unfold(vid,self.ws)
        # print("windows.shape: ",windows.shape)
        q,k,v = windows_to_qkv(windows,self.nheads)
        # print("q.shape: ",q.shape)
        attn = (q @ k.transpose(-2, -1))
        # print("attn.shape: ",attn.shape)
        inds = th.zeros_like(attn).type(th.int32)
        return attn,inds

    def flops(self,B,C,H,W):

        #
        # -- init --
        #

        ws = self.ws
        assert (H % ws == 0) and (W % ws == 0)
        N = ws**2
        nW = (H*W)//N
        _C = (C // self.nheads)
        dim = _C# * N

        #
        # -- compute --
        #

        # attn = (q @ k.transpose(-2, -1))
        nflops_outer_prod = nW *  nW * (dim + dim)
        nflops = B * self.nheads * nflops_outer_prod
        return nflops

    def radius(self,*args):
        return self.ws
