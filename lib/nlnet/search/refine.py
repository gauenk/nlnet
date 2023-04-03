
import torch as th
import torch.nn as nn
import stnls
from einops import rearrange

from . import state_mod
from dev_basics.utils import clean_code

def get_search(k,ps,ws_r,ws,nheads,stride0,stride1):
    pt = 1
    oh0,ow0,oh1,ow1 = 0,0,0,0
    dil = 1
    use_k = True
    reflect_bounds = True
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nbwd = 1
    rbwd,exact = False,False
    anchor_self = True
    use_self = anchor_self
    search = stnls.search.init("prod_refine",  k, ps, pt, ws_r, ws,
                              nheads, chnls=-1, dilation=dil,
                              stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,
                              anchor_self=anchor_self,use_self=use_self)
    return search

def init(cfg):
    return NLSRefine(cfg.k,cfg.k_s,cfg.ps,cfg.ws_r,cfg.ws,
                     cfg.nheads,cfg.stride0,cfg.stride1)

def init_from_cfg(cfg):
    return init(cfg)

@clean_code.add_methods_from(state_mod)
class NLSRefine(nn.Module):

    def __init__(self, k=7, k_s=50, ps=7, ws_r=1, ws=8,
                 nheads=1, stride0=4, stride1=1):
        super().__init__()
        self.k = k
        self.k_s = k_s
        self.ps = ps
        self.ws = ws
        self.stride0 = stride0
        self.nheads = nheads
        self.use_state_update = True
        self.search = get_search(k,ps,ws_r,ws,nheads,stride0,stride1)

    def forward(self,vid0,vid1,flows=None,state=None):
        inds_p = self.unpack_state(state)
        B,T,C,H,W = vid0.shape
        dists,inds = self.search(vid0,vid1,inds_p)
        self.update_state(state,dists,inds,vid0.shape)
        return dists,inds

    def set_flows(self,vid,flows):
        pass

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        state = [inds,None]
        def wrap(vid0,vid1):
            return self.forward(vid0,vid1,flows,state)
        return wrap

    def flops(self,B,C,H,W):
        return self.search.flops(B,C,H,W,self.k_s)

    def radius(self,*args):
        return self.ws

