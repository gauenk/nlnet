import stnls
import torch as th
import torch.nn as nn
from einops import rearrange

from . import state_mod
from dev_basics.utils import clean_code


def get_search(k,ps,ws,wt,nheads,stride0,stride1):
    pt = 1
    dil = 1
    use_k = True
    reflect_bounds = True
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nbwd = 1
    rbwd,exact = False,False
    anchor_self = False
    use_self = anchor_self
    search = stnls.search.init("prod_with_heads", fflow, bflow,
                              k, ps, pt, ws, wt, nheads, chnls=-1,
                              dilation=dil, stride0=stride0,stride1=stride1,
                              h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,
                              anchor_self=anchor_self,use_self=use_self)
    return search

def init_from_cfg(cfg):
    return init(cfg)

def init(cfg):
    return NLSearch(cfg.k,cfg.ps,cfg.ws,cfg.wt,cfg.nheads,
                    cfg.stride0,cfg.stride1,cfg.dilation,
                    cfg.use_flow,cfg.use_state_update)

@clean_code.add_methods_from(state_mod)
class NLSearch(nn.Module):

    def __init__(self, k=7, ps=7, ws=8, wt=1, nheads=1,
                 stride0=4, stride1=1, dilation=1, use_flow=True,
                 use_state_update=False):
        super().__init__()
        self.k = k
        self.ps = ps
        self.ws = ws
        self.wt = wt
        self.nheads = nheads
        self.stride0 = stride0
        self.stride1 = stride1
        self.nheads = nheads
        self.dilation = dilation
        self.use_flow = use_flow
        self.use_state_update = use_state_update
        self.search = get_search(k,ps,ws,wt,nheads,stride0,stride1)

    # -- Model API --
    def forward(self,vid0,vid1,flows=None,state=None):
        B,T,C,H,W = vid0.shape
        dists,inds = self.search(vid0,vid1)
        # self.update_state(state,dists,inds,vid0.shape)
        return dists,inds

    def set_flows(self,vid,flows):
        self.search.set_flows(flows,vid)

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        self.set_flows(vid,flows)
        def wrap(vid0,vid1):
            return self.forward(vid0,vid1,flows)
        return wrap

    def flops(self,B,C,H,W):
        return self.search.flops(B,C,H,W)

    def radius(self,*args):
        return self.ws
