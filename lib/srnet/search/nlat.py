"""

Non-local Search with Approximate Temporal Search

"""


import dnls
import torch as th
import torch.nn as nn
from einops import rearrange

def get_exact_search(k,ps,ws,wt,nheads,stride0,stride1):
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
    search = dnls.search.init("prod_with_heads", fflow, bflow,
                              k, ps, pt, ws, wt, nheads, chnls=-1,
                              dilation=dil, stride0=stride0,stride1=stride1,
                              h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,
                              anchor_self=anchor_self,use_self=use_self)
    return search

def get_refine_search(k,ps,ws_r,ws,nheads,stride0,stride1):
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
    search = dnls.search.init("prod_refine",  k, ps, pt, ws_r, ws,
                              nheads, chnls=-1, dilation=dil,
                              stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=True,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,
                              anchor_self=anchor_self,use_self=use_self)
    return search

def init_from_cfg(cfg):
    return init(cfg)

def init(cfg):
    return NLSearch(cfg.k,cfg.ps,cfg.ws_r,cfg.ws,cfg.wt,
                    cfg.nheads,cfg.stride0,cfg.stride1)

class NLSearch():

    def __init__(self, k=7, ps=7, ws_r=3, ws=8, wt=1, nheads=1, stride0=4,stride1=1):
        self.k = k
        self.ps = ps
        self.ws = ws
        self.wt = wt
        self.nheads = nheads
        self.esearch = get_exact_search(k,ps,ws,0,nheads,stride0,stride1)
        self.rsearch = get_refine_search(k,ps,ws_r,ws,nheads,stride0,stride1)

    def __call__(self,vid,**kwargs):
        B,T,C,H,W = vid.shape
        dists,inds = self.esearch(vid)
        if self.wt > 0:
            inds_t = self.apply_offsets(inds,kwargs['flows'])
            _dists,_inds = self.rsearch(vid,0,inds_t)
            dists = th.cat([dists,_dists],2)
            inds = th.cat([inds,_inds],2)
        return dists,inds

    def apply_offsets(self,inds,flows):
        inds_t = dnls.nn.temporal_inds(inds[:,0],flows,self.wt)
        inds_t = rearrange(inds_t,'b q k s tr -> b 1 (q s) k tr')
        inds_t = inds_t.contiguous()
        return inds_t

    def set_flows(self,vid,flows,aflows):
        self.esearch.set_flows(flows,vid)

    def flops(self,B,C,H,W):
        return self.esearch.flops(B,C,H,W)

    def radius(self,*args):
        return self.ws
