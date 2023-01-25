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

def interp_inds(scale,stride,T,H,W):
    return dnls.search.init("interpolate_inds",scale,stride,T,H,W)

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
    return NLSApproxSpace(cfg.k,cfg.ps,cfg.ws_r,cfg.ws,cfg.wt,
                          cfg.nheads,cfg.stride0_a,cfg.stride0,cfg.stride1)

class NLSApproxSpace(nn.Module):

    def __init__(self, k=7, ps=7, ws_r=1, ws=8, wt=1,
                 nheads=1, stride0_a=8, stride0=4, stride1=1):
        super().__init__()
        self.k = k
        self.ps = ps
        self.ws = ws
        self.wt = wt
        self.nheads = nheads
        assert stride0_a % stride0 == 0,"Must be an integer multiple."
        self.stride0 = stride0
        self.scale = stride0_a//stride0
        self.esearch = get_exact_search(k,ps,ws,wt,nheads,stride0_a,stride1)
        self.rsearch = get_refine_search(k,ps,1,ws,nheads,stride0,stride1)

    def forward(self,vid0,vid1,flows):
        B,T,C,H,W = vid0.shape
        dists,inds = self.esearch(vid0,vid1)
        dists,inds = self.upsample(vid0,vid1,dists,inds)
        return dists,inds

    def upsample(self,vid0,vid1,dists_a,inds_a):
        B,T,C,H,W = vid0.shape
        interp = interp_inds(self.scale,self.stride0,T,H,W)
        inds_f = interp(inds_a)
        dists_f = self.fill_dists(vid0,vid1,dists_a,inds_f)
        return dists_f,inds_f

    def fill_dists(self,vid0,vid1,dists_a,inds_f):
        inds_n = self.new_inds(inds_f)
        dists_n = self.rsearch(vid0,vid1,inds_n)
        dists_f = self.fill_dists(dists_a,dists_n)
        return dists_f

    # def fill_dists(dists_a,dists_n):
    #     return dists_f

    def new_inds(self,inds_f):
        inds_n = "inds_f - inds_a"
        return inds_n

    def apply_offsets(self,inds,flows):
        inds_t = dnls.nn.temporal_inds(inds[:,0],flows,self.wt)
        inds_t = rearrange(inds_t,'b q k s tr -> b 1 (q s) k tr')
        inds_t = inds_t.contiguous()
        return inds_t

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        def wrap(vid0,vid1):
            return self.forward(vid0,vid1,flows)
        return wrap

    def set_flows(self,vid,flows,aflows):
        self.esearch.set_flows(flows,vid)

    def flops(self,B,C,H,W):
        return self.esearch.flops(B,C,H,W)

    def radius(self,*args):
        return self.ws
