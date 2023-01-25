"""

Pre-computed optical flows

"""
import dnls
import torch as th
import torch.nn as nn
from einops import rearrange

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
    use_adj = True
    use_self = anchor_self
    search = dnls.search.init("prod_pf_with_index", fflow, bflow,
                              k, ps, pt, ws, wt,
                              oh0, ow0, oh1, ow1,
                              chnls=-1,dilation=dil,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,
                              search_abs=search_abs,
                              use_adj=use_adj,use_k=use_k,
                              nbwd=nbwd,rbwd=rbwd,exact=exact)
    return search

def init_from_cfg(cfg):
    return init(cfg)

def init(cfg):
    return NLPSearch(cfg.k,cfg.ps,cfg.ws,cfg.wt,cfg.nheads,
                     cfg.stride0,cfg.stride1,cfg.run_acc_flow)

class NLPSearch(nn.Module):

    def __init__(self, k=7, ps=7, ws=8, wt=1, nheads=1,
                 stride0=4, stride1=1, run_acc=False):
        self.k = k
        self.ps = ps
        self.ws = ws
        self.nheads = nheads
        self.search = get_search(k,ps,ws,wt,nheads,stride0,stride1)
        self.ofa = dnls.nn.init("ofa")
        self.run_acc = run_acc

    # def __call__(self,vid,**kwargs):
    #     B,T,C,H,W = vid.shape
    #     flows = self.get_acc_flows(**kwargs)
    #     self.search.set_flows(flows,vid)
    #     dists,inds = self.search(vid)
    #     return dists,inds

    def forward(self,vid0,vid1,flows=None,state=None):
        B,T,C,H,W = vid0.shape
        flows = self.get_acc_flows(flows)
        self.search.set_flows(flows,vid0)
        dists,inds = self.search(vid0,vid1)
        return dists,inds

    def set_flows(self,vid,flows):
        self.search.set_flows(flows,vid)

    def get_acc_flows(self,flows)
        if self.run_acc:
            return self.run_acc_flows(flows)
        return flows

    def run_acc_flows(self,flows):
        if not(flows is None):
            flows = self.ofa(flows,stride0=self.search.stride0)
        return flows

    # -- Comparison API --
    def setup_compare(self,vid,flows,aflows,inds):
        def wrap(vid0,vid1):
            return self.forward(vid0,vid1,flows)
        return wrap

    def flops(self,B,C,H,W):
        return self.search.flops(B,C,H,W)

    def radius(self,*args):
        return self.ws

