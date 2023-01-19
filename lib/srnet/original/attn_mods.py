# -- misc --
import dnls
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
from ..utils import optional

# -- clean code --
from dev_basics.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)


@register_method
def init_search(self,search_cfg):
    stype = search_cfg.search_type
    if "full" in stype:
        search = self.init_full(search_cfg)
    elif "approx" in stype:
        search = self.init_approx(search_cfg)
    elif "refine" in stype:
        search = self.init_refine(search_cfg)
    else:
        raise ValueError(f"Uknown search function [{stype}]")
    return search

@register_method
def init_full(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,stride0=4,stride1=1,
                dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = ws == -1
    use_adj = True
    oh0,ow0,oh1,ow1 = 1,1,3,3
    anchor_self = True
    # anchor_self = False
    if search_abs:
        use_adj = True
        oh0,ow0,oh1,ow1 = 1,1,3,3
    full_ws = False
    fflow,bflow = None,None
    use_self = anchor_self
    search = dnls.search.init("prod_with_index", fflow, bflow,
                              k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
                              dilation=dilation, stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=use_adj,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,full_ws=full_ws,
                              anchor_self=anchor_self,use_self=use_self)
    return search

@register_method
def init_approx(self,k=100,ps=7,pt=0,ws=21,ws_ap=3,wt=0,stride0=4,stride1=1,
                dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):

    # -- full --
    use_k = k > 0
    search_abs = ws == -1
    use_adj = True
    oh0,ow0,oh1,ow1 = 1,1,3,3
    anchor_self = True
    # anchor_self = False
    if search_abs:
        use_adj = True
        oh0,ow0,oh1,ow1 = 1,1,3,3
    full_ws = False
    fflow,bflow = None,None
    use_self = anchor_self
    search = dnls.search.init("prod_with_index", fflow, bflow,
                              k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
                              dilation=dilation, stride0=stride0,stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              use_adj=use_adj,search_abs=search_abs,
                              rbwd=rbwd,nbwd=nbwd,exact=exact,full_ws=full_ws,
                              anchor_self=anchor_self,use_self=use_self)
    return search

@register_method
def init_refine(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,
                stride0=4,stride1=1,dilation=1,rbwd=True,nbwd=1,exact=False,
                reflect_bounds=False):
    use_k = k > 0
    search_abs = False
    fflow,bflow = None,None
    oh0,ow0,oh1,ow1 = 1,1,3,3
    nheads = 1
    anchor_self = False
    use_self = anchor_self
    search = dnls.search.init("prod_refine", k, ps, pt, ws_r, ws, nheads,
                              chnls=-1,dilation=dilation,
                              stride0=stride0, stride1=stride1,
                              reflect_bounds=reflect_bounds,use_k=use_k,
                              search_abs=search_abs,use_adj=True,
                              anchor_self=anchor_self,use_self=use_self,
                              exact=exact)
    return search

@register_method
def init_wpsum(self,cfg):

    # -- unpack params --
    ps      = cfg.ps
    pt      = cfg.pt
    dil     = cfg.dil

    # -- fixed --
    exact = False
    reflect_bounds = True

    # -- init --
    wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                dilation=dil,
                                                reflect_bounds=reflect_bounds,
                                                adj=0, exact=exact)
    return wpsum

@register_method
def init_fold(self,vshape,device):
    dil     = self.search_cfg.dil
    stride0 = self.search_cfg.stride0
    only_full = False
    reflect_bounds = True
    fold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                       adj=0,only_full=only_full,
                       use_reflect=reflect_bounds,device=device)
    return fold

