# -- misc --
import dnls
from copy import deepcopy as dcopy
from easydict import EasyDict as edict
from ..utils import optional

# -- torch --
import torch as th
from einops import rearrange

# -- clean code --
from dev_basics.utils import clean_code
__methods__ = [] # self is a DataStore
register_method = clean_code.register_method(__methods__)


# @register_method
# def init_search(self,search_name,search_cfg):
#     search_cfg.search_name = search_name
#     return dnls.search.init(search_cfg)
#     # if "full" in search_name:
#     #     search = self.init_full(search_cfg)
#     # elif "approx" in search_name:
#     #     search = self.init_approx(search_cfg)
#     # elif "refine" in search_name:
#     #     search = self.init_refine(search_cfg)
#     # else:
#     #     raise ValueError(f"Uknown search function [{search_name}]")
#     # return search

# @register_method
# def init_full(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,stride0=4,stride1=1,
#                 dilation=1,rbwd=True,nbwd=1,exact=False,
#                 reflect_bounds=False):
#     use_k = k > 0
#     search_abs = ws == -1
#     use_adj = True
#     oh0,ow0,oh1,ow1 = 1,1,3,3
#     anchor_self = True
#     # anchor_self = False
#     if search_abs:
#         use_adj = True
#         oh0,ow0,oh1,ow1 = 1,1,3,3
#     full_ws = False
#     fflow,bflow = None,None
#     use_self = anchor_self
#     search = dnls.search.init("prod_with_index", fflow, bflow,
#                               k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
#                               dilation=dilation, stride0=stride0,stride1=stride1,
#                               reflect_bounds=reflect_bounds,use_k=use_k,
#                               use_adj=use_adj,search_abs=search_abs,
#                               rbwd=rbwd,nbwd=nbwd,exact=exact,full_ws=full_ws,
#                               anchor_self=anchor_self,use_self=use_self)
#     return search

# @register_method
# def init_approx(self,k=100,ps=7,pt=0,ws=21,ws_ap=3,wt=0,stride0=4,stride1=1,
#                 dilation=1,rbwd=True,nbwd=1,exact=False,
#                 reflect_bounds=False):

#     # -- full --
#     use_k = k > 0
#     search_abs = ws == -1
#     use_adj = True
#     oh0,ow0,oh1,ow1 = 1,1,3,3
#     anchor_self = True
#     # anchor_self = False
#     if search_abs:
#         use_adj = True
#         oh0,ow0,oh1,ow1 = 1,1,3,3
#     full_ws = False
#     fflow,bflow = None,None
#     use_self = anchor_self
#     search = dnls.search.init("prod_with_index", fflow, bflow,
#                               k, ps, pt, ws, wt,oh0, ow0, oh1, ow1, chnls=-1,
#                               dilation=dilation, stride0=stride0,stride1=stride1,
#                               reflect_bounds=reflect_bounds,use_k=use_k,
#                               use_adj=use_adj,search_abs=search_abs,
#                               rbwd=rbwd,nbwd=nbwd,exact=exact,full_ws=full_ws,
#                               anchor_self=anchor_self,use_self=use_self)
#     return search

# @register_method
# def init_refine(self,k=100,ps=7,pt=0,ws=21,ws_r=3,wt=0,
#                 stride0=4,stride1=1,dilation=1,rbwd=True,nbwd=1,exact=False,
#                 reflect_bounds=False):
#     use_k = k > 0
#     search_abs = False
#     fflow,bflow = None,None
#     oh0,ow0,oh1,ow1 = 1,1,3,3
#     nheads = 1
#     anchor_self = False
#     use_self = anchor_self
#     search = dnls.search.init("prod_refine", k, ps, pt, ws_r, ws, nheads,
#                               chnls=-1,dilation=dilation,
#                               stride0=stride0, stride1=stride1,
#                               reflect_bounds=reflect_bounds,use_k=use_k,
#                               search_abs=search_abs,use_adj=True,
#                               anchor_self=anchor_self,use_self=use_self,
#                               exact=exact)
#     return search

@register_method
def init_fold(self,vshape,device):
    dil     = self.dilation
    stride0 = self.stride0
    only_full = False
    reflect_bounds = True
    fold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                       adj=0,only_full=only_full,
                       use_reflect=reflect_bounds,device=device)
    return fold


@register_method
def run_fold(self,patches,vshape):

    # -- timing --
    self.timer.sync_start("fold")

    # -- init folding --
    B,ps = vshape[0],self.search_cfg.ps
    fold = self.init_fold(vshape,patches.device)

    # -- reshape for folding --
    shape_str = '(b o ph pw) n c -> b (o n) 1 1 c ph pw'
    patches = rearrange(patches,shape_str,b=B,ph=ps,pw=ps)
    patches = patches.contiguous()

    # -- fold --
    fold(patches)

    # -- unpack --
    vid = fold.vid / fold.zvid

    # -- debug --
    any_nan = th.any(th.isnan(vid))
    if any_nan:
        any_fold_nan = th.any(th.isnan(fold.vid))
        any_zero = th.any(th.abs(fold.zvid)<1e-10)
        print("[%s] found a nan!: " % __file__,any_nan,any_zero,any_fold_nan)
        exit(0)

    # -- timing --
    self.timer.sync_stop("fold")

    return vid
