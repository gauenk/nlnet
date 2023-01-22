
# -- local modules --
from . import swin
from . import nat
from . import nl
from . import nlp
from . import nlat
from . import refine
from . import csa


# -- config extraction --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,set_defaults
from dev_basics.common import extract_config,extract_pairs,cfg2lists
_fields = [] # fields for model io; populated using this code section
optional_full = partial(optional_fields,_fields)
extract_search_config = partial(extract_config,_fields) # all the aggregate fields


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Create the initial search function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def init_search(cfg):

    # -- unpack --
    __init = _optional(cfg,"__init",False)
    cfg = extract_search_config(cfg,optional)
    if __init == True: return

    # -- create module --
    modules = {"swin":swin,"nat":nat,"nl":nl,
               "refine":refine,"csa":csa,"nlp":nlp,
               "exact":nl,"nlat":nlat,"approx":nlat}
    mod = modules[cfg.search_name]
    search_fxn = getattr(mod,'init')(cfg)
    return search_fxn

def init(cfg):
    return init_search(cfg)

def extract_search_config(cfg,optional):
    pairs = {"ps":7,"pt":1,"k":10,"ws_r":1,
             "nftrs_per_head":-1,"nchnls":-1,
             "ws":21,"wt":0,"exact":False,"rbwd":True,
             "nheads":1,"stride0":4,"stride1":1,
             "reflect_bounds":True,"use_k":True,"use_adj":True,
             "search_abs":False,"anchor_self":False,
             "dist_type":"l2","search_name":"nl"}
    return extract_pairs(pairs,cfg,optional)

init_search({"__init"})


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#           Run Non-Local Search
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def run_search(vid0,vid1,state,cfg):

    # -- init --
    search_fxn = init_search(cfg)

    # -- run search --
    dists,inds = search_fxn(vid0,vid1,state)

    return dists,inds

    # if state is None:
    #     # -- dnls search --
    #     B, T, _, H, W = q_vid.shape
    #     qstart,stride0 = 0,cfg.stride0
    #     ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)
    #     dists,inds = cfg.search(q_vid,qstart,ntotal,k_vid)
    # else:
    #     # -- streaming search --
    #     dists,inds = run_state_search(q_vid,qstart,ntotal,k_vid,state)
    #     update_state(state,dists,inds)
    # return dists,inds
