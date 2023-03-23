
# -- dnls --
import torch as th
import dnls

# -- modules --
import importlib

# # -- local modules --
# from . import swin
# from . import nat
from . import nl
# from . import nlp
# from . import nlat
# from . import refine
# from . import csa

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init static variable
extract_config = econfig.extract_config # rename extraction

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Create the initial search function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

@econfig.set_init
def init_search(cfg):

    # -- unpack --
    econfig.init(cfg)
    cfg = econfig.extract_pairs(cfg,search_pairs())
    if econfig.is_init == True: return

    # -- create module --
    print(cfg.search_name)
    dnls_names = ["exact","nlat","nlas","approx_t","approx_s","approx_st","nlast",
                  "nl","nls","refine","refinement"]
    if cfg.search_name in dnls_names:
        print("dnls.")
        return load_dnls(cfg)
    else:
        print("local.")
        return load_local(cfg)

def load_dnls(search_cfg):
    return dnls.search.init(search_cfg)

def load_local(cfg):
    modules = {"nat":"nat"}
    if cfg.search_name in modules:
        mname = modules[cfg.search_name]
    else:
        mname = cfg.search_name
    return importlib.import_module("nlnet.search.%s" % mname).init(cfg)

def init(cfg):
    return init_search(cfg)

def search_pairs():
    # pairs0 = dnls.search.extract_config(cfg)
    pairs0 = {}
    pairs1 = {"ws":21,"wt":0,"ps":7,"k":10,"kr":1.,"wr":1,
              "wr_s":1,"kr_s":10,"wr_t":1,"kr_t":10,"scale":2,
              "pt":1,"exact":False,"rbwd":True,
              "nftrs_per_head":-1,"nchnls":-1,
              "nheads":1,"stride0":4,"stride1":1,
              "reflect_bounds":True,"use_k":True,"use_adj":True,
              "search_abs":False,"anchor_self":False,
              "dist_type":"l2","search_name":"nl","use_flow":True,
              "dilation":1,"use_state_update":False}
    pairs = pairs0 | pairs1
    return pairs

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
