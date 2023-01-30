
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
    econfig.set_cfg(cfg)
    cfgs = econfig({"search":search_pairs()})
    if econfig.is_init == True: return
    cfg = cfgs.search

    # -- create module --
    modules = {"exact":"nl","approx_t":"nlat","approx_s":"nlas",
               "approx_st":"nlas"}
    if cfg.search_name in modules:
        mname = modules[cfg.search_name]
    else:
        mname = cfg.search_name
    module = importlib.import_module("nlnet.search."+mname)
    search_fxn = getattr(module,'init')(cfg)
    return search_fxn

def init(cfg):
    return init_search(cfg)

def search_pairs():
    pairs = {"ps":7,"pt":1,"k":10,"ws_r":1,"k_s":10,
             "ws":21,"wt":0,"exact":False,"rbwd":True,
             "nftrs_per_head":-1,"nchnls":-1,
             "nheads":1,"stride0":4,"stride1":1,
             "reflect_bounds":True,"use_k":True,"use_adj":True,
             "search_abs":False,"anchor_self":False,
             "dist_type":"l2","search_name":"nl","use_flow":True,
             "dilation":1,"stride0_a":8,"use_state_update":False}
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
