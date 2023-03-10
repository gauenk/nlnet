
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- searching --
import dnls

# -- network --
from .net import SrNet
from .scaling import Downsample,Upsample # defaults
# from .menu import extract_menu_cfg,fill_menu
from .menu import extract_menu_cfg_impl,fill_menu


# -- dev basics --
from dev_basics.unet_arch import io as unet_io

# -- search/normalize/aggregate --
from .. import search
from .. import normz
from .. import agg

# -- io --
# from ..utils import model_io
from dev_basics import arch_io

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -- config --
    econfig.init(cfg)
    defs = econfig.extract_pairs(cfg,shared_pairs(),new=True)
    defs.nblocklists = 2*(len(defs.depth)-1)+1
    defs.nblocks = 2*np.sum(defs.depth[:-1]) * defs.depth[-1]
    device = econfig.optional(cfg,"device","cuda:0")
    pairs = {"io":io_pairs(),
             "arch":arch_pairs(defs),
             "blocklist":blocklist_pairs(defs),
             "attn":attn_pairs(defs)}
    cfgs = econfig.extract_set(pairs,new=True)
    econfigs = {"search":search.econfig,
                "normz":normz.econfig,
                "agg":agg.econfig}
    cfgs = cfgs | econfig.optional_config_dict(cfg,econfigs,new=True)
    cfgs = edict(cfgs)
    if econfig.is_init: return

    # -- fill blocks with menu --
    dfill = {"attn":["nheads","embed_dim"],
             "search":["nheads"],
             "res":["nres_per_block","res_ksize"]}
    fill_cfgs = {k:cfgs[k] for k in ["attn","search","normz","agg"]}
    blocks,blocklists = unet_io.load_arch(cfg,fill_cfgs,dfill)

    # -- view --
    scales = [{"in_dim":blocklist.in_dim,"out_dim":blocklist.out_dim} \
              for blocklist in blocklists]

    # -- init model --
    model = SrNet(cfgs.arch,blocklists,blocks)
    print(blocklists[0])

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    model = model.to(device)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def shared_pairs():
    pairs = {"embed_dim":1,
             "nheads":[1,1,1],
             "depth":[1,1,1]}
    # cfg = econfig.extract_pairs(pairs,_cfg)
    # cfg.nblocklists = 2*(len(cfg.depth)-1)+1
    # cfg.nblocks = 2*np.sum(cfg.depth[:-1]) * cfg.depth[-1]
    return pairs

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs

def attn_pairs(defs):
    pairs = {"qk_frac":1.,"qkv_bias":True,
             "token_mlp":'leff',"attn_mode":"default",
             "token_projection":'linear',
             "drop_rate_proj":0.,"attn_timer":False}
    return pairs

def blocklist_pairs(defs):
    # shape = {"depth":None,"nheads":None,
    #          "nblocklists":None,"freeze":False,
    #          "block_version":"v3"}
    info = {"mlp_ratio":4.,"embed_dim":1,"block_version":"v3",
            "freeze":False,"block_mlp":"mlp","norm_layer":"LayerNorm",
            "num_res":3,"res_ksize":3,"nres_per_block":3,}
    training = {"drop_rate_mlp":0.,"drop_rate_path":0.1}
    pairs = info | training | defs

    return pairs

def arch_pairs(defs):
    pairs = {"in_chans":3,"dd_in":3,
             "dowsample":"Downsample", "upsample":"Upsample",
             "embed_dim":None,"input_proj_depth":1,
             "output_proj_depth":1,"drop_rate_pos":0.,
             "attn_timer":False,
    }
    return pairs  | defs


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Create Up/Down Scales
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def create_scales(blocklists):
    scales = create_downsample_cfg(blocklists)
    scales += [None] # center
    scales += create_upsample_cfg(blocklists)
    return scales

def create_upsample_cfg(bcfgs):
    cfgs = []
    start = len(bcfgs)//2-1
    for l in range(start,0-1,-1):
        cfg_l = edict()
        cfg_l.in_dim = bcfgs[l+1].embed_dim*bcfgs[l+1].nheads
        if l != start:
            cfg_l.in_dim = 2 * cfg_l.in_dim
        cfg_l.out_dim = bcfgs[l].embed_dim*bcfgs[l].nheads
        cfgs.append(cfg_l)
    return cfgs

def create_downsample_cfg(bcfgs):
    cfgs = []
    nencs = len(bcfgs)//2
    for l in range(nencs):
        cfg_l = edict()
        cfg_l.in_dim = bcfgs[l].embed_dim*bcfgs[l].nheads
        cfg_l.out_dim = bcfgs[l+1].embed_dim*bcfgs[l+1].nheads
        cfgs.append(cfg_l)
    return cfgs

# def extract_search_cfg(cfg):
#     cfg = dnls.search.extract_config(cfg)
#     cfg = cfg | {"use_flow":True,"scale":2,
#                  "kr_t":-1,"wr_t":-1,
#                  "kr_s":-1,"wr_s":-1}
#     return cfg


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Search Info for Each Block from Menu
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_menu_cfg(_cfg,depth):

    """

    Extract unique values for each _block_
    This can get to sizes ~=50
    So a menu is used to simplify setting each of the 50 parameters.
    These "fill" the fixed configs above.

    """

    cfg = econfig.extract_pairs({'search_menu_name':'full',
                                 "search_v0":"exact",
                                 "search_v1":"refine"},_cfg)
    return extract_menu_cfg_impl(cfg,depth)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Create the list of blocks from block and blocklist
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def fill_blocks(blocks,blocklists,fill_pydict):
    """

    Expand from a config to a list of configs
    with len(block) == # of blocks in network

    -=-=-=- Logical Abstractions -=-=-=-
    blocklist_0 -> blocklist_1 -> ....
    block_0,block_1,... -> block_0,block_1,... ->
    <----  depth_0 ---->   <---- depth_1 ---->

    -=-=-=- This Output -=-=-=-
    block_0,block_1,......,block_D0+1,block_D0+2,...
    <---- depth_0 -------><------- depth_1 -------->

    """
    start,stop = 0,0
    for blocklist in blocklists:
        start = stop
        stop = start + blocklist.depth
        for b in range(start,stop):
            block = blocks[b]
            for field,fill_fields in fill_pydict.items():
                if not(field in block):
                    block[field] = {}
                for fill_field in fill_fields:
                    if not(fill_field in block):
                        block[field][fill_field] = {}
                    block[field][fill_field] = blocklist[fill_field]

def init_blocklists(cfg,L):
    """

    Expands dicts with field of length 1, 1/2, or Full length
    lists into a list of dicts

    """
    # converts a edict to a list of edicts
    cfgs = []
    keys = list(cfg.keys())
    for l in range(L):
        cfg_l = edict()
        for key in keys:
            if isinstance(cfg[key],list):
                mid = L//2
                eq = len(cfg[key]) == L
                eq_h = len(cfg[key]) == (mid+1)
                assert eq or eq_h,"Must be shaped for %s & %d" % (key,L)
                if eq: # index along the list
                    cfg_l[key] = cfg[key][l]
                elif eq_h: # reflect list length is half size
                    li = l if l <= mid else ((L-1)-l)
                    cfg_l[key] = cfg[key][li]
            else:
                cfg_l[key] = cfg[key]
        cfgs.append(cfg_l)
    return cfgs
