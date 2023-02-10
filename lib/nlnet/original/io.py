
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


# -- search/normalize/aggregate --
from .. import search
from .. import normz
from .. import agg

# -- io --
# from ..utils import model_io
from dev_basics import arch_io

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init static variable
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -- config --
    econfig.set_cfg(cfg)
    defs = shared_defaults(cfg)
    menu_cfgs = extract_menu_cfg(cfg,defs.depth)
    pairs = {"io":io_pairs(),
             "arch":arch_pairs(defs),
             "blocklist":blocklist_pairs({}),
             "attn":attn_pairs(defs),
             "search":search.extract_config(cfg),
             "normz":normz.extract_config(cfg),
             "agg":agg.extract_config(cfg)}
    device = econfig.optional(cfg,"device","cuda:0")
    cfgs = econfig.extract_set(pairs)
    if econfig.is_init: return

    # -- unpack --
    nblocklists = defs.nblocklists

    # -- expand blocklists --
    fields = ["blocklist"]
    blocklists = econfig.cfgs2lists(cfgs.blocklist,defs.nblocklists)

    # -- fill blocks with menu --
    fields = ["attn","search","normz","agg"]
    blocks = fill_menu(cfgs,fields,menu_cfgs)

    # -- fill blocks with blocklists --
    dfill = {"attn":["nheads","embed_dim"],"search":["nheads"]}
    fill_blocks(blocks,blocklists,dfill)

    # -- create down/up sample --
    scales = create_downsample_cfg(blocklists)
    scales += [None] # conv
    scales += create_upsample_cfg(blocklists)

    # -- init model --
    model = SrNet(cfgs.arch,blocklists,scales,blocks)

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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def shared_defaults(_cfg):
    pairs = {"embed_dim":1,
             "nheads":[1,1,1],
             "depth":[1,1,1],
             "nblocklists":5,"nblocks":5}
    cfg = econfig.extract_pairs(pairs,_cfg)
    cfg.nblocklists = 2*(len(cfg.depth)-1)+1
    cfg.nblocks = 2*np.sum(cfg.depth[:-1]) * cfg.depth[-1]
    # _cfg.nblocklists = 2*(len(cfg.depth)-1)+1
    # _cfg.nblocks = 2*np.sum(cfg.depth[:-1]) * cfg.depth[-1]
    return cfg

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
    shape = {"depth":None,"nheads":None,
             "nblocklists":None,"freeze":False,
             "block_version":"v2"}
    training = {"mlp_ratio":4.,"embed_dim":1,"num_res":0,"res_ksize":3,
                "block_mlp":"mlp","norm_layer":"LayerNorm",
                "drop_rate_mlp":0.,"drop_rate_path":0.1}
    pairs = shape | training | defs

    return pairs

def arch_pairs(defs):
    pairs = {"in_chans":3,"dd_in":3,
             "dowsample":"Downsample", "upsample":"Upsample",
             "embed_dim":None,"input_proj_depth":1,
             "output_proj_depth":1,"drop_rate_pos":0.,
             "attn_timer":False,
    }
    return pairs  | defs

# def extract_search_cfg(cfg):
#     cfg = dnls.search.extract_config(cfg)
#     cfg = cfg | {"use_flow":True,"scale":2,
#                  "kr_t":-1,"wr_t":-1,
#                  "kr_s":-1,"wr_s":-1}
#     return cfg

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

def fill_blocks(blocks,blocklists,fill_pydict):
    start,stop = 0,0
    for blocklist in blocklists:
        start = stop
        stop = start + blocklist.depth
        for b in range(start,stop):
            block = blocks[b]
            for field,fill_fields in fill_pydict.items():
                for fill_field in fill_fields:
                    block[field][fill_field] = blocklist[fill_field]

