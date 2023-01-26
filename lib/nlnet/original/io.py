
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- network --
from .net import SrNet
from .scaling import Downsample,Upsample # defaults

# -- search/normalize/aggregate --
from .. import search
from .. import normz
from .. import agg

# -- io --
from ..utils import model_io

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

    # -- fill with menu --
    fields = ["attn","search","normz","agg"]
    blocks = fill_menu(cfgs,fields,menu_cfgs)
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
        model_io.load_checkpoint(model,cfg.pretrained_path,
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
    _cfg.nblocklists = 2*(len(cfg.depth)-1)+1
    _cfg.nblocks = 2*np.sum(cfg.depth[:-1]) * cfg.depth[-1]
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
             "nblocklists":None,"freeze":False}
    training = {"mlp_ratio":4.,"embed_dim":None,
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

    # -- unpack attn name --
    # ...

    # -- unpack search name --
    # "search_vX" in ["exact","refine","approx_t","approx_s","approx_st"]
    search_menu_name = cfg.search_menu_name
    v0,v1 = cfg.search_v0,cfg.search_v0
    search_names = search_menu(search_menu_name,depth,v0,v1)

    # -- unpack normz name --
    # ...

    # -- unpack agg name --
    # ...

    # -- grouped pairs --
    pairs = {"search_name":search_names}
    L = len(search_names)

    # -- format return value; a list of pydicts --
    blocks = []
    for l in range(L):
        block_l = edict()
        for key,val_list in pairs.items():
            block_l[key] = val_list[l]
        blocks.append(block_l)

    return blocks

def search_menu(menu_name,depth,v0,v1):
    nblocks = 2*np.sum(depth[:-1]) + depth[-1]

    if menu_name == "full":
        return [v0,]*nblocks
    elif menu_name == "one":
        return [v0,] + [v1,]*(nblocks-1)
    elif menu_name == "first":
        names = []
        for depth_i in depth:
            names_i = [v0,] + [v1,]*(depth_i-1)
            names.append(names_i)
        return names
    elif menu_name == "nth":
        names = []
        for i in range(nblocks):
            if (i % menu_n == 0) or i == 0:
                names.append(v0)
            else:
                names.append(v1)
        return names
    else:
        raise ValeError("Uknown search type in menu [%s]" % menu_name)

def fill_menu(_cfgs,fields,menu_cfgs):

    # -- fields to fill --
    mfields = {"attn":[],"search":["search_name"],"normz":[],"agg":[],}

    # -- filling --
    cfgs = []
    for menu_cfg in menu_cfgs:
        cfgs_m = edict()
        for field in fields:
            cfg_f = dcopy(_cfgs[field])
            for fill_key in mfields[field]:
                cfg_f[fill_key] = menu_cfg[fill_key]
            cfgs_m[field] = cfg_f
        cfgs.append(cfgs_m)
    return cfgs

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
