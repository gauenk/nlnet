
# -- helpers --
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- network --
from .net import BiDirNet
from .scaling import Downsample,Upsample # defaults

# -- io --
from ..utils import model_io

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields
from dev_basics.common import set_defaults
from dev_basics.common import extract_config,extract_pairs,cfg2lists
_fields = [] # fields for model io; populated using this code section
optional_full = partial(optional_fields,_fields)
extract_model_config = partial(extract_config,_fields) # all the aggregate fields

# -- load the model --
def load_model(cfg):

    # -- allows for all keys to be aggregated at init --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)

    # -- config --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    device = optional(cfg,'device','cuda:0')
    io_cfg = extract_io_config(cfg,optional)
    arch_cfg = extract_arch_config(cfg,optional)
    block_cfg = extract_block_config(cfg,optional)
    attn_cfg = extract_attn_config(cfg,optional,len(block_cfg))
    search_cfg = extract_search_config(cfg,optional,len(block_cfg))
    if init: return

    # -- create up/down sample --
    up_cfg = create_upsample_cfg(block_cfg)
    down_cfg = create_downsample_cfg(block_cfg)

    # -- init model --
    model = SrNet(arch_cfg,block_cfg,attn_cfg,search_cfg,up_cfg,down_cfg)

    # -- load model --
    load_pretrained(model,io_cfg)

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

def get_defs():

    # -- shared default values --
    defs = edict()
    defs.embed_dim = 9

    # -- depth == 5 --
    defs.nheads = [1,2,4]
    defs.depth = [2,2,2]
    defs.nblocks = 5

    # -- depth == 7 --
    # defs.nheads = [1,2,4,8]
    # defs.depth = [2,2,2,2]
    # defs.nblocks = 7

    # -- depth == 9 --
    # defs.nheads = [1,2,4,8,16]
    # defs.depth = [2,2,2,2,2]
    # defs.nblocks = 7

    return defs

def extract_io_config(_cfg,optional):
    sigma = optional(_cfg,"sigma",0.)
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return extract_pairs(pairs,_cfg,optional)

def extract_search_config(_cfg,optional,nblocks):
    pairs = {"k_s":100,"k_a":100,
             "ws":21,"ws_r":3,
             "ps":7,"pt":1,"wt":0,"dil":1,
             "stride0":4,"stride1":1,"bs":-1,
             "rbwd":False,"nbwd":1,"exact":False,
             "reflect_bounds":False,
             "refine_inds":False,
             "dilation":1,"return_inds":False,
             "search_type":"stnls_prod","nheads":None,
             "use_flow":True,
    }

    # -- set shared defaults --
    defs = get_defs()
    set_defaults(defs,pairs)

    # -- extract --
    return cfg2lists(extract_pairs(pairs,_cfg,optional),nblocks)

def extract_attn_config(_cfg,optional,nblocks):
    pairs = {"qk_frac":1.,"qkv_bias":True,"scale":10.,
             "token_mlp":'leff',"embed_dim":None,"attn_mode":"default",
             "nheads":None,"token_projection":'linear',
             "drop_rate_attn":0.,"drop_rate_proj":0.}

    # -- set shared defaults --
    defs = get_defs()
    set_defaults(defs,pairs)

    # -- extract --
    return cfg2lists(extract_pairs(pairs,_cfg,optional),nblocks)

def extract_block_config(_cfg,optional):
    shape = {"depth":None,
             "nheads":None,
             "nblocks":None,"freeze":False}
    training = {"mlp_ratio":4.,"embed_dim":None,
                "block_mlp":"mlp","norm_layer":"LayerNorm",
                "drop_rate_mlp":0.,"drop_rate_path":0.1}
    pairs = shape | training

    # -- set shared defaults --
    defs = get_defs()
    set_defaults(defs,pairs)

    # -- extract --
    return cfg2lists(extract_pairs(pairs,_cfg,optional),pairs['nblocks'])

def extract_arch_config(_cfg,optional):
    pairs = {"in_chans":3,"dd_in":3,
             "dowsample":Downsample, "upsample":Upsample,
             "embed_dim":None,
             "input_proj_depth":1,
             "output_proj_depth":1,"drop_rate_pos":0.}
    # -- set shared defaults --
    defs = get_defs()
    set_defaults(defs,pairs)

    # -- extract --
    return extract_pairs(pairs,_cfg,optional)

# -- run to populate "_fields" --
load_model(edict({"__init":True}))
