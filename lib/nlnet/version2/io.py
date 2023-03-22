
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
# from .menu import extract_menu_cfg_impl,fill_menu


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
    cfg_unet_arch = econfig.optional_config(cfg,unet_io.econfig,new=True)
    if econfig.is_init: return

    # -- fill blocks with menu --
    dfill = {"attn":["nheads","embed_dim"],
             "search":["nheads"],
             "res":["nres_per_block","res_ksize"]}
    fill_cfgs = {k:cfgs[k] for k in ["attn","search","normz","agg"]}
    print(cfg_unet_arch)
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

