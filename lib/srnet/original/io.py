
# -- helpers --
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- network --
from .net import SrNet
from .scaling import Downsample,Upsample # defaults

# -- search/agg --
import ..search
import ..normz
import ..agg

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
    pairs = {"io":io_pairs(),
             "search":search.extract_config(cfg),
             "normz":normz.extract_config(cfg),
             "agg":normz.extract_config(cfg),
             "block":block_pairs(),
             "arch":arch_pairs()}
    device = econfig.optional("device","cuda:0")
    cfgs = econfig.extract(pairs)
    if econfig.is_init: return

    # -- list of configs --
    fields = ["attn","search","normz","agg","block"]
    nblocks = cfgs.arch.nblocks
    econfig.cfgs_to_lists(cfgs,fields,nblocks)

    # -- create up/down sample --
    cfgs.up = create_upsample_cfg(cfgs.block)
    cfgs.down = create_downsample_cfg(cfgs.block)

    # -- init model --
    model = SrNet(cfgs.arch,cfgs.block,cfgs.attn,cfgs.search,
                  cfgs.normz,cfgs.agg,cfgs.up,cfgs.down)

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

def io_pairs():
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":False,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return pairs

def attn_pairs():
    pairs = {"qk_frac":1.,"qkv_bias":True,
             "token_mlp":'leff',"embed_dim":None,
             "attn_mode":"default","nheads":None,
             "token_projection":'linear',
             "drop_rate_proj":0.}
    return pairs

def block_pairs():
    shape = {"depth":None,
             "nheads":None,
             "nblocks":None,"freeze":False}
    training = {"mlp_ratio":4.,"embed_dim":None,
                "block_mlp":"mlp","norm_layer":"LayerNorm",
                "drop_rate_mlp":0.,"drop_rate_path":0.1}
    pairs = shape | training

    return pairs

def arch_pairs():
    pairs = {"in_chans":3,"dd_in":3,
             "dowsample":Downsample, "upsample":Upsample,
             "embed_dim":None,
             "input_proj_depth":1,
             "output_proj_depth":1,"drop_rate_pos":0.}
    return pairs

