
# -- helpers --
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- network --
from .net import SrNet

# -- extract config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
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
    arch_cfg = extract_arch_config(cfg,optional)
    search_cfg = extract_search_config(cfg,optional)
    io_cfg = extract_io_config(cfg,optional)
    if init: return

    # -- init model --
    model = SrNet(arch_cfg,search_cfg)

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

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def extract_io_config(_cfg,optional):
    sigma = optional(_cfg,"sigma",0.)
    base = Path("weights/checkpoints/")
    pretrained_path = base / "model/model_best.pt"
    pairs = {"pretrained_load":True,
             "pretrained_path":str(pretrained_path),
             "pretrained_type":"lit",
             "pretrained_root":"."}
    return extract_pairs(pairs,_cfg,optional)

def extract_search_config(_cfg,optional):
    pairs = {"attn_mode":"dnls_k",
             "k_s":100,"k_a":100,
             "ws":21,"ws_r":3,
             "ps":7,"pt":1,"wt":0,
             "stride0":4,"stride1":1,"bs":-1,
             "rbwd":False,"nbwd":1,"exact":False,
             "reflect_bounds":False,
             "refine_inds":[False,False,False],
             "dilation":1,"return_inds":False,
             "softmax_scale":10,
             "attn_timer":False}
    return extract_pairs(pairs,_cfg,optional)

def extract_arch_config(_cfg,optional):
    pairs = {"scale":[1],"self_ensemble":False,
             "chop":False,"precision":"single",
             "cpu":False,"n_GPUs":1,"pre_train":".",
             "save_models":False,"model":"COLA","mode":"E",
             "print_model":False,"resume":0,"seed":1,
             "n_resblock":16,"n_feats":64,"n_colors":1,
             "res_scale":1,"rgb_range":1.,"stages":6,
             "blocks":3,"act":"relu","sigma":0.,
             "arch_return_inds":False,"device":"cuda:0",
             "attn_timer":False}
    return extract_pairs(pairs,_cfg,optional)

# -- run to populate "_fields" --
load_model(edict({"__init":True}))
