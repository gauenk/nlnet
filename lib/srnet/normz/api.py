#
#
#   API to access the Normalization Methods
#
#


# -- local modules --
from . import softmax

# -- config extraction --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,set_defaults
from dev_basics.common import extract_config,extract_pairs,cfg2lists
_fields = [] # fields for model io; populated using this code section
optional_full = partial(optional_fields,_fields)
extract_agg_config = partial(extract_config,_fields) # all the aggregate fields


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#       Create the initial search function
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def init_normz(cfg):

    # -- unpack --
    __init = _optional(cfg,"__init",False)
    cfg = extract_agg_config(cfg,optional)
    if __init == True: return

    # -- menu --
    modules = {"softmax":softmax}

    # -- init --
    mod = modules[cfg.agg_name]
    fxn = getattr(mod,'init')(cfg)

    # -- return --
    return fxn

def init(cfg):
    return init_normz(cfg)

def extract_normz_config(cfg,optional):
    pairs = {"scale":10,
             "normz_name":"softmax",
             "k_n":100,
             "normz_drop_rate":0.}
    return extract_pairs(pairs,cfg,optional)

init({"__init"})
