#
#
#   API to access the Aggregation Methods
#
#


# -- local modules --
from . import wpsum

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
def init_agg(cfg):

    # -- unpack --
    cfgs = econfig({"agg":agg_pairs()})
    if econfig.is_init == True: return
    cfg = cfgs.agg

    # -- menu --
    modules = {"wpsum":wpsum}

    # -- init --
    mod = modules[cfg.agg_name]
    fxn = getattr(mod,'init')(cfg)

    # -- return --
    return fxn

def init(cfg):
    return init_agg(cfg)

def agg_pairs():
    pairs = {"ps":7,"pt":1,"dilation":1,
             "exact":False,"reflect_bounds":True,
             "agg_name":"nl","k_a":10,"agg_name":"wpsum"}
    return pairs
