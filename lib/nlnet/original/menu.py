
"""

The number of layers ~= 50
This is too many params to set by hand.
Instead, fix the list of parameters using a menu.

"""

import copy
dcopy = copy.deepcopy
import numpy as np
from easydict import EasyDict as edict

# -- configs --
from dev_basics.configs import ExtractConfig
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction


def fill_menu(_cfgs,fields,menu_cfgs,mfields):
    """

    Fill the input "_cfgs" fields using a menu.

    """

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

@econfig.set_init
def get_blocks(cfg):
    """

    Extract blocks

    """

    # -- config --
    econfig.init(cfg)
    pairs = {"search_menu_name":"full",
             "search_v0":"exact","search_v1":"refine"}
    cfg = econfig.extract_pairs(cfg,pairs)
    # -- finish args --
    if econfig.is_init: return


    # -- init --
    depths = cfg.arch_depth
    nblocks = 2*np.sum(depths[:-1]) + depths[-1]
    print("nblocks: ",nblocks)

    # -- unpack attn name --
    # ...

    # -- unpack search name --
    # "search_vX" in ["exact","refine","approx_t","approx_s","approx_st"]
    search_menu_name = cfg.search_menu_name
    v0,v1 = cfg.search_v0,cfg.search_v1
    search_params = search_menu(depths,search_menu_name,v0,v1)

    # -- unpack normz name --
    # ...

    # -- unpack agg name --
    # ...


    # # -- arch params --
    # arch_params = arch_menu(search_params)

    # -- expand out blocks --
    blocks = []
    params = [search_params]
    for l in range(nblocks):
        block_l = edict()
        for param in params:
            for key,val_list in param.items():
                block_l[key] = val_list[l]
        blocks.append(block_l)

    return blocks

def search_menu(depths,menu_name,v0,v1):

    # -- init --
    params = edict()
    params.search_name = get_search_names(menu_name,depths,v0,v1)
    params.use_state_update = get_use_state_updates(params.search_name)
    return params

def get_use_state_updates(search_names):
    """
    Create derived parameters from parsed parameters

    """
    # -- fill --
    nblocks = len(search_names)
    any_refine = np.any(np.array(search_names)=="refine")
    use_state_updates = []
    for i in range(nblocks):
        use_state_updates.append(any_refine)
    return use_state_updates

def get_search_names(menu_name,depths,v0,v1):
    nblocks = 2*np.sum(depths[:-1]) + depths[-1]

    if menu_name == "full":
        return [v0,]*nblocks
    elif menu_name == "one":
        return [v0,] + [v1,]*(nblocks-1)
    elif menu_name == "once_video":
        return [v1,]*(nblocks)
    elif menu_name == "once_features":
        return [v1,]*(nblocks)
    elif menu_name == "first":
        names = []
        for depths_i in depths:
            names_i = [v0,] + [v1,]*(depths_i-1)
            names.extend(names_i)
        for depths_i in reversed(depths[:-1]):
            names_i = [v1,] + [v1,]*(depths_i-1)
            names.extend(names_i)
        return names
    elif menu_name == "first_each": # share_encdec = False
        names = []
        for depths_i in depths:
            names_i = [v0,] + [v1,]*(depths_i-1)
            names.extend(names_i)
        for depths_i in reversed(depths[:-1]):
            names_i = [v0,] + [v1,]*(depths_i-1)
            names.extend(names_i)
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
        raise ValueError("Uknown search type in menu [%s]" % menu_name)

