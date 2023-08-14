
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
             "search_v0":"exact","search_v1":"refine",
             "k_agg":10,"ps":3,"ws":7,"qk_frac":1.,"qkv_ngroups":1,
             "inner_mult":2,"nls_normalize_bwd":False,
             "attn_proj_version":"v1",
             "attn_proj_ksize":"",
             "attn_proj_stride":"",
             "attn_proj_ngroups":"nheads",
    }
    cfg = econfig.extract_pairs(cfg,pairs)
    # -- finish args --
    if econfig.is_init: return


    # -- init --
    depths = cfg.arch_depth
    nblocks = 2*np.sum(depths[:-1]) + depths[-1]
    # print("nblocks: ",nblocks)

    # -- unpack attn name --
    # ...
    names = ["qk_frac","qkv_ngroups","inner_mult",
             "attn_proj_version","attn_proj_ksize",
             "attn_proj_stride","attn_proj_ngroups"]
    lists = [cfg[name] for name in names]
    attn_params = unpack_params(depths,lists,names)
    # attn_params = get_attn_params(depths,cfg.qk_frac,
    #                               cfg.qkv_ngroups,cfg.inner_mult,
    #                               cfg.attn_proj_version,
    #                               cfg.attn_proj_ksize,
    #                               cfg.attn_proj_stride,
    #                               cfg.attn_proj_ngroups)

    # -- unpack search name --
    # "search_vX" in ["exact","refine","approx_t","approx_s","approx_st"]
    search_menu_name = cfg.search_menu_name
    v0,v1 = cfg.search_v0,cfg.search_v1
    normalize_bwd = cfg.nls_normalize_bwd
    search_params = search_menu(depths,search_menu_name,v0,v1,normalize_bwd)
    names = ["k_agg","ps","ws","stride0"]
    lists = [cfg[name] for name in names]
    search_params_u = unpack_params(depths,lists,names)
    for key in search_params_u:
        search_params[key] = search_params_u[key]


    # -- unpack normz name --
    names = ["k_agg"]
    lists = [cfg[name] for name in names]
    normz_params = unpack_params(depths,lists,names)

    # -- unpack agg name --
    # ...


    # # -- arch params --
    # arch_params = arch_menu(search_params)

    # -- expand out blocks --
    blocks = []
    params = [attn_params,search_params,normz_params]
    # params = [search_params]
    for l in range(nblocks):
        block_l = edict()
        for param in params:
            for key,val_list in param.items():
                block_l[key] = val_list[l]
        blocks.append(block_l)

    return blocks

def unpack_params(depths,lists,names):

    # -- init --
    params = edict()
    for name in names:
        params[name] = []
    

    # -- helper --
    def get_val(val,d,depths_i):
        if isinstance(val,list): val_d = val[d]
        else: val_d = val
        return [val_d,]*depths_i

    # -- downscale --
    for d,depths_i in enumerate(depths):
        for name,alist in zip(names,lists):
            params[name].extend(get_val(alist,d,depths_i))

    # -- upscale --
    for d,depths_i in reversed(list(enumerate(depths[:-1]))):
        for name,alist in zip(names,lists):
            params[name].extend(get_val(alist,d,depths_i))

    return params

def get_attn_params(depths,qk_fracs,qkv_ngroups,inner_mults,
                    attn_proj_versions,attn_proj_ksizes,
                    attn_proj_strides,attn_proj_ngroups):

    # -- init --
    params = edict()
    params.qk_frac = []
    params.qkv_ngroups = []
    params.inner_mult = []
    params.attn_proj_version = []
    params.attn_proj_ksize = []
    params.attn_proj_stride = []
    params.attn_proj_ngroup = []
    # params.embed_dim = []

    # -- helper --
    def get_val(val,d,depths_i):
        if isinstance(val,list): val_d = val[d]
        else: val_d = val
        return [val_d,]*depths_i

    # -- downscale --
    for d,depths_i in enumerate(depths):
        params.qk_frac.extend(get_val(qk_fracs,d,depths_i))
        params.qkv_ngroups.extend(get_val(qkv_ngroups,d,depths_i))
        params.inner_mult.extend(get_val(inner_mults,d,depths_i))
        params.attn_proj_version.extend(get_val(attn_proj_versions,d,depths_i))
        params.attn_proj_ksize.extend(get_val(attn_proj_ksizes,d,depths_i))
        params.attn_proj_stride.extend(get_val(attn_proj_strides,d,depths_i))
        params.attn_proj_ngroup.extend(get_val(attn_proj_ngroups,d,depths_i))
        # params.embed_dim.extend(get_val(embed_dims,d,depths_i))

    # -- upscale --
    for d,depths_i in reversed(list(enumerate(depths[:-1]))):
        params.qk_frac.extend(get_val(qk_fracs,d,depths_i))
        params.qkv_ngroups.extend(get_val(qkv_ngroups,d,depths_i))
        params.inner_mult.extend(get_val(inner_mults,d,depths_i))
        params.attn_proj_version.extend(get_val(attn_proj_versions,d,depths_i))
        params.attn_proj_ksize.extend(get_val(attn_proj_ksizes,d,depths_i))
        params.attn_proj_stride.extend(get_val(attn_proj_strides,d,depths_i))
        params.attn_proj_ngroup.extend(get_val(attn_proj_ngroups,d,depths_i))
        # params.embed_dim.extend(get_val(embed_dims,d,depths_i))

    # print(qkv_ngroups)
    # print(params.qk_frac)
    # print(params.qkv_ngroups)

    return params

def search_menu(depths,menu_name,v0,v1,nls_normalize_bwd):

    # -- init --
    params = edict()
    params.search_name = get_search_names(menu_name,depths,v0,v1)
    params.use_state_update = get_use_state_updates(params.search_name)
    L = len(params.use_state_update)
    params.normalize_bwd = [nls_normalize_bwd,]*L
    # names = ["qk_frac","qkv_ngroups","inner_mult",
    #          "attn_proj_version","attn_proj_ksize",
    #          "attn_proj_stride","attn_proj_ngroups"]
    # lists = [cfg[name] for name in names]
    # unpack_params(depths,lists,names)

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

