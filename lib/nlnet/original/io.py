
# -- helpers --
import copy
dcopy = copy.deepcopy
import numpy as np
import torch as th
from pathlib import Path
from functools import partial
from easydict import EasyDict as edict

# -- searching --
import stnls

# -- network --
from . import menu
from .net import SrNet
from .scaling import Downsample,Upsample # defaults

# -- search/normalize/aggregate --
import stnls

# -- io --
# from ..utils import model_io
from dev_basics import arch_io

# -- configs --
from dev_basics.configs import ExtractConfig,dcat
econfig = ExtractConfig(__file__) # init extraction
extract_config = econfig.extract_config # rename extraction

# -- load the model --
@econfig.set_init
def load_model(cfg):

    # -=-=-=-=-=-=-=-=-=-=-
    #
    #        Config
    #
    # -=-=-=-=-=-=-=-=-=-=-

    # -- init --
    econfig.init(cfg)
    device = econfig.optional(cfg,"device","cuda")

    # -- unpack local vars --
    local_pairs = {"io":io_pairs(),
                   "arch":arch_pairs(),
                   "blocklist":blocklist_pairs()}
    cfgs = econfig.extract_dict_of_pairs(cfg,local_pairs,restrict=True)
    cfg = dcat(cfg,econfig.flatten(cfgs)) # update cfg

    # -- unpack lib dependencies --
    dep_pairs = {"menu":menu.econfig,
                 "attn":stnls.nn.non_local_attn.extract_config,
                 "search":stnls.search.extract_config,
                 "normz":stnls.normz.extract_config,
                 "agg":stnls.agg.extract_config}
    cfgs = dcat(cfgs,econfig.extract_dict_of_econfigs(cfg,dep_pairs))
    cfg = dcat(cfg,econfig.flatten(cfgs))

    # -- specific update --
    cfg.nblocklists = 2*(len(cfg.arch_depth)-1)+1
    update_archs(cfgs.arch,cfg.search_menu_name,cfg.nblocklists//2)
    update_archs(cfg,cfg.search_menu_name,cfg.nblocklists//2)

    # -- end init --
    if econfig.is_init: return

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    #
    #     Construct Network Configs
    #
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    # -- fill blocks with menu --
    fill_fields = {"attn":["qk_frac","qkv_ngroups",
                           "inner_mult",
                           "attn_proj_version",
                           "attn_proj_ksize",
                           "attn_proj_stride",
                           "attn_proj_ngroups"],
                   "search":["search_name","use_state_update",
                             "normalize_bwd","k_agg","ps","ws",
                             "stride0","stride1","k","ref_itype","self_action"],
                   "normz":["k_agg"],
                   "agg":["inner_mult"],}
    fields = ["attn","search","normz","agg"]
    menu_blocks = menu.get_blocks(cfg)
    blocks = menu.fill_menu(cfgs,fields,menu_blocks,fill_fields)
    # print([block.search.search_name for block in blocks])
    # block_fields = ["attn","search","normz","agg"]
    # block_cfgs = [cfgs[f] for f in block_fields]
    # blocks_lib.copy_cfgs(block_cfgs,blocks)
    # print(blocks[0].search)
    # print(blocks[0].agg)
    # print(blocks[0].attn)
    # # # print(blocks[0].search['k_agg'])
    # exit()


    # -- expand blocklists --
    # fields = ["blocklist"]
    blocklists = init_blocklists(cfgs.blocklist,cfg.nblocklists)

    # -- fill blocks with blocklists --
    dfill = {"attn":["nheads","embed_dim"],"search":["nheads"],
             "res":["nres_per_block","res_ksize","res_bn",
                    "stg_depth","stg_nheads","stg_ngroups"],
             "agg":["nheads","embed_dim"]}
    fill_blocks(blocks,blocklists,dfill)


    # -- create down/up sample --
    scales = create_scales(blocklists)

    # -- init model --
    model = SrNet(cfgs.arch,cfgs.search,blocklists,scales,blocks)
    # model.spynet.eval()

    # -- load model --
    load_pretrained(model,cfgs.io)

    # -- device --
    # model = model.to(device)

    return model

def load_pretrained(model,cfg):
    if cfg.pretrained_load:
        print("Loading model: ",cfg.pretrained_path)
        arch_io.load_checkpoint(model,cfg.pretrained_path,
                                cfg.pretrained_root,cfg.pretrained_type)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#     Configs for "io"
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def update_archs(arch,search_menu_name,ndepth):
    # -> we must have its own search_cfg --
    arch.use_search_input = "none"
    arch.share_encdec = True
    # arch.share_encdec = False
    # # arch.share_encdec = [False,]*ndepth
    # if search_menu_name == "once_video":
    #     arch.use_search_input = "video"
    #     arch.share_encdec = True#[True,]*ndepth
    # elif search_menu_name == "once_features":
    #     arch.use_search_input = "features"
    #     arch.share_encdec = True#[True,]*len(depths)

def shared_defaults():
    pairs = {"arch_nheads":[1,1,1],
             "arch_depth":[1,1,1]}
             # "arch_nheads":[1,1,1],
             # "arch_depth":[1,1,1]}
    # cfg = econfig.extract_pairs(_cfg,pairs,new=False)
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

def blocklist_pairs():
    defs = shared_defaults()
    info = {"mlp_ratio":4.,"block_version":"v4","append_noise":False,
            "embed_dim":None,"freeze":False,"block_mlp":"mlp","norm_layer":"LayerNorm",
            "num_res":3,"res_ksize":3,"nres_per_block":3,"res_bn":False,
            "stg_depth":2,"stg_nheads":4,"stg_ngroups":1,"up_method":"convT"}
    training = {"drop_rate_mlp":0.,"drop_rate_path":0.1}
    # pairs = info | training | defs
    pairs = {**info, **training, **defs}
    return pairs

def arch_pairs():
    defs = shared_defaults()
    pairs = {"in_chans":3,"dd_in":4,"append_noise":False,
             "dowsample":"Downsample", "upsample":"Upsample",
             "input_proj_depth":1,"input_norm_layer":None,
             "output_proj_depth":1,"drop_rate_pos":0.,
             "attn_timer":False,"use_spynet":True,
             "use_second_order_flows":False,
             "spynet_path":"./weights/spynet/spynet_sintel_final-3d2a1287.pth",
             "use_spynet":False
    }
    return {**pairs, **defs}


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Create Up/Down Scales
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def create_scales(blocklists):
    scales = create_downsample_cfg(blocklists)
    scales += [None] # center
    scales += create_upsample_cfg(blocklists)
    return scales

def create_upsample_cfg(bcfgs):
    cfgs = []
    start = len(bcfgs)//2-1
    for l in range(start,0-1,-1):
        cfg_l = edict()
        cfg_l.in_dim = bcfgs[l+1].embed_dim*bcfgs[l+1].nheads
        if l != start:
            cfg_l.in_dim = 2 * cfg_l.in_dim
        cfg_l.out_dim = bcfgs[l].embed_dim*bcfgs[l].nheads
        cfg_l.up_method = bcfgs[l].up_method
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

# def extract_search_cfg(cfg):
#     cfg = stnls.search.extract_config(cfg)
#     cfg = cfg | {"use_flow":True,"scale":2,
#                  "kr_t":-1,"wr_t":-1,
#                  "kr_s":-1,"wr_s":-1}
#     return cfg


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Search Info for Each Block from Menu
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# def extract_menu_cfg(_cfg,depth):

#     """

#     Extract unique values for each _block_
#     This can get to sizes ~=50
#     So a menu is used to simplify setting each of the 50 parameters.
#     These "fill" the fixed configs above.

#     """

#     cfg = econfig.extract_pairs(_cfg,
#                                 {'search_menu_name':'full',
#                                  "search_v0":"exact",
#                                  "search_v1":"refine"},new=False)
#     return extract_menu_cfg_impl(cfg,depth)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Create the list of blocks from block and blocklist
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def fill_blocks(blocks,blocklists,fill_pydict):
    """

    Expand from a config to a list of configs
    with len(block) == # of blocks in network

    -=-=-=- Logical Abstractions -=-=-=-
    blocklist_0 -> blocklist_1 -> ....
    block_0,block_1,... -> block_0,block_1,... ->
    <----  depth_0 ---->   <---- depth_1 ---->

    -=-=-=- This Output -=-=-=-
    block_0,block_1,......,block_D0+1,block_D0+2,...
    <---- depth_0 -------><------- depth_1 -------->

    """
    start,stop = 0,0
    for blocklist in blocklists:
        start = stop
        stop = start + blocklist.depth
        for b in range(start,stop):
            block = blocks[b]
            for field,fill_fields in fill_pydict.items():
                if not(field in block):
                    block[field] = {}
                for fill_field in fill_fields:
                    write_field = get_write_field(fill_field)
                    if not(write_field in block):
                        block[field][write_field] = {}
                    block[field][write_field] = blocklist[fill_field]

def get_write_field(read_field):
    if "arch_" in read_field:
        return read_field.split("arch_")[1]
    else:
        return read_field

def init_blocklists(cfg,L):
    """

    Expands dicts with field of length 1, 1/2, or Full length
    lists into a list of dicts

    """
    # converts a edict to a list of edicts
    cfgs = []
    keys = list(cfg.keys())
    for l in range(L):
        cfg_l = edict()
        for key in keys:
            write_key = get_write_field(key)
            if isinstance(cfg[key],list):
                mid = L//2
                eq = len(cfg[key]) == L
                eq_h = len(cfg[key]) == (mid+1)
                assert eq or eq_h,"Must be shaped for %s & %d" % (key,L)
                if eq: # index along the list
                    cfg_l[write_key] = cfg[key][l]
                elif eq_h: # reflect list length is half size
                    li = l if l <= mid else ((L-1)-l)
                    cfg_l[write_key] = cfg[key][li]
            else:
                cfg_l[write_key] = cfg[key]
        cfg_l['enc_dec'] = 'enc' if l < L//2 else 'dec'
        cfgs.append(cfg_l)
    return cfgs
