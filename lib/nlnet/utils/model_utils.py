
from easydict import EasyDict as edict

def apply_freeze(model,freeze):
    if freeze is False: return
    unset_names = []
    for name,param in model.named_parameters():
        # print(name)
        bname = name.split(".")[0]
        bnum = block_name2num(bname)
        if bnum == -1: unset_names.append(name)
        freeze_b = freeze[bnum]
        if freeze_b is True:
            param.requires_grad_(False)

def cfgs_slice(cfgs,start,stop,trans=False):
    _cfgs = edict()
    for key,cfg_list in cfgs.items():
        _cfgs[key] = [cfg_list[i] for i in range(start,stop)]
    return cfgs

