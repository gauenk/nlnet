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
