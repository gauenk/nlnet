import torch as th
from pathlib import Path

def remove_lightning_load_state(state):
    print("Please remove me.")
    names = list(state.keys())
    for name in names:
        name_og = name.split(".")[0]
        if name_og == "sim_model":
            del state[name]
            continue
        name_new = name.split(".")[1:]
        name_new = ".".join(name_new)
        state[name_new] = state[name]
        del state[name]

def resolve_path(path,root):
    print("Please remove me.")
    if not Path(path).exists():
        path_ = Path(root) / Path(path)
        if not(path_.exists()):
            path_ = Path(root) / "output/checkpoints/" / Path(path)
        path = path_
    assert Path(path).exists()
    return str(path)

def load_checkpoint(model, path, root, wtype="git"):
    print("Please remove me.")
    full_path = resolve_path(path,root)
    if wtype in ["git","original"]:
        load_checkpoint_git(model,full_path)
    elif wtype in ["lightning","lit"]:
        load_checkpoint_lit(model,full_path)
    elif "b2c" in wtype: # b2cg = git or b2cl = lit
        load_checkpoint_b2c(model,full_path,wtype)
    else:
        raise ValueError(f"Uknown checkpoint weight type [{wtype}]")

def load_checkpoint_lit(model,path):
    print("Please remove me.")
    state = read_checkpoint_lit(path)
    model.load_state_dict(state)

def load_checkpoint_git(model,path):
    print("Please remove me.")
    # -- filename --
    state = read_checkpoint_git(path)
    model.load_state_dict(state)

def read_checkpoint_lit(path):
    print("Please remove me.")
    weights = th.load(path)
    state = weights['state_dict']
    remove_lightning_load_state(state)
    return state

def read_checkpoint_git(path):
    print("Please remove me.")
    state = th.load(path)
    return state

def read_b2c(path,wtype):
    print("Please remove me.")
    # -- read original weights --
    if wtype[-1] == "g":
        state = read_checkpoint_git(path)
    elif wtype[-1] == "l":
        state = read_checkpoint_lit(path)
    else: # default == "lit"
        state = read_checkpoint_lit(path)
    return state

def load_checkpoint_b2c(model,path,wtype):
    print("Please remove me.")

    # -- read saved --
    state = read_b2c(path,wtype)
    print(list(state.keys()))

