import nlnet
import torch as th
from easydict import EasyDict as edict
import copy
dcopy = copy.deepcopy

cfg = edict()
cfg.stride0 = 4
cfg.ps = 7
cfg.k = 10
cfg.embed_dim = 9
cfg.ws = 5
cfg.wt = 0
cfg.drop_rate_path = 0


models = edict()
models["0"] = nlnet.load_model(cfg).to("cuda:0")
models["1"] = nlnet.version2.load_model(cfg).to("cuda:0")
models["0"].load_state_dict(models["1"].state_dict())
img = th.randn(1,3,3,256,256).to("cuda:0")
clean = th.randn(1,3,3,256,256).to("cuda:0")

flows = edict()
flows.fflow = th.randn(1,3,2,256,256).to("cuda:0")
flows.bflow = th.randn(1,3,2,256,256).to("cuda:0")

noisy = edict()
noisy["0"] = img.clone().requires_grad_(True)
noisy["1"] = img.clone().requires_grad_(True)

deno = edict()
for i in range(2):
    print(i)
    deno[str(i)] = models[str(i)](noisy[str(i)],flows)

# -- compare fwd --
diff = th.mean((deno["0"] - deno["1"])**2)
print("Fwd: ",diff)

# -- loss --
for i in range(2):
    loss = th.mean((clean - deno[str(i)])**2)
    loss.backward()

# -- compare bwd --
print(noisy["0"].grad.shape)
diff = th.mean((noisy["0"].grad - noisy["1"].grad)**2)
print("Bwd: ",diff)

# -=-=-=-=-=-=-=-=-=-=-=-=-
#
#     Compare Configs
#
# -=-=-=-=-=-=-=-=-=-=-=-=-



cfg0 = nlnet.original.extract_config(dcopy(cfg))
cfg1 = nlnet.version2.extract_config(dcopy(cfg))
print(cfg0)
print(cfg1)

print("Missing keys in cfg1: ")
for key0 in cfg0:
    if not(key0 in cfg1):
        print(key0)

print("Missing keys in cfg0: ")
for key1 in cfg1:
    if not(key1 in cfg0):
        print(key1)

