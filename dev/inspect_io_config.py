import nlnet
import torch as th
from easydict import EasyDict as edict
from dev_basics.utils.timer import TimeIt,ExpTimer
from dev_basics.utils.gpu_mem import MemIt,GpuMemer


cfg = nlnet.extract_config({})
# print(cfg)
cfg.embed_dim = 9
# cfg.search_menu_name = "first"
cfg.search_menu_name = "full"
# cfg.search_menu_name = "once_video"
cfg.ws = 15
cfg.wt = 3
cfg.wr = 1
cfg.kr = 1.
cfg.arch_depth = [20,1]
cfg.arch_nheads = [1,1]
cfg.nres_per_block = 0
cfg.num_res = 0
model = nlnet.load_model(cfg)
# print(cfg)

timer = ExpTimer()
memer = GpuMemer()


B,T,C,H,W = 5,7,3,128,128
video = th.randn((B,T,C,H,W),device=cfg.device)
clean = th.randn((B,T,C,H,W),device=cfg.device)
flows = edict()
flows.fflow = th.randn((B,T,2,H,W),device=cfg.device)
flows.bflow = th.randn((B,T,2,H,W),device=cfg.device)
# with th.no_grad():
with TimeIt(timer,"fwd"):
    with MemIt(memer,"fwd"):
        deno = model(video,flows)

# print(timer)
# print(memer)
# print(deno.shape)

# loss = th.mean((deno - clean)**2)
# with TimeIt(timer,"bwd"):
#     with MemIt(memer,"bwd"):
#         loss.backward()

print(timer)
print(memer)
