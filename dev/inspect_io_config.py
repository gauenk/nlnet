import nlnet
import torch as th
from easydict import EasyDict as edict
from dev_basics.utils.timer import TimeIt,ExpTimer
from dev_basics.utils.gpu_mem import MemIt,GpuMemer
import cache_io



def run_exp(cfg):

    # print(cfg)

    # -- load --
    net_cfg = nlnet.extract_config(cfg)
    # print(net_cfg)
    model = nlnet.load_model(net_cfg)
    cfg.device = net_cfg.device
    # print(cfg)
    
    # -- init timers --
    timer = ExpTimer()
    memer = GpuMemer()
    
    # -- init cuda --
    video = th.randn((1,2,3,64,64),device=cfg.device)
    flows = edict()
    flows.fflow = th.randn((1,2,2,64,64),device=cfg.device)
    flows.bflow = th.randn((1,2,2,64,64),device=cfg.device)
    with th.no_grad():
        deno = model(video,flows)
        
    # -- real test --
    B,T,C,H,W = cfg.B,cfg.T,3,cfg.H,cfg.W
    video = th.randn((B,T,C,H,W),device=cfg.device)
    clean = th.randn((B,T,C,H,W),device=cfg.device)
    flows = edict()
    flows.fflow = th.randn((B,T,2,H,W),device=cfg.device)
    flows.bflow = th.randn((B,T,2,H,W),device=cfg.device)
    with th.no_grad():
        with TimeIt(timer,"fwd"):
            with MemIt(memer,"fwd"):
                deno = model(video,flows)
    
    # print(timer)
    # print(memer)
    # print(deno.shape)
    
    loss = th.mean((deno - clean)**2)
    with TimeIt(timer,"bwd"):
        with MemIt(memer,"bwd"):
            pass
            # loss.backward()
    
    # print(timer)
    # print(memer)

    # -- results --
    results = edict()
    for key,val in timer.items():
        results[key] = val
    for key,(val0,val1) in memer.items():
        results[key+"_res"] = val0
        results[key+"_alloc"] = val1
    return results

def main():

    exps_cfg = {"group0":
                {"embed_dim":[9]},
                "group1":
                {"search_menu_name":["first","full","once_video"]},
                "cfg":
                {"ws":21,"wt":3,"wr":1,"kr":1.,
                 "arch_depth":[10,1],"arch_nheads":[1,1],
                 "nres_per_block":3,"num_res":3,"qk_frac":1.},
                "listed0":{"H":[64,128,256,512],"W":[64,128,256,512]},
                "global0":{"B":[1],"T":[7]},
                }
    exps = cache_io.exps.unpack(exps_cfg)

    # -- run exps -
    records = cache_io.run_exps(exps,run_exp,
                                name = ".cache_io/dev/inspect_io_config",
                                version = "v1",
                                clear=True,skip_loop=False,
                                clear_fxn=None,
                                enable_dispatch="slurm")

    # -- view --
    records = records.rename(columns={"search_menu_name":"smn"})
    print(records[['smn',"timer_fwd","timer_bwd","fwd_res","bwd_res","B","T","H"]])

if __name__ == "__main__":
    main()
