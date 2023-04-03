

# -- linalg --
import torch as th

# -- flow --
import stnls
from dev_basics import flow

# -- timing --
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.misc import set_seed

# -- local --
from . import api


def run(cfg):

    # -- unpack --
    set_seed(cfg.seed)
    name = cfg.search_name
    device = "cuda:0"
    B = 1
    F_HD = cfg.nftrs_per_head
    T = cfg.nframes
    HD = cfg.nheads
    H = cfg.H
    W = cfg.W
    F = cfg.nftrs_per_head * cfg.nheads

    # -- data --
    vid = th.randn((B,cfg.nframes,F,cfg.H,cfg.W),device=device,dtype=th.float32)

    # -- run optial flow --
    flows = flow.orun(vid,False)
    fflow,bflow = flows.fflow,flows.bflow
    aflows = stnls.nn.accumulate_flow(flows,stride0=cfg.stride0)
    afflow,abflow = aflows.fflow,aflows.bflow

    # -- get the inds --
    nl_search = stnls.search.NonLocalSearch(cfg.ws,cfg.wt,cfg.ps,cfg.k,nheads=cfg.nheads)
    _,inds = nl_search(vid,vid,fflow,bflow)
    th.cuda.synchronize()

    # -- setup comparison function --
    search = api.init(cfg)
    search_fxn = stnls.search.utils.search_wrap(cfg.search_name,search)
    # search_fxn = search.setup_compare(vid,flows,aflows,inds)

    # -- prepare-run -
    res = {"flops":[],"radius":[],"time":[],"mem_res":[],"mem_alloc":[],"radius":[]}
    res['flops'] = search.flops(B,T,HD,F_HD,cfg.H,cfg.W)
    res['radius'] = search.radius(cfg.H,cfg.W)
    th.cuda.synchronize()
    search_fxn(vid,vid,fflow,bflow,inds,afflow,abflow)
    th.cuda.synchronize()

    # -- run --
    memer = GpuMemer(True)
    timer = ExpTimer(True)
    with MemIt(memer,name):
        with TimeIt(timer,name):
            search_fxn(vid,vid,fflow,bflow,inds,afflow,abflow)

    # -- unpack --
    res['time'] = timer[name]
    res['mem_res'] = memer[name]['res']
    res['mem_alloc'] = memer[name]['alloc']

    # -- copy to res --
    for key in res:
        if key == "name": continue
        res[key] = res[key]
    print(res)

    return res
