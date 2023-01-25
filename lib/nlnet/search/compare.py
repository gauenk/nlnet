

# -- linalg --
import torch as th

# -- flow --
import dnls
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
    F = cfg.nftrs_per_head * cfg.nheads

    # -- data --
    vid = th.randn((1,cfg.nframes,F,cfg.H,cfg.W),device=device,dtype=th.float32)

    # -- run optial flow --
    flows = flow.orun(vid,False)
    aflows = dnls.nn.ofa.run(flows,stride0=cfg.stride0)

    # -- get the inds --
    nl_search = api.nl.init(cfg)
    _,inds = nl_search(vid,vid)
    th.cuda.synchronize()

    # -- setup comparison function --
    search = api.init(cfg)
    search_fxn = search.setup_compare(vid,flows,aflows,inds)

    # -- prepare-run -
    res = {"flops":[],"radius":[],"time":[],"mem_res":[],"mem_alloc":[],"radius":[]}
    kwargs = {"inds":inds,"flows":flows,"aflows":aflows}
    res['flops'] = search.flops(1,F,cfg.H,cfg.W)/(1.*10**9)
    res['radius'] = search.radius(cfg.H,cfg.W)
    th.cuda.synchronize()
    search_fxn(vid,vid)
    th.cuda.synchronize()

    # -- run --
    memer = GpuMemer(True)
    timer = ExpTimer(True)
    with MemIt(memer,name):
        with TimeIt(timer,name):
            search_fxn(vid,vid)

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
