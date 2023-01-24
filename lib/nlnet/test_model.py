
# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

# -- data mngmnt --
from pathlib import Path
from easydict import EasyDict as edict

# -- data --
import data_hub

# -- networks --
import nlnet
from . import id_model

# -- dev basics --
from functools import partial
from dev_basics.aug_test import test_x8
from dev_basics import flow
from dev_basics import net_chunks
from dev_basics.utils.misc import optional,slice_flows,set_seed
from dev_basics.utils.misc import rslice,write_pickle,read_pickle
from dev_basics.utils.gpu_mem import GpuMemer,MemIt
from dev_basics.utils.timer import ExpTimer,TimeIt
from dev_basics.utils.metrics import compute_psnrs,compute_ssims,compute_strred
from dev_basics.utils import vid_io

# -- config --
from functools import partial
from dev_basics.common import optional as _optional
from dev_basics.common import optional_fields,extract_config,extract_pairs
_fields = []
optional_full = partial(optional_fields,_fields)
extract_test_config = partial(extract_config,_fields)

def _extract_test_config(_cfg,optional):
    pairs = {"device":"cuda:0","seed":123,
             "frame_start":0,"frame_end":-1,
             "aug_test":False,"longest_space_chunk":False,
             "flow":False,"burn_in":False,"arch_name":None,
             "saved_dir":"./output/saved_examples/","uuid":"uuid_def",
             "flow_sigma":-1,"internal_adapt_nsteps":0,
             "internal_adapt_nepochs":0,"nframes":0,
             "save_deno":True}
    cfg = extract_pairs(pairs,_cfg,optional)
    return cfg

def run(cfg):

    # -- config --
    init = _optional(cfg,'__init',False) # purposefully weird key
    optional = partial(optional_full,init)
    tcfg = _extract_test_config(cfg,optional)
    if init: return

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    # -- set device --
    th.cuda.set_device(int(tcfg.device.split(":")[1]))

    # -- set seed --
    set_seed(tcfg.seed)

    # -- init results --
    results = edict()
    results.psnrs = []
    results.ssims = []
    results.noisy_psnrs = []
    results.deno_fns = []
    results.vid_frames = []
    results.vid_name = []
    results.timer_flow = []
    results.timer_deno = []
    results.timer_adapt = []
    results.timer_attn = []
    results.timer_extract = []
    results.timer_search = []
    results.timer_agg = []
    results.timer_fold = []
    results.deno_mem_res = []
    results.deno_mem_alloc = []
    results.adapt_mem_res = []
    results.adapt_mem_alloc = []
    results.strred = []

    # -- load model --
    model = load_model(cfg)

    # -- data --
    imax = 255.
    data,loaders = data_hub.sets.load(cfg)
    indices = data_hub.filter_subseq(data[cfg.dset],cfg.vid_name,
                                     tcfg.frame_start,tcfg.frame_end)
    print(indices)

    for index in indices:

        # -- create timer --
        timer = ExpTimer()
        memer = GpuMemer()

        # -- clean memory --
        th.cuda.empty_cache()
        # print("index: ",index)

        # -- unpack --
        sample = data[cfg.dset][index]
        region = sample['region']
        noisy,clean = sample['noisy'],sample['clean']
        noisy,clean = noisy.to(tcfg.device),clean.to(tcfg.device)
        vid_frames = sample['fnums'].numpy()
        print("[%d] noisy.shape: " % index,noisy.shape)

        # -- resample noise for flow --
        if tcfg.flow_sigma >= 0:
            noisy_f = th.normal(clean,tcfg.flow_sigma)
        else:
            noisy_f = noisy
        # print(th.std(noisy_f - clean).item())

        # -- optical flow --
        with TimeIt(timer,"flow"):
            flows = flow.orun(noisy_f,tcfg.flow,ftype="svnlb")

        # -- augmented testing --
        if tcfg.aug_test:
            aug_fxn = partial(test_x8,model)#,use_refine=cfg.aug_refine_inds)
        else:
            aug_fxn = model.forward

        # -- chunked processing --
        chunk_cfg = net_chunks.extract_chunks_config(cfg)
        if tcfg.longest_space_chunk:
            set_longest_spatial_chunk(chunk_cfg,noisy.shape)
        fwd_fxn = net_chunks.chunk(chunk_cfg,aug_fxn)
        chunk_fwd = fwd_fxn

        # -- run once for setup gpu --
        if tcfg.burn_in:
            with th.no_grad():
                noisy_a = noisy[[0],...,:128,:128].contiguous()
                flows_a = flow.orun(noisy_a,False)
                fwd_fxn(noisy_a/imax,flows_a)
            if hasattr(model,'reset_times'):
                model.reset_times()

        # -- internal adaptation --
        adapt_psnrs = [0.]
        run_adapt = tcfg.internal_adapt_nsteps > 0
        run_adapt = run_adapt and (tcfg.internal_adapt_nepochs > 0)
        with MemIt(memer,"adapt"):
            with TimeIt(timer,"adapt"):
                if run_adapt:
                    noisy_a = noisy[:5]
                    clean_a = clean[:5]
                    flows_a = flow.slice_at(flows,slice(0,5),1)
                    region_gt = get_region_gt(noisy_a.shape)
                    adapt_psnrs = model.run_internal_adapt(
                        noisy_a,cfg.sigma,flows=flows_a,
                        clean_gt = clean_a,region_gt = region_gt,
                        chunk_fwd=chunk_fwd)
                    if hasattr(model,'reset_times'):
                        model.reset_times()

        # -- denoise! --
        with MemIt(memer,"deno"):
            with TimeIt(timer,"deno"):
                with th.no_grad():
                    deno = fwd_fxn(noisy/imax,flows)
                deno = deno.clamp(0.,1.)*imax
        mtimes = model.times

        # -- unpack if exists --
        if hasattr(model,'mem_res'):
            if model.mem_res != -1:
                memer["deno"] = (model.mem_res,model.mem_alloc)

        # -- save example --
        out_dir = Path(tcfg.saved_dir) / tcfg.arch_name / str(tcfg.uuid)
        if tcfg.save_deno:
            print("Saving to %s" % out_dir)
            deno_fns = vid_io.save_burst(deno,out_dir,"deno")
        else:
            deno_fns = ["" for _ in range(deno.shape[0])]

        # -- psnr --
        noisy_psnrs = compute_psnrs(noisy,clean,div=imax)
        psnrs = compute_psnrs(clean,deno,div=imax)
        ssims = compute_ssims(clean,deno,div=imax)
        strred = compute_strred(clean,deno,div=imax)
        print(psnrs,np.mean(psnrs),strred)

        # -- append results --
        results.psnrs.append(psnrs)
        results.ssims.append(ssims)
        results.strred.append(strred)
        results.noisy_psnrs.append(noisy_psnrs)
        results.deno_fns.append(deno_fns)
        results.vid_frames.append(vid_frames)
        results.vid_name.append([cfg.vid_name])
        for name,(mem_res,mem_alloc) in memer.items():
            key = "%s_%s" % (name,"mem_res")
            results[key].append([mem_res])
            key = "%s_%s" % (name,"mem_alloc")
            results[key].append([mem_alloc])
        for name,time in timer.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)
        for name,time in mtimes.items():
            if not(name in results):
                results[name] = []
            results[name].append(time)

    # -- clear --
    th.cuda.empty_cache()
    th.cuda.synchronize()

    return results

def load_model(cfg):
    if cfg.arch_name == "nlnet":
        return nlnet.load_model(cfg)
    else:
        raise ValueError(f"Uknown arch_name [{arch_name}]")


# -- used for adaptation --
def get_region_gt(vshape):

    t,c,h,w = vshape
    hsize = min(h//4,128)
    wsize = min(w//4,128)
    tsize = min(t//4,5)

    t_start = max(t//2 - tsize//2,0)
    t_end = min(t_start + tsize,t)
    if t == 3: t_start = 0

    h_start = max(h//2 - hsize//2,0)
    h_end = min(h_start + hsize,h)

    w_start = max(w//2 - wsize//2,0)
    w_end = min(w_start + wsize,w)

    region_gt = [t_start,t_end,h_start,w_start,h_end,w_end]
    return region_gt

