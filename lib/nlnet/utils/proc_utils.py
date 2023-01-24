"""
Processing Utils

"""

import math
import numpy as np
import torch as th
from functools import partial
from easydict import EasyDict as edict

def _vprint(verbose,*args,**kwargs):
    if verbose:
        print(*args,**kwargs)

def extract_proc_cfg(in_cfg):
    def_cfg = {"spatial_crop_size":0,
               "spatial_crop_overlap":0,
               "temporal_crop_size":0,
               "temporal_crop_overlap":0}
    cfg = edict()
    for key in def_cfg:
        if key in in_cfg:
            cfg[key] = in_cfg[key]
        else:
            cfg[key] = def_cfg[key]
    return cfg

def proc_wrapper(cfg,model):
    model.forward = get_fwd_fxn(cfg,model.forward)

def get_fwd_fxn(cfg,model):
    s_verbose = True
    t_verbose = True
    s_size = cfg.spatial_crop_size
    s_overlap = cfg.spatial_crop_overlap
    t_size = cfg.temporal_crop_size
    t_overlap = cfg.temporal_crop_overlap
    model_fwd = lambda vid,flows: model(vid,flows=flows)
    if not(s_size is None) and not(s_size == "none") and not(s_size <= 0):
        schop_p = lambda vid,flows: spatial_chop(s_size,s_overlap,model_fwd,vid,
                                                 flows=flows,verbose=s_verbose)
    else:
        schop_p = model_fwd
    if not(t_size is None) and not(t_size == "none") and not(t_size <= 0):
        tchop_p = lambda vid,flows: temporal_chop(t_size,t_overlap,schop_p,vid,
                                                  flows=flows,verbose=t_verbose)
        fwd_fxn = tchop_p # rename
    else:
        fwd_fxn = schop_p
    return fwd_fxn

def expand2square(timg,factor=16.0):
    b, t, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = th.zeros(b,t,3,X,X).type_as(timg) # 3, h,w
    mask = th.zeros(b,t,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

def get_chunks(size,chunk_size,overlap):
    """

    Thank you to https://github.com/Devyanshu/image-split-with-overlap/

    args:
      size = original size
      chunk_size = size of output chunks
      overlap = percent (from 0.0 - 1.0) of overlap for each chunk

    This code splits an input size into chunks to be used for
    split processing

    """
    points = [0]
    stride = max(int(chunk_size * (1-overlap)),1)
    if size <= chunk_size: return [0]
    assert stride > 0
    counter = 1
    while True:
        pt = stride * counter
        if pt + chunk_size >= size:
            points.append(size - chunk_size)
            break
        else:
            points.append(pt)
        counter += 1
    points = list(np.unique(points))
    return points

def get_spatial_chunk(vid,h_chunk,w_chunk,size):
    return vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size]

def fill_spatial_chunk(vid,ivid,h_chunk,w_chunk,size):
    vid[...,h_chunk:h_chunk+size,w_chunk:w_chunk+size] += ivid

def get_spatial_chunk_flow(flows,h_chunk,w_chunk,ssize):
    out_flows = edict()
    out_flows.fflow = flows.fflow[...,h_chunk:h_chunk+ssize,w_chunk:w_chunk+ssize]
    out_flows.bflow = flows.bflow[...,h_chunk:h_chunk+ssize,w_chunk:w_chunk+ssize]

    # -- contig --
    out_flows.fflow = out_flows.fflow.contiguous().clone()
    out_flows.bflow = out_flows.bflow.contiguous().clone()

    return out_flows

def get_temporal_chunk_flow(flows,t_slice):
    if flows is None:
        return None
    out_flows = edict()
    out_flows.fflow = flows.fflow[...,t_slice,:,:,:].contiguous().clone()
    out_flows.bflow = flows.bflow[...,t_slice,:,:,:].contiguous().clone()

    # -- endpoints --
    out_flows.fflow[...,-1,:,:,:] = 0.
    out_flows.bflow[...,0,:,:,:] = 0.

    return out_flows

def spatial_chop(ssize,overlap,fwd_fxn,vid,flows=None,verbose=False):
    """
    overlap is a _percent_

    """
    vprint = partial(_vprint,verbose)
    H,W = vid.shape[-2:] # .... H, W
    deno,Z = th.zeros_like(vid),th.zeros_like(vid)
    h_chunks = get_chunks(H,ssize,overlap)
    w_chunks = get_chunks(W,ssize,overlap)
    vprint("h_chunks,w_chunks: ",h_chunks,w_chunks)
    for h_chunk in h_chunks:
        for w_chunk in w_chunks:
            vid_chunk = get_spatial_chunk(vid,h_chunk,w_chunk,ssize)
            flow_chunk = get_spatial_chunk_flow(flows,h_chunk,w_chunk,ssize)
            vprint("s_chunk: ",h_chunk,w_chunk,vid_chunk.shape)
            if flows: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            else: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
            ones = th.ones_like(deno_chunk)
            fill_spatial_chunk(deno,deno_chunk,h_chunk,w_chunk,ssize)
            fill_spatial_chunk(Z,ones,h_chunk,w_chunk,ssize)
    deno /= Z # normalize across overlaps
    return deno


def temporal_chop(tsize,overlap,fwd_fxn,vid,flows=None,verbose=True):
    """
    overlap is a __percent__
    """
    vprint = partial(_vprint,verbose)
    nframes = vid.shape[-4]
    t_chunks = get_chunks(nframes,tsize,overlap)
    vprint("t_chunks: ",t_chunks)
    deno,Z = th.zeros_like(vid),th.zeros_like(vid)
    for t_chunk in t_chunks:

        # -- extract --
        t_slice = slice(t_chunk,t_chunk+tsize)
        vid_chunk = vid[...,t_slice,:,:,:]
        vprint("t_chunk: ",t_chunk,vid_chunk.shape)
        flow_chunk = get_temporal_chunk_flow(flows,t_slice)

        # -- process --
        if flows: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)
        else: deno_chunk = fwd_fxn(vid_chunk,flow_chunk)

        # -- accumulate --
        ones = th.ones_like(deno_chunk)
        deno[...,t_slice,:,:,:] += deno_chunk
        Z[...,t_slice,:,:,:] += ones
    deno /= Z
    return deno

