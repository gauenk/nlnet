

def update_state(self,state,dists,inds,vshape):
    if not(self.use_state_update): return
    T,C,H,W = vshape[-4:]
    nH = (H-1)//self.stride0+1
    nW = (W-1)//self.stride0+1
    state[1] = state[0]
    state[0] = self.inds_rs0(inds.detach(),nH,nW)

def inds_rs0(self,inds,nH,nW):
    if not(inds.ndim == 5): return inds
    rshape = 'b h (T nH nW) k tr -> T nH nW b h k tr'
    inds = rearrange(inds,rshape,nH=nH,nW=nW)
    return inds

def inds_rs1(self,inds):
    if not(inds.ndim == 7): return inds
    rshape = 'T nH nW b h k tr -> b h (T nH nW) k tr'
    inds = rearrange(inds,rshape)
    return inds


def run_state_search(q_vid,qstart,ntotal,k_vid,state,
                     flows,recompute_dists):
    dists,inds = stream_search(q_vid,qstart,ntotal,k_vid,state,flows,recompute_dists)
    return dists,inds

def run_recompute_dists(q_vid,k_vid,inds,fstart):
    pass

def stream_search(q_vid,qstart,ntotal,k_vid,state,flows,recompute_dists=False):
    fstart = state.fstart
    dists_new,inds_new = search_new(q_vid,qstart,ntotal,k_vid,fstart)
    if recompute_dists: dists = run_recompute_dists(q_vid,k_vid,inds,fstart)
    else: dists = state.dists
    dists = th.cat([state.dists,dists_new],0)
    inds = th.cat([state.inds,inds_new],0)
    return dists,inds

def update_state(state,dists,inds):
    if state is None: return
    elif state.type == "overlap":
        state.dists = dists
        state.inds = inds
    elif state.type == "new_frame":
        state.dists = dists
        state.inds = inds

def update_state_overlap(stat,dists,inds):
    return

def update_state_new_frame(stat,dists,inds):
    return
