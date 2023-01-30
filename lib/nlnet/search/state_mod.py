

# -- logic --
from einops import rearrange


# -- clean coding --
from dev_basics.utils import clean_code
__methods__ = []
register_method = clean_code.register_method(__methods__)

# -- State API --
@register_method
def update_state(self,state,dists,inds,vshape):
    if not(self.use_state_update): return
    T,C,H,W = vshape[-4:]
    nH = (H-1)//self.stride0+1
    nW = (W-1)//self.stride0+1
    state[1] = self.inds_rs0(inds.detach(),nH,nW)

@register_method
def inds_rs0(self,inds,nH,nW):
    if not(inds.ndim == 5): return inds
    rshape = 'b h (T nH nW) k tr -> T nH nW b h k tr'
    inds = rearrange(inds,rshape,nH=nH,nW=nW)
    return inds

@register_method
def inds_rs1(self,inds):
    if not(inds.ndim == 7): return inds
    rshape = 'T nH nW b h k tr -> b h (T nH nW) k tr'
    inds = rearrange(inds,rshape)
    return inds

@register_method
def unpack_state(self,state):
    if state is None:
        raise ValueError("Must have state not None.")
    return self.inds_rs1(state[0])


