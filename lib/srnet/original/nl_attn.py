
# -- torch network deps --
import torch as th
import torch.nn as nn
from torch.nn.functional import unfold
from einops import rearrange,repeat

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- rescale flow --
from dev_basics import flow

# -- project deps --
from .proj import ConvQKV

# -- search/normalize/aggregate --
from .. import search
from .. import normz
from .. import agg

# -- local --
# from .state import update_state,run_state_search

# -- dnls --
import dnls

# -- modules --
# from . import inds_buffer
from . import attn_mods
from dev_basics.utils import clean_code

@clean_code.add_methods_from(attn_mods)
class NonLocalAttention(nn.Module):
    def __init__(self, dim_mult, attn_cfg, search_cfg, normz_cfg, agg_cfg):
        super().__init__()

        # -- init configs --
        dim = attn_cfg.embed_dim*attn_cfg.nheads*dim_mult
        self.dim = dim
        self.attn_cfg = attn_cfg
        self.search_cfg = search_cfg

        # -- attn info --
        self.token_projection = attn_cfg.token_projection
        self.qkv = ConvQKV(dim,attn_cfg.nheads,
                           dim_mult*attn_cfg.embed_dim,
                           attn_cfg.qk_frac,bias=attn_cfg.qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_cfg.drop_rate_proj)

        # -- init attn fxns --
        self.search = search.init(search_cfg)
        self.normz = normz.init(normz_cfg)
        self.agg = agg.init(agg_cfg)

        # -- init vars of interest --
        self.ps = search_cfg.ps
        self.use_flow = search_cfg.use_flow
        self.stride0 = search_cfg.stride0
        self.dilation = search_cfg.dilation
        self.k_s = search_cfg.k
        self.k_n = normz_cfg.k_n
        self.k_a = agg_cfg.k_a

    def get_qkv(self,vid):

        # -- compute --
        B, T, C, H, W = vid.shape
        vid = vid.view(B*T,C,H,W)
        q_vid, k_vid, v_vid = self.qkv(vid,None)

        # -- reshape --
        q_vid = q_vid.view(B,T,-1,H,W)
        k_vid = k_vid.view(B,T,-1,H,W)
        v_vid = v_vid.view(B,T,-1,H,W)

        return q_vid,k_vid,v_vid

    def run_search(self,q_vid,k_vid,flows,state):
        dists,inds = self.search(q_vid,k_vid,flows,state)
        return dists,inds

    def run_normalize(self,dists):
        dists = self.normz(dists)
        return dists

    def run_aggregation(self,v_vid,dists,inds):
        patches = self.agg(v_vid,dists,inds)
        return patches

    def run_transform(self,patches):
        patches = self.proj(patches)
        patches = self.proj_drop(patches)
        return patches

    def run_fold(self,patches,vshape):
        raise NotImplementedError("")

    def forward(self, vid, flows=None, state=None):

        # -- update flow --
        B,T,C,H,W = vid.shape
        if self.use_flow: flows = flow.rescale_flows(flows,H,W)
        self.search.set_flows(vid,flows)

        # -- extract --
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)

        # -- normalize --
        dists = self.run_normalize(dists)

        # -- aggregate --
        patches = self.run_aggregation(v_vid,dists,inds)

        # -- transform --
        patches = self.run_transform(patches)

        # -- fold --
        vid = self.run_fold(patches,vid.shape)

        return vid

    def get_patches(self,vid):
        vid = rearrange(vid,'B T C H W -> (B T) C H W')
        patches = unfold(vid,(self.ps,self.ps))
        patches = rearrange(patches,'b (p2 d) n -> (b n p2) 1 d',d=self.dim)
        return patches

    def extra_repr(self) -> str:
        str_repr = "Attention: \n" + str(self.attn_cfg) + "\n"*5
        str_repr += "Search: \n" + str(self.search_cfg) + "\n"*5
        return str_repr

    def flops(self, H, W):

        # -- init flops --
        flops = 0

        # -- num of reference points --
        nrefs = ((H-1)//self.stride0+1) * ((W-1)//self.stride0+1)

        # -- convolution flops --
        flops += self.qkv.flops(H,W)
        # print("product: ",self.qkv.flops(H,W))

        # -- non-local search --
        C = self.qkv.to_q.out_channels
        vshape = (1,C,H,W)
        flops += self.search.flops(1,C,H,W)
        # print(vshape)
        # print("search flops: ",self.search.flops(1,C,H,W))

        # -- normalize --
        flops += self.normz.flops()

        # # -- weighted patch sum --
        # k = self.search.k
        # nheads = self.search.nheads
        # chnls_per_head = C//nheads
        # flops += self.wpsum.flops(nrefs,chnls_per_head,nheads,k)
        # # print("wpsum flops: ",self.wpsum.flops(nrefs,chnls_per_head,nheads,k))
        flops += self.agg.flops()

        # -- projection --
        flops += nrefs * self.dim * self.dim

        # -- fold --
        ps = self.search_cfg.ps
        flops += nrefs * ps * ps
        # print(flops)

        return flops

