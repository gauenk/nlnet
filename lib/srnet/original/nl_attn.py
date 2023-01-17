
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
from timm.models.layers import trunc_normal_

# -- rescale flow --
from dev_basics import flow

# -- project deps --
from .proj import ConvQKV

# -- local --
from .state import update_state,run_state_search

# -- dnls --
import dnls

class NonLocalAttention(nn.Module):
    def __init__(self, dim_mult, attn_cfg, search_cfg):

        super().__init__()

        # -- init configs --
        dim = attn_cfg.embed_dim*attn_cfg.nheads*dim_mult
        self.dim = dim
        self.attn_cfg = attn_cfg
        self.search_cfg = search_cfg
        self.stride0 = self.search_cfg.stride0
        self.use_flow = self.search_cfg.use_flow

        # -- attn info --
        self.qkv = ConvQKV(dim,attn_cfg.nheads,
                           dim_mult*attn_cfg.embed_dim,
                           attn_cfg.qk_frac,bias=attn_cfg.qkv_bias)
        self.token_projection = attn_cfg.token_projection
        self.attn_drop = nn.Dropout(attn_cfg.drop_rate_attn)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(attn_cfg.drop_rate_proj)
        self.softmax = nn.Softmax(dim=-1)

        # -- init search --
        self.search = self.init_search(search_cfg)
        self.wpsum = self.init_wpsum(search_cfg)

    def get_weights(self,module):
        weights = []
        for name,mod in module.named_parameters():
            flat = mod.data.ravel()
            weights.append(flat)
        weights = th.cat(weights,0)
        return weights

    def get_qkv(self,vid):

        # -- compute --
        B, T, C, H, W = vid.shape
        vid = vid.view(B*T,C,H,W)
        q_vid, k_vid, v_vid = self.qkv(vid,None)
        q_vid = q_vid * self.attn_cfg.scale

        # -- reshape --
        q_vid = q_vid.view(B,T,-1,H,W)
        k_vid = k_vid.view(B,T,-1,H,W)
        v_vid = v_vid.view(B,T,-1,H,W)

        return q_vid,k_vid,v_vid

    def run_softmax(self,dists,vshape):
        dists = self.softmax(dists)
        dists = self.attn_drop(dists)
        dists = dists.contiguous()
        return dists

    def run_aggregation(self,v_vid,dists,inds):

        # -- params --
        B, T, _, H, W = v_vid.shape
        stride0 = self.stride0
        ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)

        # -- limiting --
        k_a = self.search_cfg.k_a
        dists = dists[...,:k_a].contiguous()
        inds = inds[...,:k_a].contiguous()

        # -- agg --
        patches = self.wpsum(v_vid,dists,inds)

        # -- reshape --
        ps = patches.shape[-1]
        shape_str = 'b h (o n) c ph pw -> (b o ph pw) n (h c)'
        patches = rearrange(patches,shape_str,o=ntotal)
        return patches

    def run_fold(self,patches,vshape):

        # -- init folding --
        B,ps = vshape[0],self.search_cfg.ps
        fold = self.init_fold(vshape,patches.device)

        # -- reshape for folding --
        shape_str = '(b o ph pw) n c -> b (o n) 1 1 c ph pw'
        patches = rearrange(patches,shape_str,b=B,ph=ps,pw=ps)
        patches = patches.contiguous()

        # -- fold --
        fold(patches,0)

        # -- unpack --
        vid = fold.vid / fold.zvid

        # -- debug --
        any_nan = th.any(th.isnan(vid))
        if any_nan:
            any_fold_nan = th.any(th.isnan(fold.vid))
            any_zero = th.any(th.abs(fold.zvid)<1e-10)
            print("found a nan!: ",any_nan,any_zero,any_fold_nan)
            exit(0)
        return vid

    def forward(self, vid, flows=None, state=None):

        # -- update flow --
        B,T,C,H,W = vid.shape
        if self.use_flow: flows = flow.rescale_flows(flows,H,W)
        self.search.update_flow(vid.shape,vid.device,flows)

        # -- qkv --
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- run search --
        dists,inds = self.run_search(q_vid,k_vid,state)

        # -- softmax --
        dists = self.run_softmax(dists,vid.shape)

        # -- aggregate --
        patches = self.run_aggregation(v_vid,dists,inds)

        # -- post-process --
        patches = self.proj(patches)
        patches = self.proj_drop(patches)

        # -- fold --
        vid = self.run_fold(patches,vid.shape)

        return vid

    def run_search(self,q_vid,k_vid,state):
        if state is None:
            # -- dnls search --
            B, T, _, H, W = q_vid.shape
            qstart,stride0 = 0,self.stride0
            ntotal = T*((H-1)//stride0+1)*((W-1)//stride0+1)
            dists,inds = self.search(q_vid,qstart,ntotal,k_vid)
        else:
            # -- streaming search --
            dists,inds = run_state_search(q_vid,qstart,ntotal,k_vid,state)
            update_state(state,dists,inds)
        return dists,inds

    def init_search(self,search_cfg):
        stype = search_cfg.search_type
        if "dnls" in stype:
            search = self.init_dnls(search_cfg)
        else:
            raise ValueError(f"Uknown search function [{stype}]")
        return search

    def init_dnls(self,cfg):

        # -- unpack params --
        k       = cfg.k_s
        if k == 0: k = -1
        ps      = cfg.ps
        pt      = cfg.pt
        ws      = cfg.ws
        wt      = cfg.wt
        dil     = cfg.dil
        stride0 = cfg.stride0
        stride1 = cfg.stride1
        nbwd    = cfg.nbwd
        rbwd    = cfg.rbwd
        exact   = cfg.exact
        nheads  = cfg.nheads
        # print("k,ps,ws,wt: ",k,ps,ws,wt)
        # print(rbwd,nbwd)

        # -- fixed --
        fflow,bflow = None,None
        use_k = k > 0
        reflect_bounds = True
        use_search_abs = False
        only_full = False
        use_adj = False
        full_ws = True
        stype = "prod_with_heads"

        # -- init --
        search = dnls.search.init(stype,fflow, bflow, k,
                                  ps, pt, ws, wt, nheads,
                                  chnls=-1,dilation=dil,
                                  stride0=stride0,stride1=stride1,
                                  reflect_bounds=reflect_bounds,
                                  use_k=use_k,use_adj=use_adj,
                                  search_abs=use_search_abs,full_ws=full_ws,
                                  h0_off=0,w0_off=0,h1_off=0,w1_off=0,
                                  exact=exact,nbwd=nbwd,rbwd=rbwd)
        return search

    def init_nat(self):

        # -- unpack params --
        k       = self.k_s
        if k == 0: k = -1
        ps      = self.ps
        pt      = self.pt
        ws      = self.ws
        wt      = self.wt
        dil     = self.dil
        stride0 = self.stride0
        stride1 = self.stride1
        nbwd    = self.nbwd
        rbwd    = self.rbwd
        exact   = self.exact
        nheads  = self.nheads

        # -- fixed --
        use_k = k > 0
        reflect_bounds = True
        use_search_abs = False
        only_full = False
        use_adj = False
        full_ws = True
        stype = "prod_with_heads"

        # -- init --
        search = nat

        return search

    def init_wpsum(self,cfg):

        # -- unpack params --
        ps      = cfg.ps
        pt      = cfg.pt
        dil     = cfg.dil

        # -- fixed --
        exact = False
        reflect_bounds = True

        # -- init --
        wpsum = dnls.reducers.WeightedPatchSumHeads(ps, pt, h_off=0, w_off=0,
                                                    dilation=dil,
                                                    reflect_bounds=reflect_bounds,
                                                    adj=0, exact=exact)
        return wpsum

    def init_fold(self,vshape,device):
        dil     = self.search_cfg.dil
        stride0 = self.search_cfg.stride0
        only_full = False
        reflect_bounds = True
        fold = dnls.iFoldz(vshape,None,stride=stride0,dilation=dil,
                           adj=0,only_full=only_full,
                           use_reflect=reflect_bounds,device=device)
        return fold

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

        # -- weighted patch sum --
        k = self.search.k
        nheads = self.search.nheads
        chnls_per_head = C//nheads
        flops += self.wpsum.flops(nrefs,chnls_per_head,nheads,k)
        # print("wpsum flops: ",self.wpsum.flops(nrefs,chnls_per_head,nheads,k))

        # -- projection --
        flops += nrefs * self.dim * self.dim

        # -- fold --
        ps = self.search_cfg.ps
        flops += nrefs * ps * ps
        # print(flops)

        return flops

