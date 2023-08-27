# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from einops.layers.torch import Rearrange
from copy import deepcopy as dcopy

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
# from .nl_attn_vid import NonLocalAttentionVideo
from stnls.pytorch.nn import NonLocalAttentionStack

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
from . import attn_mods
from .shared import get_norm_layer
from .mlps import init_mlp
# from .sk_conv import SKUnit
from .res import ResBlockList
from .misc import LayerNorm2d
from .rstb import RSTBWithInputConv
from .chnls_attn import ChannelAttention

class BlockV13(nn.Module):

    def __init__(self, btype, blocklist, block):
        super().__init__()

        # -- unpack vars --
        self.type = btype
        self.blocklist = blocklist
        self.dim = blocklist.embed_dim * blocklist.nheads
        self.mlp_ratio = blocklist.mlp_ratio
        self.block_mlp = blocklist.block_mlp
        self.drop_mlp_rate = blocklist.drop_rate_mlp
        self.drop_path_rate = blocklist.drop_rate_path
        norm_layer = get_norm_layer(blocklist.norm_layer)
        mult = 2 if self.type == "dec" else 1

        # -- modify embed_dim --
        block.attn.embed_dim *= mult
        edim = block.attn.embed_dim * blocklist.nheads
        self.edim = edim

        # -- norm layers --
        # self.norm1 = nn.Identity()
        # self.norm2 = nn.Identity()
        self.norm1 = LayerNorm2d(edim)
        self.norm2 = LayerNorm2d(edim)

        # -- init non-local attn --
        attn = dcopy(block.attn)
        search = block.search
        normz = block.normz
        agg = block.agg
        self.attn = NonLocalAttentionStack(attn,search,normz,agg)

        # -- search --
        self.search = stnls.search.init(search_cfg)


        # -- init non-linearity --
        dprate = blocklist.drop_rate_path
        ksize = block.res.res_ksize
        nres = block.res.nres_per_block
        bn = block.res.res_bn
        stg_depth = block.res.stg_depth
        stg_nheads = block.res.stg_nheads
        stg_ngroups = block.res.stg_ngroups
        self.channel_attn_0 = ChannelAttention(edim)
        self.channel_attn_1 = ChannelAttention(edim)
        self.res = RSTBWithInputConv(edim, ksize, nres, dim=edim,
                                     depth=stg_depth,num_heads=stg_nheads,
                                     groups=stg_ngroups)
        self.drop_path = DropPath(dprate) if dprate > 0. else nn.Identity()


    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, vid, flows=None, state=None):

        # -=-=-=-=-=-=-=-=-=-=-=-=-
        #       Init/Unpack
        # -=-=-=-=-=-=-=-=-=-=-=-=-

        # -- create shortcut --
        B,T,C,H,W = vid.shape
        shortcut = vid

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    Non-Local Attn Layer
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = self.norm1(vid)
        # print("[in] vid.shape: ",vid.shape)
        vid = self.run_attn(vid, flows=flows, state=state)
        # print("[out] vid.shape: ",vid.shape)
        # print("[attn] delta: ",th.mean((shortcut-vid)**2).item())
        vid = self.channel_attn_0(vid)
        # print("[chnl_attn] delta: ",th.mean((shortcut-vid)**2).item())

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #   Non-Linearity & Residual
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = shortcut + self.drop_path(vid)
        # print("[shortcut] delta: ",th.mean((shortcut-vid)**2).item())
        vid = self.norm2(vid)
        vid = self.res(vid)
        vid = self.channel_attn_1(vid)

        return vid

    def run_attn(self,vid,flows=None,state=None):

        # -- update flow --
        B,T,C,H,W = vid.shape
        if self.use_flow: flows = rescale_flows(flows,H,W)

        # -- extract --
        in_vid = vid
        vid = self.norm_layer(vid)
        q_vid,k_vid,v_vid = self.get_qkv(vid)

        # -- search --
        dists,inds = self.run_search(q_vid,k_vid,flows,state)
        # print(inds.shape)

        # -- normalize --
        weights,inds = self.run_normalize(dists,inds)

        # -- aggregate --
        vid = self.run_aggregation(v_vid,weights,inds)

        # -- projection --
        vid = self.run_projection(vid)

        # -- timing --
        self.timer.sync_stop("attn")
        if self.use_timer:
            self.times.update_times(self.timer)

        return vid


    def run_search(self,q_vid,k_vid,flows,state):
        # self.timer.sync_start("search")
        if self.search_name == "refine":
            inds_p = self.inds_rs1(state[0])
            dists,inds = self.search(q_vid,k_vid,inds_p)
        elif self.search_name == "rand_inds":
            dists,inds = self.search(q_vid,k_vid)
        else:
            dists,inds = self.search(q_vid,k_vid,flows.fflow,flows.bflow)
        self.update_state(state,dists,inds,q_vid.shape)
        # self.timer.sync_stop("search")
        return dists,inds

    def run_normalize(self,dists,inds):
        # self.timer.sync_start("normz")
        dists,inds = self.normz(dists,inds)
        # self.timer.sync_stop("normz")
        return dists,inds

    def run_aggregation(self,v_vid,weights,inds):
        # -- aggregate patches --
        # self.timer.sync_start("agg")
        stack = self.stacking(v_vid,weights,inds)
        stack = rearrange(stack,'b hd k t c h w -> b t k (hd c) h w')
        # self.timer.sync_stop("agg")
        return stack

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

    def flops(self,H,W):
        flops = 0
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        # flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops

class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

