# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
from .nl_attn import NonLocalAttention

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
from . import attn_mods
from .shared import get_norm_layer
from .mlps import init_mlp
from .sk_conv import SKUnit
from .res import ResBlockList
from .misc import LayerNorm2d

class BlockV3(nn.Module):

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
        attn = block.attn
        search = block.search
        normz = block.normz
        agg = block.agg
        self.attn = NonLocalAttention(attn,search,normz,agg)

        # -- init local attn --
        self.sk_attn = SKUnit(in_features=edim,
                              out_features=edim,M=2,G=8,r=2)

        # -- init non-linearity --
        dprate = blocklist.drop_rate_path
        ksize = block.res.res_ksize
        nres = block.res.nres_per_block
        self.res = ResBlockList(nres, edim, ksize)
        self.drop_path = DropPath(dprate) if dprate > 0. else nn.Identity()

        # -- init combining layers [local vs non-local select] --
        vector_length = 32
        self.fc_share = nn.Linear(in_features=edim,out_features=vector_length)
        self.fc_0 = nn.Linear(in_features=vector_length,out_features=edim)
        self.fc_1 = nn.Linear(in_features=vector_length,out_features=edim)
        self.softmax = nn.Softmax(dim=1)

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
        nl_vid = self.attn(vid, flows=flows, state=state)

        # -=-=-=-=-=-=-=-=-=-=-=-=-
        #     Local Attn Layer
        # -=-=-=-=-=-=-=-=-=-=-=-=-

        sk_vid = self.sk_attn(vid)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #           Combo
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        combo_vid = th.stack((nl_vid,sk_vid),dim=1)
        weights = self.compute_pair_weights(combo_vid)
        vid = (combo_vid*weights).sum(dim=1)
        # vid = nl_vid

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #   Non-Linearity & Residual
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        vid = shortcut + self.drop_path(vid)
        vid = self.norm2(vid)
        vid = vid + self.drop_path(self.res(vid))

        return vid

    def compute_pair_weights(self,combo):
        U = th.sum(combo,dim=1)
        attn_vec = U.mean(-1).mean(-1)
        attn_vec = self.fc_share(attn_vec)
        attn_vec_0 = self.fc_0(attn_vec)
        attn_vec_1 = self.fc_1(attn_vec)
        vector = th.stack((attn_vec_0,attn_vec_1),dim=1)
        vector = self.softmax(vector)[...,None,None]
        return vector

    def flops(self,H,W):
        flops = 0
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


