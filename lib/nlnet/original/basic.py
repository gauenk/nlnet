
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
from .mlps import FastLeFF,LeFF,Mlp
from .nl_attn import NonLocalAttention

# -- benchmarking --
from dev_basics.utils.timer import ExpTimerList

# -- clean coding --
from . import attn_mods
from dev_basics.utils import clean_code

# @clean_code.add_methods_from(bench_mods)
class BlockList(nn.Module):
    def __init__(self, btype, blocklist, blocks):
        super().__init__()
        self.blocklist = blocklist
        self.blocks = nn.ModuleList(
            [Block(btype,blocklist,blocks[d])
             for d in range(blocklist.depth)])
        # self.blocks = nn.ModuleDict([
        #     ["block_%d" % d,Block(btype,blocklist,blocks[d])]
        #     for d in range(blocklist.depth)
        # ])

    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, vid, flows=None, state=None):
        state_b = [state[0],None]
        # for name,blk in self.blocks.items():
        for blk in self.blocks:
            vid = blk(vid,flows,state_b)
            state_b = [state_b[1],None]
        state[1] = state_b[0]
        return vid

    def flops(self,h,w):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(h,w)
        return flops

    @property
    def times(self):
        times = ExpTimerList()
        for blk in self.blocks:
            times.update_times(blk.attn.timer)
            blk.attn.reset_times()
        return times

    def reset_times(self):
        for blk in self.blocks:
            blk.attn.reset_times()

class Block(nn.Module):

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
        dpath = self.drop_path_rate
        mult = 2 if self.type == "dec" else 1

        # -- modify embed_dim --
        block.attn.embed_dim *= mult

        # -- init attn --
        attn = block.attn
        search = block.search
        normz = block.normz
        agg = block.agg
        self.attn = NonLocalAttention(attn,search,normz,agg)

        # -- init layer --
        self.norm1 = norm_layer(self.dim*mult)
        self.drop_path = DropPath(dpath) if dpath > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim*mult)
        self.mlp = init_mlp(self.block_mlp,self.mlp_ratio,
                            self.drop_mlp_rate,self.dim*mult)

    def extra_repr(self) -> str:
        return str(self.blocklist)

    def forward(self, vid, flows=None, state=None):

        # -=-=-=-=-=-=-=-=-
        #    Attn Layer
        # -=-=-=-=-=-=-=-=-

        # -- create shortcut --
        B,T,C,H,W = vid.shape
        shortcut = vid

        # -- norm layer --
        vid = vid.view(B*T,C,H*W)
        vid = self.norm1(vid.transpose(1,2)).transpose(1,2)
        vid = vid.view(B, T, C, H, W)

        # -- run attention --
        vid = self.attn(vid, flows=flows, state=state)

        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        #    Fully Connected Layer
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # -- view for ffn --
        vid = vid.view(B*T,C,H*W).transpose(1,2)
        shortcut = shortcut.view(B*T,C,H*W).transpose(1,2)

        # -- FFN --
        vid = shortcut + self.drop_path(vid)
        vid = vid + self.drop_path(self.mlp(self.norm2(vid)))
        vid = vid.transpose(1,2).view(B,T,C,H,W)
        return vid

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

def init_mlp(block_mlp,mlp_ratio,drop,dim):
    act_layer = nn.GELU
    mlp_hidden_dim = int(dim*mlp_ratio)
    if block_mlp in ['ffn','mlp']:
        mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
    elif block_mlp=='leff':
        mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

    elif block_mlp=='fastleff':
        mlp =  FastLeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
    else:
        raise Exception("FFN error!")
    return mlp

def get_norm_layer(layer_s):
    if layer_s == "LayerNorm":
        return nn.LayerNorm
    else:
        raise ValueError("Uknown norm layer %s" % layer_s)
