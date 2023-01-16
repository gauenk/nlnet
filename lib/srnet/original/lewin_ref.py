
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat
from functools import partial

# -- extra deps --
import math
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple

# -- project deps --
from .mlps import FastLeFF,LeFF,Mlp
from .attn import Attention
from .window_attn import WindowAttention
from .window_attn_ref import WindowAttentionRefactored
from .window_attn_dnls import WindowAttentionDnls
from .window_utils import window_partition,window_reverse
from .product_attn import ProductAttention
from .l2_attn import L2Attention

def select_attn(attn_mode,sub_attn_mode):
    if attn_mode == "window":
        return select_window_attn(sub_attn_mode)
    elif attn_mode == "product":
        return select_prod_attn(sub_attn_mode)
    elif attn_mode == "stream":
        return select_prod_attn(sub_attn_mode,"stream")
    elif attn_mode == "l2":
        return select_l2_attn(sub_attn_mode)
    else:
        raise ValueError(f"Uknown window attn type [{attn_mode}]")

def select_prod_attn(sub_attn_mode):
    return partial(ProductAttention,search_fxn=sub_attn_mode)

def select_l2_attn(sub_attn_mode):
    return L2Attention

def select_window_attn(attn_mode):
    if attn_mode == "default" or attn_mode == "original":
        return WindowAttention
    elif attn_mode == "refactored":
        return WindowAttentionRefactored
    elif attn_mode == "dnls":
        return WindowAttentionDnls
    else:
        raise ValueError(f"Uknown window attn type [{attn_mode}]")

class LeWinTransformerBlockRefactored(nn.Module):

    def __init__(self, block_cfg, attn_cfg, search_cfg):
        super().__init__()

        # -- unpack vars --
        self.dim = block_cfg.dim
        self.input_resolution = block_cfg.input_resolution
        self.num_heads = block_cfg.num_heads
        self.mlp_ratio = block_cfg.mlp_ratio
        self.block_mlp = block_cfg.block_mlp

        # -- init layer --
        self.norm1 = norm_layer(dim)
        self.attn_mode = attn_mode
        self.attn = init_attn_block(attn_cfg,search_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if block_mlp in ['ffn','mlp']:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop)
        elif block_mlp=='leff':
            self.mlp =  LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)

        elif block_mlp=='fastleff':
            self.mlp =  FastLeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")


    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, vid, flows=None, state=None):

        # -=-=-=-=-=-=-=-=-
        #    Attn Layer
        # -=-=-=-=-=-=-=-=-

        # -- create shortcut --
        B,T,C,H,W = vid.shape
        shortcut = vid

        # -- norm layer --
        vid = vid.view(B*T,C,H*W).transpose(1,2)
        vid = self.norm1(vid)
        vid = vid.transpose(1,2).view(B, T, C, H, W)

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
        # H, W = self.input_resolution
        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H*W, self.win_size*self.win_size)

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


