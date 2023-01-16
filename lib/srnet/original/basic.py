
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- drop path --
from timm.models.layers import DropPath

# -- project deps --
from .mlps import FastLeFF,LeFF,Mlp
from .nl_attn import NonLocalAttention

class BasicBlockList(nn.Module):
    def __init__(self, block_cfg, attn_cfg, search_cfg):
        super().__init__()
        self.block_cfg = block_cfg
        self.blocks = nn.ModuleList([
            BasicBlock(block_cfg,attn_cfg,search_cfg)
            for _ in range(block_cfg.depth)
        ])

    def extra_repr(self) -> str:
        return str(self.block_cfg)

    def forward(self, vid, flows=None, state=None):
        for blk in self.blocks:
            vid = blk(vid,h,w,mask,flows,state)
        return vid

    def flops(self,h,w):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops(h,w)
        return flops

class BasicBlock(nn.Module):

    def __init__(self, block_cfg, attn_cfg, search_cfg):
        super().__init__()

        # -- unpack vars --
        self.block_cfg = block_cfg
        self.dim = block_cfg.embed_dim * block_cfg.nheads
        self.nheads = block_cfg.nheads
        self.mlp_ratio = block_cfg.mlp_ratio
        self.block_mlp = block_cfg.block_mlp
        self.drop_mlp_rate = block_cfg.drop_rate_mlp
        self.drop_path_rate = block_cfg.drop_rate_path
        norm_layer = get_norm_layer(block_cfg.norm_layer)
        dpath = self.drop_path_rate

        # -- init layer --
        self.norm1 = norm_layer(self.dim)
        self.attn_mode = attn_cfg.attn_mode
        self.attn = NonLocalAttention(attn_cfg,search_cfg)
        self.drop_path = DropPath(dpath) if dpath > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        self.mlp = init_mlp(self.block_mlp,self.mlp_ratio,self.drop_mlp_rate,self.dim)

    def extra_repr(self) -> str:
        return str(self.block_cfg)

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
