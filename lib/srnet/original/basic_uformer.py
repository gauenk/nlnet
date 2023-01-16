
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- project deps --
from .lewin import LeWinTransformerBlock
from .lewin_ref import LeWinTransformerBlockRefactored


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
        self.dim = block_cfg.dim * block_cfg.nheads
        self.input_resolution = block_cfg.input_resolution
        self.nheads = block_cfg.nheads
        self.mlp_ratio = block_cfg.mlp_ratio
        self.block_mlp = block_cfg.block_mlp
        self.drop = block_cfg.drop

        # -- init layer --
        self.norm1 = norm_layer(self.dim)
        self.attn_mode = attn_mode
        self.attn = init_attn_block(attn_cfg,search_cfg)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(self.dim)
        self.mlp = init_mlp(self.block_mlp,self.mlp_ratio,self.drop,self.dim)

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

def create_basic_enc_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                           mlp_ratio,qk_frac,qkv_bias,qk_scale,drop_rate,
                           attn_drop_rate,norm_layer,use_checkpoint,
                           token_projection,token_mlp,shift_flag,attn_mode,
                           k,ps,pt,ws,wt,dil,stride0,stride1,
                           nbwd,rbwd,num_enc,exact,bs,update_dists,
                           drop_path,l):
    mult = 2**l
    isize = img_size // 2**l
    nheads = num_heads[l]
    layer = BasicUformerLayer(dim=embed_dim[l]*nheads,
                              output_dim=embed_dim[l]*nheads,
                              input_resolution=(isize,isize),
                              depth=depths[l],
                              num_heads=num_heads[l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=drop_path[sum(depths[:l]):sum(depths[:l+1])],
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                              ws=ws[l], wt=wt[l], dil=dil[l],
                              stride0=stride0[l], stride1=stride1[l],
                              nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l],
                              bs=bs[l], qk_frac=qk_frac[l],
                              update_dists=update_dists[l])
    return layer

def create_basic_conv_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                            mlp_ratio,qk_frac,qkv_bias,qk_scale,drop_rate,
                            attn_drop_rate,norm_layer,use_checkpoint,
                            token_projection,token_mlp,shift_flag,attn_mode,
                            k,ps,pt,ws,wt,dil,stride0,stride1,
                            nbwd,rbwd,num_enc,exact,bs,update_dists,
                            drop_path,l):
    nheads = num_heads[l]
    isize = img_size // 2**l
    layer = BasicUformerLayer(dim=embed_dim[l]*nheads,
                              output_dim=embed_dim[l]*nheads,
                              input_resolution=(isize,isize),
                              depth=depths[num_enc],
                              num_heads=num_heads[l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=drop_path,
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              attn_mode=attn_mode[l], k=k[l], ps=ps[l], pt=pt[l],
                              ws=ws[l], wt=wt[l], dil=dil[l],
                              stride0=stride0[l], stride1=stride1[l],
                              nbwd=nbwd[l], rbwd=rbwd[l], exact=exact[l],
                              bs=bs[l], qk_frac=qk_frac[l],
                              update_dists=update_dists[l])
    return layer

def create_basic_dec_layer(base,embed_dim,img_size,depths,num_heads,win_size,
                           mlp_ratio,qk_frac,qkv_bias,qk_scale,drop_rate,
                           attn_drop_rate,norm_layer,use_checkpoint,
                           token_projection,token_mlp,shift_flag,
                           modulator,cross_modulator,attn_mode,
                           k,ps,pt,ws,wt,dil,stride0,stride1,
                           nbwd,rbwd,num_enc,exact,bs,update_dists,
                           drop_path,l):
    # -- size --
    _l = (num_enc - l)
    lr = num_enc - l - 1
    isize = img_size // (2**lr)
    nheads = 2*num_heads[lr]
    # print("[dec]: ",l,lr,2**lr,nheads)

    # -- drop paths --
    # l == 0 | dec_dpr[:depths[5]]
    # l == 1 | dec_dpr[sum(depths[5:6]):sum(depths[5:7])]
    nbs = num_enc+1
    if l == 0:
        dpr = drop_path[:depths[nbs]]
    else:
        s = sum(depths[nbs:nbs+l])
        e = sum(depths[nbs:nbs+l+1])
        dpr = drop_path[s:e]
    # print(dpr)
    # print(depths,l,num_enc+1,num_enc+1+l)
    # print(mult)
    # print(l,_l,lr,2**(_l),mult)
    # print("num_enc: ",num_enc)
    # print(drop_path)
    # print(dpr)
    # print("[dec] l,mult,num_heads: ",l,mult,num_heads[num_enc+1+l])

    # -- init --
    layer = BasicUformerLayer(dim=embed_dim[lr]*nheads,
                              output_dim=embed_dim[lr]*nheads,
                              input_resolution=(isize,isize),
                              depth=depths[num_enc+1+l],
                              num_heads=num_heads[num_enc+1+l],
                              win_size=win_size,
                              mlp_ratio=mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=dpr,
                              norm_layer=norm_layer,
                              use_checkpoint=use_checkpoint,
                              token_projection=token_projection,
                              token_mlp=token_mlp,shift_flag=shift_flag,
                              modulator=modulator,cross_modulator=cross_modulator,
                              attn_mode=attn_mode[lr], k=k[lr], ps=ps[lr], pt=pt[lr],
                              ws=ws[lr], wt=wt[lr], dil=dil[lr],
                              stride0=stride0[lr], stride1=stride1[lr],
                              nbwd=nbwd[lr], rbwd=rbwd[lr], exact=exact[lr],
                              bs=bs[lr], qk_frac=qk_frac[lr],
                              update_dists=update_dists[lr])
    return layer

