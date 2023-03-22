"""

The primary SrNet class

"""


# -- torch network deps --
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from functools import partial

# -- extra deps --
import dnls
from timm.models.layers import trunc_normal_

# -- project deps --
# from .basic import BlockList
from .blocklist import BlockList
from .scaling import Downsample,Upsample
from .proj import InputProj,InputProjSeq,OutputProj,OutputProjSeq
from ..utils.model_utils import apply_freeze,cfgs_slice

# -- benchmarking --
from ..utils.timer import ExpTimerList

class SrNet(nn.Module):

    def __init__(self, arch_cfg, search_cfg, blocklists, scales, blocks):
        super().__init__()

        # -- init --
        self.num_blocks = len(blocklists)
        assert self.num_blocks % 2 == 1,"Must be odd."
        self.num_encs = len(blocklists)//2
        self.num_decs = len(blocklists)//2
        self.dd_in = arch_cfg.dd_in
        num_encs = self.num_encs
        self.pos_drop = nn.Dropout(p=arch_cfg.drop_rate_pos)
        block_keys = ["blocklist","attn","search","normz","agg"]
        self.use_search_input = arch_cfg.use_search_input
        self.share_encdec = arch_cfg.share_encdec

        # -- dev --
        self.inspect_print = False

        # -- benchmarking --
        self.attn_timer = arch_cfg.attn_timer
        self.times = ExpTimerList(arch_cfg.attn_timer)

        # -- input/output --
        nhead0 = blocklists[0].nheads
        self.input_proj = InputProjSeq(depth=arch_cfg.input_proj_depth,
                                       in_channel=arch_cfg.dd_in,
                                       out_channel=arch_cfg.embed_dim*nhead0,
                                       kernel_size=3, stride=1,
                                       act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2*arch_cfg.embed_dim*nhead0,
                                      out_channel=arch_cfg.in_chans,
                                      kernel_size=3,stride=1)

        # -- init --
        start,stop = 0,0

        # -- encoder layers --
        enc_list = []
        for enc_i in range(num_encs):

            # -- init --
            start = stop
            stop = start + blocklists[enc_i].depth
            blocklist_i = blocklists[enc_i]
            blocks_i = [blocks[i] for i in range(start,stop)]
            enc_layer = BlockList("enc",blocklist_i,blocks_i)
            down_layer = Downsample(scales[enc_i].in_dim,scales[enc_i].out_dim)
            setattr(self,"encoderlayer_%d" % enc_i,enc_layer)
            setattr(self,"dowsample_%d" % enc_i,down_layer)

            # -- add to list --
            paired_layer = [enc_layer,down_layer]
            enc_list.append(paired_layer)

        self.enc_list = enc_list

        # -- center --
        start = stop
        stop = start + blocklists[num_encs].depth
        blocklist_i = blocklists[num_encs]
        blocks_i = [blocks[i] for i in range(start,stop)]
        setattr(self,"conv",BlockList("conv",blocklist_i,blocks_i))

        # -- decoder --
        dec_list = []
        for dec_i in range(num_encs+1,2*num_encs+1):

            # -- init --
            start = stop
            stop = start + blocklists[dec_i].depth
            blocklist_i = blocklists[dec_i]
            blocks_i = [blocks[i] for i in range(start,stop)]
            up_layer = Upsample(scales[dec_i].in_dim,scales[dec_i].out_dim)
            dec_layer = BlockList("dec",blocklist_i,blocks_i)
            setattr(self,"upsample_%d" % dec_i,up_layer)
            setattr(self,"decoderlayer_%d" % dec_i,dec_layer)

            # -- add to list --
            paired_layer = [up_layer,dec_layer]
            dec_list.append(paired_layer)

        self.dec_list = dec_list
        self.apply(self._init_weights)

        # -- first search --
        search_cfg.nheads = arch_cfg.arch_nheads[0]
        if self.use_search_input == "video":
            search_cfg.nheads = 1
        self.search = dnls.search.init(search_cfg)

    def _apply_freeze(self):
        if all([f is False for f in self.freeze]): return
        apply_freeze(self,self.freeze)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @th.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @th.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def search_input(self, video, features, flows, state):
        if self.use_search_input == "none": return None
        srch = video if self.use_search_input == "video" else features
        with th.no_grad():
            dists,inds = self.search(srch,srch,flows.fflow,flows.bflow)
            self.update_state(state,dists,inds,srch.shape)

    def forward(self, vid, flows=None, states=None):


        # -- unpack --
        b,t,c,h,w = vid.shape

        # -- Input Projection --
        y = self.input_proj(vid)
        y = self.pos_drop(y)
        num_encs = self.num_encs

        # -- init states --
        if states is None:
            states = [None,None]# for _ in range(2*num_encs+1)]

        # -- optional search --
        self.search_input(vid,y,flows,states)

        # -- enc --
        z = y
        encs = []
        share_states = []
        states_i = [states[0],None]
        for i,(enc,down) in enumerate(self.enc_list):

            # -- optionally save for decoder --
            if self.share_encdec:
                share_states.append(states_i)

            # -- forward --
            z = enc(z,flows=flows,state=states_i)
            self.iprint("[enc] i: %d" % i,z.shape)
            encs.append(z)
            z = down(z)
            self.iprint("[dow] i: %d" % i,z.shape)

            # -- downsample states --
            states_i = self.down_states(states_i)

        # -- middle --
        iH,iW = z.shape[-2:]
        z = self.conv(z,flows=flows,state=states_i)
        self.iprint("[mid]: ",z.shape)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):

            # -- load encoder state --
            if self.share_encdec:
                states_i = share_states[-(i+1)]

            # -- forward --
            i_rev = (num_encs-1)-i
            z = up(z)
            self.iprint("[up] i: %d" % i,z.shape)
            z = self.cat_pad(z,encs[i_rev],-3)
            self.iprint("[cat] i: %d" % i,z.shape)
            z = dec(z,flows=flows,state=states_i)
            self.iprint("[dec] i: %d" % i,z.shape)

            # # -- update state --
            # states_i = self.up_states(states_i)

        # -- Output Projection --
        y = self.output_proj(z)

        # -- residual connection --
        out = vid + y if self.dd_in == 3 else y

        # -- timing --
        self.update_block_times()
        # print("done.")

        return out

    def cat_pad(self,z,enc,dim):

        # -- find edges --
        eH,eW = enc.shape[-2:]
        b,t,c,zH,zW = z.shape
        dH = eH - zH
        dW = eW - zW
        if dH == 0 and dW == 0:
            return th.cat([z,enc],dim)

        # -- compute pads --
        dH_half = int(dH/2)
        dH_r = dH - dH_half
        dW_half = int(dW/2)
        dW_r = dW - dW_half
        pads = (dW_r,dW_half,dH_r,dH_half)
        z = rearrange(z,'b t c h w -> (b t) c h w')
        z = F.pad(z,pads)
        z = rearrange(z,'(b t) c h w -> b t c h w',b=b)

        return th.cat([z,enc],dim)


    def down_states(self,states):
        return [self.down_inds(states[0]),self.down_inds(states[1])]

    def up_states(self,states):
        return [self.up_inds(states[0]),self.up_inds(states[1])]

    def down_state(self,state):
        return self.down_inds(state)

    def up_state(self,state):
        return self.up_inds(state)

    def down_inds(self,inds):
        if inds is None: return inds

        # -- downsample shape --
        T,H,W,B,HD,K,_ = inds.shape
        inds = inds[:,::2,::2].clone()

        # -- downsample values --
        inds[...,1] = th.div(inds[...,1],2,rounding_mode='floor')
        inds[...,2] = th.div(inds[...,2],2,rounding_mode='floor')

        # -- clip --
        inds[...,1] = th.clip(inds[...,1],min=0,max=H//2-1)
        inds[...,2] = th.clip(inds[...,2],min=0,max=W//2-1)

        # -- unique across K --
        print(inds.shape)
        inds = dnls.nn.jitter_unique_inds(inds,5,K,H,W)

        return inds

    def up_inds(self,inds):
        pass
        return inds

    @property
    def max_batch_size(self):
        return -1

    def iprint(self,*args,**kwargs):
        if self.inspect_print:
            print(*args,**kwargs)

    def reset_times(self):
        if self.attn_timer is False: return
        self.times = ExpTimerList(self.use_timer)
        for i in range(self.num_encs):
            layer_i = getattr(self,"encoderlayer_%d" % i)
            layer_i.reset_times()
            layer_i = getattr(self,"decoderlayer_%d" % i)
            layer_i.reset_times()
        layer_i = getattr(self,"conv")
        layer_i.reset_times()

    def update_block_times(self):
        if self.attn_timer is False: return
        for i in range(self.num_encs):
            layer_i = getattr(self,"encoderlayer_%d" % i)
            self.times.update_times(layer_i.times)
            layer_i.reset_times()
            layer_i = getattr(self,"decoderlayer_%d" % i)
            self.times.update_times(layer_i.times)
            layer_i.reset_times()
        layer_i = getattr(self,"conv")
        self.times.update_times(layer_i.times)
        layer_i.reset_times()

    def update_state(self,state,dists,inds,vshape):
        # if not(self.use_state_update): return
        T,C,H,W = vshape[-4:]
        nH = (H-1)//self.search.stride0+1
        nW = (W-1)//self.search.stride0+1
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

    def flops(self,h,w):

        # -- init flops --
        flops = 0

        # -- Input Projection --
        flops += self.input_proj.flops(h,w)
        num_encs = self.num_enc_layers

        # -- enc --
        encs = []
        for i,(enc,down) in enumerate(self.enc_list):
            _h,_w = h//(2**i),w//(2**i)
            flops += enc.flops(_h,_w)
            flops += down.flops(_h,_w)

        # -- middle --
        mod = 2**num_encs
        _h,_w = h//mod,w//mod
        flops += self.conv.flops(_h,_w)

        # -- dec --
        for i,(up,dec) in enumerate(self.dec_list):
            i_rev = num_encs-1-i
            _h,_w = h//(2**(i_rev)),w//(2**(i_rev))
            flops += up.flops(_h,_w)
            flops += dec.flops(_h,_w)

        # -- Output Projection --
        flops += self.output_proj.flops(h,w)

        return flops

