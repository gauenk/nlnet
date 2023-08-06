"""

The primary SrNet class

"""


# -- torch network deps --
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange,repeat
from einops.layers.torch import Rearrange

# -- misc --
from functools import partial
from easydict import EasyDict as edict

# -- extra deps --
import stnls
from timm.models.layers import trunc_normal_

# -- project deps --
# from .basic import BlockList
from .blocklist import BlockList
from .scaling import Downsample,Upsample,UpsampleRvrt
from .proj import InputProj,InputProjSeq,OutputProj,OutputProjSeq,get_input_proj_rvrt
from ..utils.model_utils import apply_freeze,cfgs_slice
from .spynet import SpyNet

# -- benchmarking --
from ..utils.timer import ExpTimerList

class SrNet(nn.Module):

    def __init__(self, arch_cfg, search_cfg, blocklists, blocks):
        super().__init__()

        # -- init --
        self.num_blocks = len(blocklists)
        assert self.num_blocks % 2 == 1,"Must be odd."
        self.num_encs = len(blocklists)//2
        self.num_decs = len(blocklists)//2
        self.dd_in = arch_cfg.dd_in
        num_encs = self.num_encs
        # self.pos_drop = nn.Dropout(p=arch_cfg.drop_rate_pos)
        block_keys = ["blocklist","attn","search","normz","agg"]
        self.use_search_input = arch_cfg.use_search_input
        self.share_encdec = arch_cfg.share_encdec
        self.append_noise = arch_cfg.append_noise
        self.upscale = 1

        # -- [optional] optical flow --
        # self.spynet = SpyNet(arch_cfg.spynet_path)
        # print("spynet.")

        # -- dev --
        self.inspect_print = False

        # -- benchmarking --
        self.attn_timer = arch_cfg.attn_timer
        self.times = ExpTimerList(arch_cfg.attn_timer)

        # -- input/output --
        nhead0 = blocklists[0].nheads
        embed_dim0 = blocklists[0].embed_dim
        edim0 = embed_dim0*nhead0
        # edim0 = embed_dim0
        out_chnls = 3
        self.input_proj = get_input_proj_rvrt(edim0)
        # self.input_proj = InputProjSeq(depth=arch_cfg.input_proj_depth,
        #                                in_channel=3,#arch_cfg.dd_in,
        #                                out_channel=edim0,
        #                                kernel_size=3, stride=1,
        #                                act_layer=nn.LeakyReLU,
        #                                norm_layer=arch_cfg.input_norm_layer)
        # self.output_proj = OutputProj(in_channel=2*embed_dim0*nhead0,
        #                               out_channel=out_chnls,
        #                               kernel_size=3,stride=1)

        # -- init --
        start,stop = 0,0


        # -- upsample --
        import math
        scale = 4
        mid_ftrs = 64
        block_mult = 4 if blocklists[-1].block_version == "v13" else 1
        # num_feat = embed_dim0*nhead0*block_mult
        # num_feat = embed_dim0*block_mult
        in_ftrs = embed_dim0*nhead0*block_mult
        num_feat = 64
        out_channel = 3
        self.conv_before_upsampler = nn.Sequential(
            nn.Conv3d(in_ftrs,num_feat,kernel_size=(1, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.upsampler = UpsampleRvrt(4, num_feat)
        self.conv_last = nn.Conv3d(num_feat,
                                   out_channel, kernel_size=(1, 3, 3),
                                   padding=(0, 1, 1))

        # -- main layers --
        self.block_layer = BlockList("enc",blocklists[0],blocks)
        self.apply(self._init_weights)


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

    def upsample(self,lqs,feats):
        # hr = th.cat([feats[k] for k in feats], dim=2)
        # hr = self.reconstruction(hr)
        hr = self.conv_before_upsampler(feats.transpose(1, 2))
        hr = self.upsampler(hr)
        hr = self.conv_last(hr).transpose(1, 2)
        hr += th.nn.functional.interpolate(lqs, size=hr.shape[-3:], mode='trilinear', align_corners=False)
        return hr

    def forward(self, vid_in, flows=None, states=None):


        # -- unpack --
        b,t,c,h,w = vid_in.shape

        # -- [optional] noise channel --
        if c == 4:
            noise = vid_in[:,:,[3],:,:].contiguous()

        # if self.upscale == 4:
        #     vid = vid_in.clone()
        # else:
        #     # vid = F.interpolate(vid_in[:, :, :3, :, :].view(-1, 3, h, w),
        #     #                     scale_factor=0.25, mode='bicubic')\
        #     #        .view(b, t, 3, h // 4, w // 4)
        # vid = get_input_proj_rvrt(edim0)

        # -- Input Projection --
        y = self.input_proj(vid_in)
        # print("y.shape:" ,y.shape)
        # num_encs = self.num_encs

        # -- init states --
        if states is None:
            states = [None,None]

        # -- forward --
        z = self.block_layer(y,flows=flows,state=states)

        # # -- middle --
        # iH,iW = z.shape[-2:]
        # z = self.conv(z,flows=flows,state=states_i)
        # self.iprint("[mid]: ",z.shape)

        # -- Output Projection --
        out = self.upsample(vid_in[:,:,:3],z)

        # -- timing --
        self.update_block_times()

        return out

    def run_append_noise(self,z,noise):
        b,t,c,h,w = z.shape
        H,W = noise.shape[-2:]
        rH,rW = H//h,W//w # a multiple of 2
        z = th.cat([z,noise[...,::rH,::rW]],-3)
        return z

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


    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)
        zflow = th.zeros((n,1,2,h,w),device=lqs.device)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)
        flows_backward = th.cat([zflow,flows_backward],1)

        # if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
        #     flows_forward = None
        # else:
        flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)
        flows_forward = th.cat([flows_forward,zflow],1)

        flows = edict()
        flows.fflow = flows_forward
        flows.bflow = flows_backward
        return flows

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
        inds = stnls.nn.jitter_unique_inds(inds,5,K,H,W)

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

