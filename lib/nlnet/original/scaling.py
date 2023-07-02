
# -- torch network deps --
import torch as th
import torch.nn as nn
from einops import rearrange,repeat

# -- extra deps --
import math


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#
#    Scaling Image Resolution
#
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        # print("x.shape: ",x.shape)
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        # B, L, C = x.shape
        # import pdb;pdb.set_trace()
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x)#.flatten(2).transpose(1,2).contiguous()  # B H*W C
        # print("out.shape:" ,out.shape)
        BT,C,H,W = out.shape
        out = out.view(B,T,C,H,W)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        # print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class UpsamplePixelShuffle(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        kernel_size = 3
        self.in_conv = nn.Conv2d(
            in_channel, 4*in_channel, kernel_size,
            padding=(kernel_size//2),stride=1, bias=True)
        self.out_conv = nn.Conv2d(
            in_channel, out_channel, kernel_size,
            padding=(kernel_size//2),stride=1, bias=True)

        self.deconv = nn.PixelShuffle(2)
        self.in_channel = in_channel
        self.out_channel = out_channel

    def extra_repr(self) -> str:
        str_repr = "Upsample(in=%d,out=%d" % (self.in_channel,self.out_channel)
        return str_repr

    def forward(self, x):
        # print("x.shape: ",x.shape)
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)

        # B, L, C = x.shape
        # H = int(math.sqrt(L))
        # W = int(math.sqrt(L))
        # x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.in_conv(x)
        # print("x.shape: ",x.shape)
        x = self.deconv(x)#.flatten(2).transpose(1,2).contiguous() # B H*W C
        # print("x.shape: ",x.shape)
        out = self.out_conv(x)
        # print("out.shape: ",out.shape)
        BT,C,H,W = out.shape
        out = out.view(B,T,C,H,W)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        # print("Upsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, up_method="deconv"):
        super(Upsample, self).__init__()
        self.up_method = up_method
        if up_method == "convT":
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
            )
        elif up_method == "interp":
            self.deconv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1),
            )
        else:
            raise ValueError(f"Uknown upsample method [{up_method}]")
        self.in_channel = in_channel
        self.out_channel = out_channel

    def extra_repr(self) -> str:
        str_repr = "Upsample(in=%d,out=%d)" % (self.in_channel,self.out_channel)
        return str_repr

    def forward(self, x):
        # print("x.shape: ",x.shape)
        B, T, C, H, W = x.shape
        x = x.view(B*T,C,H,W)
        if self.up_method == "convT":
            out = self.deconv(x)
        elif self.up_method == "interp":
            out = th.nn.functional.interpolate(x,size=(2*H,2*W),
                                               mode='bilinear',align_corners=False)
            out = self.deconv(out)
        else:
            raise ValueError(f"Uknown upsample method [{self.up_method}]")
        out = out.view(B,T,-1,2*H,2*W)
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        # print("Upsample:{%.2f}"%(flops/1e9))
        return flops
