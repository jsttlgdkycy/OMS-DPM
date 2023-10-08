import math
from turtle import forward
import torch
import torch.nn as nn
import random
from ipdb import set_trace

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv #默认是True
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0) 

    def forward(self, x): #
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x) 
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2) 
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels 
        out_channels = in_channels if out_channels is None else out_channels 
        self.out_channels = out_channels 
        self.use_conv_shortcut = conv_shortcut 
        self.norm1 = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                out_channels,
                                                kernel_size=3,
                                                stride=1,
                                                padding=1) 
        self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                            out_channels,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0) 


    def forward(self, x, temb):
        h = x 
        h = self.norm1(h) 
        h = nonlinearity(h) 
        h = self.conv1(h) 

        temb_embedding = self.temb_proj(nonlinearity(temb))
        if False: 
            scale, shift = torch.chunk(temb_embedding, 2, dim=1)
            h = h * scale[:, :, None, None] + shift[:, :, None, None] 
        else:
            h = h + temb_embedding[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)#

        if self.in_channels != self.out_channels: 
            use_conv_shortcut = 0
            if use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x) 
        else:
            shortcut_type = 0
            if shortcut_type==0:
                x = x
            elif shortcut_type==1:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h 


class AttnBlock(nn.Module): 
    def __init__(self, in_channels, attn_num=1):
        super().__init__()
        self.in_channels = in_channels
        self.attn_num = attn_num

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0) 
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0) 

    def forward(self, x):
        if self.attn_num==0:
            return x 

        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_) 
        assert q.size(1) % self.attn_num == 0

        # compute attention
        b, c, h, w = q.shape
        ch = c // self.attn_num
        q = q.reshape(b*self.attn_num, ch, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b*self.attn_num, ch, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw   
        w_ = w_ * (int(c)**(-0.5)) 
        w_ = torch.nn.functional.softmax(w_, dim=2) 

        # attend to values
        v = v.reshape(b*self.attn_num, ch, h*w) 
        w_ = w_.permute(0, 2, 1)  
        h_ = torch.bmm(v, w_) 
        h_ = h_.reshape(b, c, h, w) 

        h_ = self.proj_out(h_) 

        return x+h_ 


class SimpleNet(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config 
        ch, out_ch, ch_mult = config.supernet.ch, config.supernet.out_ch, tuple(config.supernet.ch_mult) 
        num_res_blocks = config.supernet.num_res_blocks 
        num_res_blocks_mid = config.supernet.num_res_blocks_mid 
        dropout = config.supernet.dropout 
        in_channels = config.supernet.in_channels 
        resolution = config.data.image_size 
        num_attn_blocks = config.supernet.num_attn_blocks 
        resamp_with_conv = config.supernet.resamp_with_conv 
        num_timesteps = config.diffusion.num_diffusion_timesteps 
        
        self.ch = ch
        self.temb_ch = config.supernet.time_embedding 
        self.num_resolutions = len(ch_mult) 
        self.num_res_blocks = num_res_blocks
        self.num_res_blocks_mid = num_res_blocks_mid 
        self.resolution = resolution
        self.in_channels = in_channels
        self.num_attn_blocks = num_attn_blocks


        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),                                                        
        ])
        

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, 
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1) 

        curr_res = resolution 
        in_ch_mult = (128,)+ch_mult 
        self.down = nn.ModuleList() 
        block_in = None 
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList() 
            AttnSequence = nn.ModuleList()
            block_in = in_ch_mult[i_level] 
            block_out = ch_mult[i_level] 
            for i_block in range(self.num_res_blocks): 
                block.append(ResnetBlock(in_channels=block_in, 
                                         out_channels=block_out, 
                                         temb_channels=self.temb_ch, 
                                         dropout=dropout)) 

                block_in = block_out 
                attn = nn.Module()
                attn.attn_head = nn.ModuleList()
                for i_attn in range(self.num_attn_blocks):
                    attn.attn_head.append(AttnBlock(block_in)) 
                AttnSequence.append(attn)
            down = nn.Module()
            down.block = block 
            down.AttnSequence = AttnSequence 
            down.downsample = Downsample(block_in, resamp_with_conv)
            curr_res = curr_res // 2 
            self.down.append(down) 

        # middle
        self.mid = nn.Module()
        block = nn.ModuleList()
        AttnSequence = nn.ModuleList()
        for i_block in range(self.num_res_blocks_mid):
            block.append(ResnetBlock(in_channels=block_in,
                                     out_channels=block_in,
                                     temb_channels=self.temb_ch,
                                     dropout=dropout))
            attn = nn.Module()
            attn.attn_head = nn.ModuleList()
            for i_attn in range(self.num_attn_blocks):
                attn.attn_head.append(AttnBlock(block_in))
            AttnSequence.append(attn)
        self.mid.block = block
        self.mid.AttnSequence = AttnSequence

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)): 
            block = nn.ModuleList() 
            AttnSequence = nn.ModuleList()
            block_out = ch_mult[i_level] 
            skip_in = ch_mult[i_level] 
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = in_ch_mult[i_level] 
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout)) 
                block_in = block_out
                attn = nn.Module()
                attn.attn_head = nn.ModuleList()
                for i_attn in range(self.num_attn_blocks):
                    attn.attn_head.append(AttnBlock(block_in)) 
                AttnSequence.append(attn)
            up = nn.Module()
            up.block = block
            up.AttnSequence = AttnSequence
            up.upsample = Upsample(block_in, resamp_with_conv)
            curr_res = curr_res * 2 
            self.up.insert(0, up)  

        # end
        self.norm_out = torch.nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1) 

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch) 
        for i in range(4):
            temb = self.temb.dense[i](temb) 
            if i != 3:
                temb = nonlinearity(temb)

        # downsampling
        hs = [self.conv_in(x)] 
        for i_level in range(self.num_resolutions): 
            for i_block in range(self.num_res_blocks): 
                h = self.down[i_level].block[i_block](hs[-1], temb)
                for i_attn in range(self.num_attn_blocks): 
                    h = self.down[i_level].AttnSequence[i_block].attn_head[i_attn](h) 
                hs.append(h) 
            if i_level != self.num_resolutions-1: 
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1] 
        for i_block in range(self.num_res_blocks_mid):
            h = self.mid.block[i_block](h, temb) 
            for i_attn in range(self.num_attn_blocks):
                h = self.down[i_level].AttnSequence[i_block].attn_head[i_attn](h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop().to(h.device)], dim=1), temb) 
                for i_attn in range(self.num_attn_blocks):
                    h = self.up[i_level].AttnSequence[i_block].attn_head[i_attn](h) 
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) 
        return h

