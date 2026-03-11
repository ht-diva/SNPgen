
import torch
import torch.nn as nn
import torch.nn.functional as F

from snpgen.models.modules.vae.layers import (
    MultiKernelConv1d,
    ResnetConv1dBlock, ResnetMultiKernelConv1dBlock,
    Downsample1d, Normalize, nonlinearity,
)
from snpgen.models.modules.attention import AttentionBlock
from snpgen.models.modules.utils import Conv1dSamePadding


########################################################################################################
#                                               Encoder                                                #
########################################################################################################
class Encoder(nn.Module):
    def __init__(self, *, input_ch,
                 in_ch_proj_dim=128,
                 conv1d_channels=[128,256,512],
                 z_channels=4,
                 num_res_blocks=2,
                 mid_attn=True,
                 attn_pos=[], # list of positions w.r.t. 'conv1d_channels' at which to add attention blocks
                 multi_kernel=False,
                 dropout=0.0,
                 resamp_with_conv=True,
                 kernel_size=3
                ):
        super().__init__()

        self.input_ch = input_ch
        self.num_res_blocks = num_res_blocks
        self.conv1d_channels = conv1d_channels
        self.num_resolutions = len(conv1d_channels)
        self.mid_attn = mid_attn

        if multi_kernel:
            resnet_conv1d_block = ResnetMultiKernelConv1dBlock
            self.conv_in = MultiKernelConv1d(
                input_ch, in_ch_proj_dim, kernels=[3,5,7], stride=1
            )
        else:
            resnet_conv1d_block = ResnetConv1dBlock
            conv1d = Conv1dSamePadding if kernel_size > 3 else torch.nn.Conv1d
            padding = 'same' if kernel_size > 3 else 1
            self.conv_in = conv1d(
                input_ch, in_ch_proj_dim, kernel_size=kernel_size, stride=1, padding=padding
            )

        in_ch = [in_ch_proj_dim] + conv1d_channels
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = in_ch[i_level]
            block_out = conv1d_channels[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    resnet_conv1d_block(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        kernel_size=kernel_size
                    )
                )
                block_in = block_out
                if (i_level+1) in attn_pos:
                    attn.append(
                        AttentionBlock(
                            channels=block_in,
                            num_heads=1,
                            num_head_channels=-1,
                            use_checkpoint=False,
                            dropout=0.,
                            xformers=False,
                            use_bias=False,
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample1d(block_in, with_conv=resamp_with_conv, kernel_size=kernel_size)
            self.down.append(down)


        # middle
        self.mid = nn.Module()
        self.mid.block_1 = resnet_conv1d_block(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            kernel_size=3
        )
        if mid_attn:
            self.mid.attn_1 = AttentionBlock(
                channels=block_in, num_heads=1, num_head_channels=-1, use_checkpoint=False,
                dropout=0., xformers=False, use_bias=False
            )
        self.mid.block_2 = resnet_conv1d_block(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            kernel_size=3
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv1d(
            block_in,
            2 * z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )


    def forward(self, x):
        x = x.squeeze(-1) # bs, ch, L

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        if self.mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar