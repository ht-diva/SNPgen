import torch
import torch.nn as nn
import torch.nn.functional as F

from snpgen.models.modules.vae.layers import (
    MultiKernelConv1d,
    ResnetConv1dBlock, ResnetMultiKernelConv1dBlock,
    Upsample1d, Normalize, nonlinearity,
)

from snpgen.models.modules.attention import AttentionBlock
from snpgen.models.modules.utils import Conv1dSamePadding
   

########################################################################################################
#                                               Decoder                                                #
########################################################################################################
class Decoder(nn.Module):
    def __init__(self, *, z_channels, out_ch,
                 z_dim,
                 conv1d_channels=[128,256,512],
                 num_res_blocks=2,
                 attn_pos=[], # list of positions w.r.t. 'conv1d_channels' at which to add attention blocks
                 mid_attn=True,
                 multi_kernel=False,
                 dropout=0.0,
                 resamp_with_conv=True,
                 kernel_size=3,
                 **kwargs
                ):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.conv1d_channels = conv1d_channels
        self.num_resolutions = len(conv1d_channels)
        self.z_dim = z_dim
        self.dropout = dropout
        self.mid_attn = mid_attn

        block_in = conv1d_channels[-1]
        
        # z to block_in
        if multi_kernel:
            resnet_conv1d_block = ResnetMultiKernelConv1dBlock
            self.conv_in = MultiKernelConv1d(
                z_channels, block_in, kernels=[3,5,7], stride=1
            )
        else:
            resnet_conv1d_block = ResnetConv1dBlock
            self.conv_in = torch.nn.Conv1d(
                z_channels, block_in, kernel_size=3, stride=1, padding=1
            )

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
                channels=block_in, num_heads=1, num_head_channels=-1,
                use_checkpoint=False, dropout=0., xformers=False, use_bias=False
            )
        self.mid.block_2 = resnet_conv1d_block(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            kernel_size=3
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = conv1d_channels[i_level]
            for i_block in range(self.num_res_blocks + 1):
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
                            channels=block_in, num_heads=1, num_head_channels=-1,
                            use_checkpoint=False, dropout=0., xformers=False, use_bias=False
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample1d(block_in, with_conv=resamp_with_conv, kernel_size=kernel_size)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        conv1d = Conv1dSamePadding if kernel_size > 3 else torch.nn.Conv1d
        padding = 'same' if kernel_size > 3 else 1
        self.conv_out = conv1d(
            block_in, out_ch, kernel_size=kernel_size, stride=1, padding=padding
        )

    def get_cls_layer(self, **kwargs):
        ''' Returns the last layer weights to which the classification head is attached.'''
        return None
    
    def get_last_layer(self):
        ''' Returns the last layer weights.'''
        return self.conv_out.weight

    def forward(self, z, argmax=True):
        # z: bs, ch, h

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        if self.mid_attn:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h) # bs, ch_out, L

        if argmax:
            h = h.argmax(dim=1) # bs, L

        return h