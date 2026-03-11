
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from snpgen.models.modules.utils import get_num_groups, Conv1dSamePadding


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class MultiKernelConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
        **kwargs,
    ):
        super().__init__()

        kwargs.pop('padding', None) # remove padding from kwargs because it is automatically handled in the convs below
        kwargs.pop('kernel_size', None) # remove kernel_size from kwargs because it is automatically handled in the convs below

        self.out_channels_splits = self.split_out_channels(out_channels, len(kernels))

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=in_channels, out_channels=c, kernel_size=k, padding=k//2 if k % 2 != 0 else 0, **kwargs)
            for (k, c) in zip(kernels, self.out_channels_splits)
        ])

    def split_out_channels(self, out_channels, num_splits):
        split_size = out_channels // num_splits
        remainder = out_channels % num_splits

        splits = [split_size] * num_splits
        for i in range(remainder):
            splits[i] += 1

        return splits

    def forward(self, x):
        x = torch.cat([conv(x) for conv in self.convs], dim=1)
        return x


class MultiKernelConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
    ):
        super().__init__()
        
        num_groups = get_num_groups(in_channels, num_groups=32)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.gelu = nn.GELU()
        self.multi_kernel_conv = MultiKernelConv1d(in_channels=in_channels, out_channels=out_channels, kernels=kernels)

    def forward(self, x):
        x = self.norm(x)
        x = self.gelu(x)
        x = self.multi_kernel_conv(x)
        return x


class MultiKernelConvTransposeBlock(MultiKernelConvBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
    ):
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernels=kernels)

        self.convs = nn.ModuleList([
            nn.ConvTranspose1d(in_channels=in_channels, out_channels=c, kernel_size=k, padding=k//2 if k % 2 != 0 else 0)
            for (k, c) in zip(kernels, self.out_channels_splits)
        ])
    

class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
        depth: int = 2,
        pooling: bool = True,
    ):
        super().__init__()
        self.pooling = pooling

        self.conv_blocks = nn.ModuleList([
            MultiKernelConvBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernels=kernels)
            for i in range(depth)
        ])

        if self.pooling:
            self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        
        x = self.conv_blocks[0](x)

        for conv_block in self.conv_blocks[1:]:
            x = conv_block(x) + x

        if self.pooling:
            x = self.maxpool(x)

        return x
    

class ConvTranspose1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernels: list[int],
        depth: int = 2,
        upsample: bool = True,
    ):
        super().__init__()
        self.upsample = upsample

        self.conv_blocks = nn.ModuleList([
            MultiKernelConvTransposeBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernels=kernels)
            for i in range(depth)
        ])

        if self.upsample:
            self.upsample_ops = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        
        x = self.conv_blocks[0](x)

        for conv_block in self.conv_blocks[1:]:
            x = conv_block(x) + x

        if self.upsample:
            x = self.upsample_ops(x)

        return x
    

class Conv2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
    ):
        super().__init__()
        
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size//2 if kernel_size % 2 != 0 else 0
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x


class ConvTranspose2DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        output_padding: int = 1,
    ):
        super().__init__()
        
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride,
            padding=kernel_size//2 if kernel_size % 2 != 0 else 0,
            output_padding=output_padding
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.leaky_relu(x)
        return x
    

########################## SD Encoder/Decoder ##########################

def nonlinearity(x):
    # Swish / SiLU
    return x * torch.sigmoid(x)

class NonLinearity(nn.Module):
    def forward(self, x):
        return nonlinearity(x)
    
def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

class Downsample1d(nn.Module):
    def __init__(self, in_channels, with_conv, kernel_size=3):
        super().__init__()
        self.with_conv = with_conv
        self.kernel_size = kernel_size

        if self.with_conv:
            conv1d = Conv1dSamePadding if kernel_size > 3 else torch.nn.Conv1d
            padding = 'same' if self.kernel_size > 3 else 0
            self.conv = conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=2, padding=padding
            )

    def forward(self, x):
        if self.with_conv:
            if self.kernel_size <= 3:
                # no asymmetric padding in torch conv, must do it ourselves
                pad = (0, 1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            #x = torch.nn.functional.max_pool1d(x, kernel_size=2, stride=2)
            x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
        return x

class Upsample1d(nn.Module):
    def __init__(self, in_channels, with_conv, kernel_size=3):
        super().__init__()
        self.with_conv = with_conv
        self.kernel_size = kernel_size

        if self.with_conv:
            conv1d = Conv1dSamePadding if kernel_size > 3 else torch.nn.Conv1d
            padding = 'same' if self.kernel_size > 3 else 1
            self.conv = conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetConv1dBlock(nn.Module):
    def __init__(
        self,
        kernel_size=3,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.dropout = dropout
        self.kernel_size = kernel_size
        self._build_module()

    def _build_module(self):
        in_channels = self.in_channels
        out_channels = self.out_channels
        dropout = self.dropout
        kernel_size = self.kernel_size

        conv1d = Conv1dSamePadding if kernel_size > 3 else torch.nn.Conv1d
        padding = 'same' if self.kernel_size > 3 else 1

        self.norm1 = Normalize(in_channels)
        self.conv1 = conv1d(
            in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
        )

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = conv1d(
            out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
                )
            else:
                self.nin_shortcut = torch.nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    
class ResnetMultiKernelConv1dBlock(ResnetConv1dBlock):

    def _build_module(self):
        in_channels = self.in_channels
        out_channels = self.out_channels
        dropout = self.dropout
        kernels = [3,5,7]

        self.norm1 = Normalize(in_channels)
        self.conv1 = MultiKernelConv1d(
            in_channels, out_channels, kernels=kernels, stride=1
        )

        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = MultiKernelConv1d(
            out_channels, out_channels, kernels=kernels, stride=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = MultiKernelConv1d(
                    in_channels, out_channels, kernels=kernels, stride=1
                )
            else:
                self.nin_shortcut = MultiKernelConv1d(
                    in_channels, out_channels, kernels=[1], stride=1
                )