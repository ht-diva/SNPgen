import torch
import torch.nn as nn


class TinyEncoder(nn.Module):
    def __init__(self, input_ch=3, z_channels=1):
        super().__init__()
        self.input_ch = input_ch
        self.conv = nn.Conv1d(input_ch, 2 * z_channels, kernel_size=1)

    def forward(self, x):
        return torch.chunk(self.conv(x), 2, dim=1)


class TinyDecoder(nn.Module):
    def __init__(self, z_channels=1, out_ch=3):
        super().__init__()
        self.conv = nn.Conv1d(z_channels, out_ch, kernel_size=1)

    def get_last_layer(self):
        return self.conv.weight

    def forward(self, z, argmax=True):
        out = self.conv(z)
        if argmax:
            out = out.argmax(dim=1)
        return out
