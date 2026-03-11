import functools

import torch
import torch.nn as nn

from snpgen.models.modules.utils import Conv1dSamePadding

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .permute(1, 0, 2)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1)
        return h



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_ch=3, ndf=64, depth=3, kernel_size=4, use_actnorm=False, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_ch (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            depth (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        self.kernel_size = kernel_size
        
        if not use_actnorm:
            norm_layer = nn.BatchNorm1d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm1d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm1d
        else:
            use_bias = norm_layer != nn.BatchNorm1d

        kw = kernel_size
        padw = 'same' if kw > 4 else 1
        sequence = [
            Conv1dSamePadding(input_ch, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, depth):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                Conv1dSamePadding(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**depth, 8)
        sequence += [
            Conv1dSamePadding(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            Conv1dSamePadding(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


# Adapted from https://github.com/Stability-AI/StableCascade/blob/7a7d341f729ccaa042920a1fac3e7b9326079aca/modules/stage_a.py#L118
class WDiscriminator(nn.Module):
    def __init__(self, input_ch=3, c_cond=0, c_hidden=512, depth=6, kernel_size=3, sigmoid=False, **kwargs):
        super().__init__()
        self.kernel_size = kernel_size
        d = max(depth - 3, 3)
        padding = 'same' if kernel_size > 3 else 1

        layers = [
            nn.utils.spectral_norm(Conv1dSamePadding(input_ch, c_hidden // (2 ** d), kernel_size=kernel_size, stride=2, padding=padding)),
            nn.LeakyReLU(0.2),
        ]
        for i in range(depth - 1):
            input_ch = c_hidden // (2 ** max((d - i), 0))
            c_out = c_hidden // (2 ** max((d - 1 - i), 0))
            layers.append(nn.utils.spectral_norm(Conv1dSamePadding(input_ch, c_out, kernel_size=kernel_size, stride=2, padding=padding)))
            layers.append(nn.InstanceNorm1d(c_out))
            layers.append(nn.LeakyReLU(0.2))
        self.encoder = nn.Sequential(*layers)
        self.shuffle = nn.Conv1d((c_hidden + c_cond) if c_cond > 0 else c_hidden, 1, kernel_size=1)
        self.output_fn = nn.Sigmoid() if sigmoid else nn.Identity()

    def forward(self, x, cond=None):
        x = self.encoder(x)
        if cond is not None:
            cond = cond.view(cond.size(0), cond.size(1), 1, 1, ).expand(-1, -1, x.size(-2), x.size(-1))
            x = torch.cat([x, cond], dim=1)
        x = self.shuffle(x)
        x = self.output_fn(x)
        return x
