# Copyright 2024 Nikolai Körber. All Rights Reserved.
#
# Based on:
# https://github.com/facebookresearch/NeuralCompression/blob/main/neuralcompression/models/_hific_encoder_decoder.py,
# https://github.com/facebookresearch/NeuralCompression/blob/main/neuralcompression/layers/_channel_norm.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hyper-encoder similar to https://arxiv.org/abs/2309.15505 (Sec. 3.2) based on HiFiC"""

import math
import torch
import torch.nn as nn
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass


def calculate_padding(input_size, kernel_size, stride, dilation=1):
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding = max(0, (input_size - effective_kernel_size) % stride)
    return padding


class ChannelNorm2D(nn.Module):
    """
    Channel normalization layer.

    This implements the channel normalization layer as described in the
    following paper:

    High-Fidelity Generative Image Compression
    F. Mentzer, G. Toderici, M. Tschannen, E. Agustsson

    Using this layer provides more stability to model outputs when there is a
    shift in image resolutions between the training set and the test set.

    Args:
        input_channels: Number of channels to normalize.
        epsilon: Divide-by-0 protection parameter.
        affine: Whether to include affine parameters for the noramlized output.
    """

    def __init__(self, input_channels: int, epsilon: float = 1e-3, affine: bool = True):
        super().__init__()

        if input_channels <= 1:
            raise ValueError(
                "ChannelNorm only valid for channel counts greater than 1."
            )

        self.epsilon = epsilon
        self.affine = affine

        if affine is True:
            self.gamma = nn.Parameter(torch.ones(1, input_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, input_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        mean = torch.mean(x, dim=1, keepdim=True)
        variance = torch.var(x, dim=1, keepdim=True)

        x_normed = (x - mean) * torch.rsqrt(variance + self.epsilon)

        if self.affine is True:
            x_normed = self.gamma * x_normed + self.beta

        return x_normed


def _channel_norm_2d(input_channels, affine=True):
    return ChannelNorm2D(
        input_channels,
        affine=affine,
    )


class _ResidualBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding=padding),
            _channel_norm_2d(channels, affine=True),
        )

    def forward(self, x):
        return x + self.sequence(x)


@dataclass
class HyperEncoderOutput(BaseOutput):
    """
    The output of [`HyperEncoder`].

    Args:
        z (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Continous encoder output.
        z_hat (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            VQ quantized encoder output.
        indices (`torch.IntTensor` of shape `(batch_size, height, width)`):
            Indices of quantized encoder output.
        commit_loss (`torch.FloatTensor`):
            VQ-VAE commitment loss
    """

    z: torch.FloatTensor = None
    z_hat: torch.FloatTensor = None
    indices: torch.IntTensor = None
    commit_loss: torch.FloatTensor = None


# Note that in PerCo the bit-rate is controlled via upper bound, similar to Agustsson et al., ICCV 2019 
class HyperEncoder(ModelMixin, ConfigMixin):
    """PerCo Hyper-Encoder.

    Call with, e.g. target_rate=0.1250:

    # spatial size, codebook size
    cfg_ss, cfg_cs = cfg[target_rate]

    enc = HyperEncoder(cfg_ss=cfg_ss, cfg_cs=cfg_cs)

    # official configuration:
    # Hyper-encoder has been provided by the authors
    # 0.1250 (4,080,128) # 0.0019 (5,868,032)
    # paper reports incorrect number of params (4-8M; checked with authors)

    # our configuration
    # 0.1250 (6051592) # 0.0019 (20424328)
    
    Args:
        in_channels: shape of the input tensor
        num_residual_blocks: number of residual blocks
        num_filters_base: number of filters (residual depth)
        num_filters_out: number of output filters
        codebook_dim: output of hyper-encoder (low-dimensional -> improved VQ-GAN)
        cfg_ss: VQ spatial size
        cfg_cs: VQ codebook size
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            num_residual_blocks=9,
            num_filters_base=192,
            num_filters_out=320,
            codebook_dim=32,
            cfg_ss=64,
            cfg_cs=256,
    ):
        super().__init__()

        # project input to num_filters_base
        self.proj_in = nn.Sequential(
            _channel_norm_2d(in_channels, affine=True),
            nn.Conv2d(in_channels, num_filters_base, kernel_size=3, padding=1),
            _channel_norm_2d(num_filters_base, affine=True),
        )

        # residual blocks
        resid_blocks = []
        for _ in range(num_residual_blocks):
            resid_blocks.append(_ResidualBlock((num_filters_base)))
        self.resid_blocks = nn.Sequential(*resid_blocks)

        # downsampling if required
        down_blocks = []
        num_down = int(math.log(64 // cfg_ss, 2))
        num_filters_prev = num_filters_base
        for i in range(num_down):
            num_filters = num_filters_base * 2 ** (i + 1)

            # Calculate padding to maintain spatial dimensions
            padding = calculate_padding(input_size=num_filters_prev, kernel_size=3, stride=2)

            down_block = nn.Sequential(
                nn.Conv2d(
                    num_filters_prev,
                    num_filters,
                    kernel_size=3,
                    stride=2,
                    padding=padding),
                _channel_norm_2d(num_filters, affine=True),
                nn.ReLU(),
            )
            down_blocks.append(down_block)
            num_filters_prev = num_filters

        self.down_blocks = nn.Sequential(*down_blocks)

        # project to latent_dim
        self.proj_out = nn.Sequential(
            nn.Conv2d(num_filters_prev, num_filters_out, kernel_size=1, padding=0),
        )

        # VQ
        self.quantizer = VectorQuantize(dim=num_filters_out,
                                        codebook_size=cfg_cs,
                                        codebook_dim=codebook_dim)

    def forward(self, z: Tensor) -> Tensor:
        z = self.proj_in(z)
        z = z + self.resid_blocks(z)
        z = self.down_blocks(z)
        z = self.proj_out(z)

        # (B,C,H,W) -> (B,H,W,C)
        z_perm = z.permute(0, 2, 3, 1)
        b, h, w, c = z_perm.shape

        # (B,H*W,C)
        z_perm = z_perm.view(b, -1, c)
        # (B,H*W,C), (B,H*W)
        z_hat, indices, commit_loss = self.quantizer(z_perm)
        # (B,H,W,C)
        z_hat = z_hat.view(b, h, w, c)
        # (B,C,H,W)
        z_hat = z_hat.permute(0, 3, 1, 2)
        # (B,H,W)
        indices = indices.view(b, h, w)

        return HyperEncoderOutput(z=z, z_hat=z_hat, indices=indices, commit_loss=commit_loss)
