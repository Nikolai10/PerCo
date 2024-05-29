# Copyright 2024 Nikolai Körber. All Rights Reserved.
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

"""Hyper-encoder as described in https://arxiv.org/abs/2309.15505 (Sec. 3.2)."""

import torch
import torch.nn as nn
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from dataclasses import dataclass

from compressai.layers import AttentionBlock, conv3x3
from compressai.models.utils import conv, deconv


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBottleneckBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int):
        super().__init__()

        self.layers = nn.Sequential(
            conv1x1(in_ch, in_ch // 2),
            nn.ReLU(inplace=True),
            conv3x3(in_ch // 2, in_ch // 2),
            nn.ReLU(inplace=True),
            conv1x1(in_ch // 2, in_ch),
        )

    def forward(self, image: Tensor) -> Tensor:
        return image + self.layers(image)


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

    Args:
        in_channels: shape of the input tensor
        N: see code below
        M: see code below 
        codebook_dim: output dimension of hyper-encoder (backbone)
        cfg_ss: VQ spatial size
        cfg_cs: VQ codebook size
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            in_channels: int = 4,
            N=192,
            M=320,
            codebook_dim=32,
            cfg_ss=64,
            cfg_cs=256,
    ):
        super().__init__()

        vq_spatialdim = cfg_ss
        self.backbone = nn.Sequential(
            conv1x1(in_channels, N) if vq_spatialdim >= 16 else conv(in_channels, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, N) if vq_spatialdim >= 32 else conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            AttentionBlock(N),
            conv1x1(N, N) if vq_spatialdim == 64 else conv(N, N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            ResidualBottleneckBlock(N),
            conv1x1(N, M),
            AttentionBlock(M)
        )

        # VQ
        self.quantizer = VectorQuantize(dim=M,
                                        codebook_size=cfg_cs,
                                        use_cosine_sim = True,
                                        codebook_dim=codebook_dim)

    def forward(self, z: Tensor) -> Tensor:
        z = self.backbone(z)

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
