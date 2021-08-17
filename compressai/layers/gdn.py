# Copyright 2020 InterDigital Communications, Inc.
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

# import torch
import os
import sys
sys.path.insert(0, "../../")
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
# import torch.nn.functional as F

from compressai.ops.parametrizers import NonNegativeParametrizer





class GDN(nn.Cell):
    r"""Generalized Divisive Normalization layer.

    Introduced in `"Density Modeling of Images Using a Generalized Normalization
    Transformation" <https://arxiv.org/abs/1511.06281>`_,
    by Balle Johannes, Valero Laparra, and Eero P. Simoncelli, (2016).

    .. math::

       y[i] = \frac{x[i]}{\sqrt{\beta[i] + \sum_j(\gamma[j, i] * x[j]^2)}}

    """
    def __init__(self,
                 in_channels,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=0.1):
        super().__init__()

        beta_min = float(beta_min)
        gamma_init = float(gamma_init)
        self.inverse = bool(inverse)

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min)
        # beta = torch.ones(in_channels)
        # change
        ones = ops.Ones()
        beta = ones(in_channels, mindspore.float32)

        beta = self.beta_reparam.init(beta)
        # self.beta = nn.Parameter(beta)
        # change
        self.beta = mindspore.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer()
        # gamma = gamma_init * torch.eye(in_channels)
        # change
        eye = ops.Eye()
        gamma = gamma_init * eye(in_channels, in_channels, mindspore.float32)

        gamma = self.gamma_reparam.init(gamma)
        # self.gamma = nn.Parameter(gamma)
        # change
        self.gamma = mindspore.Parameter(gamma)

    def construct(self, x):
        _, C, _, _ = x.shape

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        # norm = F.conv2d(x**2, gamma, beta)
        # change # conv2d no beta
        conv2d = ops.Conv2D(out_channel=gamma.shape[0], kernel_size=gamma.shape[2])
        norm = conv2d(x**2, gamma)

        if self.inverse:
            # norm = torch.sqrt(norm)
            # change
            sqrt = ops.Sqrt()
            norm = sqrt(norm)
        else:
            # norm = torch.rsqrt(norm)
            # change
            sqrt = ops.Sqrt()
            norm = 1. / sqrt(norm)

        out = x * norm

        return out


class GDN1(GDN):
    r"""Simplified GDN layer.

    Introduced in `"Computationally Efficient Neural Image Compression" <http://arxiv.org/abs/1912.08771>`_,
    by Johnston, Nick, Elad Eban, Ariel Gordon, and Johannes Ballé, (2019).

    .. math::

        y[i] = \frac{x[i]}{\beta[i] + \sum_j(\gamma[j, i] * |x[j]|}

    """
    def construct(self, x):
        _, C, _, _ = x.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1, 1)
        # norm = F.conv2d(torch.abs(x), gamma, beta)
        # change # conv2d no beta
        conv2d = ops.Conv2D(out_channel=gamma.shape[0], kernel_size=gamma.shape[2])
        norm = conv2d(mindspore.Tensor.abs(x), gamma)

        if not self.inverse:
            norm = 1. / norm

        out = x * norm

        return out


if __name__ == '__main__':
    a = GDN(192)
    zeros = ops.Zeros()
    x = zeros((192,192,2,2), mindspore.float32)
    y = a.construct(x)
    print(y.shape)
