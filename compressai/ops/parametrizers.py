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
# import torch.nn as nn
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops

from .bound_ops import LowerBound


class NonNegativeParametrizer(nn.Cell):
    """
    Non negative reparametrization.

    Used for stability during training.
    """
    def __init__(self, minimum=0, reparam_offset=2**-18):
        super().__init__()

        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)

        pedestal = self.reparam_offset**2
        # self.register_buffer('pedestal', torch.Tensor([pedestal]))
        # change TODO BY J: no register
        self.pedestal = Tensor([pedestal], mindspore.float32)
        bound = (self.minimum + self.reparam_offset**2)**.5
        self.lower_bound = LowerBound(bound)

    def init(self, x):
        sqrt = ops.Sqrt()
        max = ops.Maximum()
        return sqrt(max(x + self.pedestal, self.pedestal))

    def construct(self, x):
        out = self.lower_bound(x)
        out = out**2 - self.pedestal
        return out
