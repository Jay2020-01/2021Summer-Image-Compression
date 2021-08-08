import numpy as np
import pytest

import mindspore as msp

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import WithLossCell, Momentum
from mindspore.ops import composite as C

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

#####################

# self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')


class LowerBoundFunction(nn.Cell):
    def __init__(self):
        super(LowerBoundFunction, self).__init__()

    def construct(self, input_, bound):
        return msp.ops.maximum(input_, bound)

    def bprop(self, input_, bound, out, dout): #dout是梯度  out是推理值
        # pass_through_if = (input_ >= bound) | (dout < 0)
        pass_through_if = ((input_ >= bound).astype(input_.dtype) + (dout < 0).astype(input_.dtype)).astype('Bool')
        # print(pass_through_if)

        # print(out) #out为construct结果  #pytorch此处是已经求完导的值，但msp中是正向推导的值
        # print(dout) #dout貌似是梯度
        # print("---")
        print(pass_through_if.astype(dout.dtype) * dout)
        return pass_through_if.astype(dout.dtype) * dout, pass_through_if.astype(dout.dtype) * dout #第二个用不上


class LowerBound(nn.Cell):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.

    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    def __init__(self, bound):
        super().__init__()
        # self.register_buffer('bound', torch.Tensor([float(bound)]))
        # change TODO BY J: no register
        self.bound = Tensor([float(bound)])

    # @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)

    def forward(self, x):
        # change
        # if torch.jit.is_scripting():
        #     return torch.max(x, self.bound)
        return self.lower_bound(x)


if __name__=="__main__":
    grad_all = C.GradOperation(get_all=True)

    lowerboundfunc = LowerBoundFunction()
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([0, 1, 5]).astype(np.int32))
    test = lowerboundfunc(x, y)
    ret = grad_all(lowerboundfunc)(x, y)
    # print(ret)
    # print(lowerboundfunc)

    # p = (x >= y)
    # p.astype(x.dtype)