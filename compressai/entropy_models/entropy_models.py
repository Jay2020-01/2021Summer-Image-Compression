import os
import sys
sys.path.insert(0, "../../")
print("当前的工作目录：",os.getcwd())
print("python搜索模块的路径集合",sys.path)

import numpy as np
import scipy.stats

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as nps
from mindspore.common.initializer import initializer, Zero, One

from compressai._CXX import \
    pmf_to_quantized_cdf as _pmf_to_quantized_cdf  # pylint: disable=E0611,E0401
from compressai.ops import LowerBound


class _EntropyCoder:
    """Proxy class to an actual entropy coder class.
    """
    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders
        if method not in available_entropy_coders():
            methods = ', '.join(available_entropy_coders())
            raise ValueError(f'Unknown entropy coder "{method}"'
                             f' (available: {methods})')

        if method == 'ans':
            from compressai import ans  # pylint: disable=E0611
            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == 'rangecoder':
            import range_coder  # pylint: disable=E0401
            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder
    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf, precision=16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    # cdf = torch.IntTensor(cdf)
    # change
    cdf = Tensor(cdf, dtype=mindspore.int32)
    return cdf



class EntropyModel(nn.Cell):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """
    def __init__(self,
                 likelihood_bound=1e-9,
                 entropy_coder=None,
                 entropy_coder_precision=16):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        # self.register_buffer('_offset', torch.IntTensor())
        # self.register_buffer('_quantized_cdf', torch.IntTensor())
        # self.register_buffer('_cdf_length', torch.IntTensor())
        # change TODO BY J: no register in mindspore
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def construct(self, *args):
        raise NotImplementedError()

    # @torch.jit.unused
    def _get_noise_cached(self, x):
        # use simple caching method to avoid creating a new tensor every call
        half = float(0.5)
        if not hasattr(self, '_noise'):
            setattr(self, '_noise', x.new(x.size()))
        self._noise.resize_(x.size())
        self._noise.uniform_(-half, half)
        return self._noise

    def _quantize(self, inputs, mode, means=None):
        # type: (Tensor, str, Optional[Tensor]) -> Tensor
        if mode not in ('noise', 'dequantize', 'symbols'):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == 'noise':
            # if torch.jit.is_scripting():
            #     half = float(0.5)
            #     noise = torch.empty_like(inputs).uniform_(-half, half)
            # else:
            #     noise = self._get_noise_cached(inputs)
            # change TODO BY J: no torch.jit.is_scripting()
            noise = self._get_noise_cached(inputs)
            inputs = inputs + noise
            return inputs

        outputs = inputs
        if means is not None:
            outputs -= means

        # outputs = torch.round(outputs)
        # change
        round = ops.Round()
        # outputs = round(outputs) TODO BY J: ops.Round is not support

        if mode == 'dequantize':
            if means is not None:
                outputs += means
            return outputs

        assert mode == 'symbols', mode
        outputs = outputs.int()
        return outputs

    @staticmethod
    def _dequantize(inputs, means=None):
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.float()
        return outputs

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        # cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        # change
        zeros = ops.Zeros()
        cdf = zeros((len(pmf_length), max_length + 2), dtype=mindspore.int32)
        for i, p in enumerate(pmf):
            # prob = torch.cat((p[:pmf_length[i]], tail_mass[i]), dim=0)
            # change
            concat = ops.Concat()
            prob = concat((p[:pmf_length[i]], tail_mass[i]), axis=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, :_cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError('Uninitialized CDFs. Run update() first')

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f'Invalid CDF size {self._quantized_cdf.size()}')

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError('Uninitialized offsets. Run update() first')

        if len(self._offset.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._offset.size()}')

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError('Uninitialized CDF lengths. Run update() first')

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f'Invalid offsets size {self._cdf_length.size()}')

    # 未用, 未修改
    # def compress(self, inputs, indexes, means=None):
    #     """
    #     Compress input tensors to char strings.
    #
    #     Args:
    #         inputs (torch.Tensor): input tensors
    #         indexes (torch.IntTensor): tensors CDF indexes
    #         means (torch.Tensor, optional): optional tensor means
    #     """
    #     symbols = self._quantize(inputs, 'symbols', means)
    #
    #     if len(inputs.size()) != 4:
    #         raise ValueError('Invalid `inputs` size. Expected a 4-D tensor.')
    #
    #     if inputs.size() != indexes.size():
    #         raise ValueError(
    #             '`inputs` and `indexes` should have the same size.')
    #
    #     self._check_cdf_size()
    #     self._check_cdf_length()
    #     self._check_offsets_size()
    #
    #     strings = []
    #     for i in range(symbols.size(0)):
    #         rv = self.entropy_coder.encode_with_indexes(
    #             symbols[i].reshape(-1).int().tolist(),
    #             indexes[i].reshape(-1).int().tolist(),
    #             self._quantized_cdf.tolist(),
    #             self._cdf_length.reshape(-1).int().tolist(),
    #             self._offset.reshape(-1).int().tolist())
    #         strings.append(rv)
    #     return strings
    #
    # #重点 均继承于此
    # def decompress(self, strings, indexes, means=None):
    #     """
    #     Decompress char strings to tensors.
    #
    #     Args:
    #         strings (str): compressed tensors
    #         indexes (torch.IntTensor): tensors CDF indexes
    #         means (torch.Tensor, optional): optional tensor means
    #     """
    #
    #     if not isinstance(strings, (tuple, list)):
    #         raise ValueError('Invalid `strings` parameter type.')
    #
    #     if not len(strings) == indexes.size(0):
    #         raise ValueError('Invalid strings or indexes parameters')
    #
    #     if len(indexes.size()) != 4:
    #         raise ValueError('Invalid `indexes` size. Expected a 4-D tensor.')
    #
    #     self._check_cdf_size()
    #     self._check_cdf_length()
    #     self._check_offsets_size()
    #
    #     if means is not None:
    #         if means.size()[:-2] != indexes.size()[:-2]:
    #             raise ValueError('Invalid means or indexes parameters')
    #         if means.size() != indexes.size() and \
    #                 (means.size(2) != 1 or means.size(3) != 1):
    #             raise ValueError('Invalid means parameters')
    #
    #     cdf = self._quantized_cdf
    #     outputs = cdf.new(indexes.size()) #是个函数，对应cdf的数据
    #
    #     for i, s in enumerate(strings):
    #         values = self.entropy_coder.decode_with_indexes(
    #             s, indexes[i].reshape(-1).int().tolist(), cdf.tolist(),
    #             self._cdf_length.reshape(-1).int().tolist(),
    #             self._offset.reshape(-1).int().tolist())
    #         outputs[i] = torch.Tensor(values).reshape(outputs[i].size())
    #     outputs = self._dequantize(outputs, means)
    #     return outputs


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`_
    for an introduction.
    """
    def __init__(self,
                 channels,
                 *args,
                 tail_mass=1e-9,
                 init_scale=10,
                 filters=(3, 3, 3, 3),
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)

        # Create parameters
        # self._biases = nn.ParameterList()
        # self._factors = nn.ParameterList()
        # self._matrices = nn.ParameterList()
        # change TODO BY J: not sure parameterlist->list
        self._biases = []
        self._factors = []
        self._matrices = []

        filters = (1, ) + self.filters + (1, )
        scale = self.init_scale**(1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            # matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            # matrix.data.fill_(init)
            # self._matrices.append(nn.Parameter(matrix))
            # change
            matrix = initializer(init, shape=(channels, filters[i + 1], filters[i]), dtype=mindspore.float32)
            self._matrices.append(mindspore.Parameter(matrix))

            # bias = torch.Tensor(channels, filters[i + 1], 1)
            # nn.init.uniform_(bias, -0.5, 0.5)
            # self._biases.append(nn.Parameter(bias))
            # change
            bias = np.random.uniform(0.5, -0.5, (channels, filters[i + 1], 1))
            bias = Tensor(bias, mindspore.float32)
            self._biases.append(mindspore.Parameter(bias))

            if i < len(self.filters):
                # factor = torch.Tensor(channels, filters[i + 1], 1)
                # nn.init.zeros_(factor)
                # self._factors.append(nn.Parameter(factor))
                # change
                zeros = ops.Zeros()
                factor = zeros((channels, filters[i + 1], 1), mindspore.float32)
                self._factors.append(mindspore.Parameter(factor))

        # self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        # init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        # self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)
        # change
        self.quantiles = mindspore.Parameter(Tensor(shape=(channels, 1, 3), dtype=mindspore.float32, init=Zero()))
        init = Tensor([-self.init_scale, 0, self.init_scale], mindspore.float32)
        # self.quantiles.data = init.repeat((self.quantiles.shape[0], 1, 1))
        self.quantiles = mindspore.Parameter(mindspore.numpy.repeat(init, self.quantiles.shape[0]).reshape((-1, 1, 3)))

        # target = np.log(2 / self.tail_mass - 1)
        # self.register_buffer('target', torch.Tensor([-target, 0, target]))
        # change
        target = np.log(2 / self.tail_mass - 1)
        self.target = Tensor([-target, 0, target], mindspore.float32)  # TODO BY J: no register in mindspore

    def _medians(self):
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # pylint: disable=E0203
            return

        medians = self.quantiles[:, 0, 1]

        # minima = medians - self.quantiles[:, 0, 0]
        # minima = torch.ceil(minima).int()
        # minima = torch.clamp(minima, min=0)
        #
        # maxima = self.quantiles[:, 0, 2] - medians
        # maxima = torch.ceil(maxima).int()
        # maxima = torch.clamp(maxima, min=0)
        # change

        minima = medians - self.quantiles[:, 0, 0]
        rint = ops.Rint()
        ceil = ops.Ceil()
        minima = rint(ops.ceil(minima))
        # minima = torch.clamp(minima, min=0)  TODO BY J: no clamp in mindspore

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = rint(ceil(maxima))
        # maxima = torch.clamp(maxima, min=0)  TODO BY J: no clamp in mindspore

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max()
        # samples = torch.arange(max_length)
        # change
        samples = Tensor(mindspore.numpy.arange(max_length), mindspore.float32)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        # sign = -torch.sign(lower + upper)
        # pmf = torch.abs(
        #     torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        sign = ops.Sign()
        abs = ops.Abs()
        sigmoid = ops.Sigmoid()
        sign = -sign(lower + upper)
        pmf = abs(
            sigmoid(sign * upper) - sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        # tail_mass = torch.sigmoid(lower[:, 0, :1]) +\
        #     torch.sigmoid(-upper[:, 0, -1:])
        # change
        tail_mass = sigmoid(lower[:, 0, :1]) + \
                    sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2

    def loss(self):
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        # loss = torch.abs(logits - self.target).sum()
        # change
        abs = ops.Abs()
        loss = abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs, stop_gradient):
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = self._matrices[i]
            # change no detach in mindspore
            # if stop_gradient:
            #     matrix = matrix.detach()
            # logits = torch.matmul(F.softplus(matrix), logits) #矩阵与input相乘，每一层乘在一起
            # change
            matmul = ops.Matmul()
            softplus = ops.Softplus()
            logits = matmul(softplus(matrix), logits)  # 矩阵与input相乘，每一层乘在一起

            bias = self._biases[i]
            # change no detach in mindspore
            # if stop_gradient:
            #     bias = bias.detach()
            logits += bias

            if i < len(self._factors):
                factor = self._factors[i]
                # change no detach in mindspore
                # if stop_gradient:
                #     factor = factor.detach()
                # logits += torch.tanh(factor) * torch.tanh(logits)
                # change
                tanh = ops.Tanh()
                logits += tanh(factor) * tanh(logits)
        return logits

    # 未用, 未修改
    # @torch.jit.unused
    # def _likelihood(self, inputs):
    #     half = float(0.5)
    #     v0 = inputs - half
    #     v1 = inputs + half
    #     lower = self._logits_cumulative(v0, stop_gradient=False)
    #     upper = self._logits_cumulative(v1, stop_gradient=False)
    #     sign = -torch.sign(lower + upper) #符号函数
    #     sign = sign.detach()
    #     likelihood = torch.abs(
    #         torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)) #得到类似正态的离散曲线，中心值和方差与图像内容有关
    #     return likelihood

    def construct(self, x):
        # Convert to (channels, ... , batch) format
        # x = x.permute(1, 2, 3, 0).contiguous()
        # shape = x.size()
        # values = x.reshape(x.size(0), 1, -1) #一维向量
        # change
        transpose = ops.Transpose()
        x = transpose(x, (1, 2, 3, 0))
        shape = x.shape
        values = x.reshape((x.shape[0], 1, -1))  # 一维向量

        # Add noise or quantize

        outputs = self._quantize(values,
                                 'noise' if self.training else 'dequantize',
                                 self._medians())

        # if not torch.jit.is_scripting():
        #     likelihood = self._likelihood(outputs)
        #     if self.use_likelihood_bound:
        #         likelihood = self.likelihood_lower_bound(likelihood)
        # else:
        #     # TorchScript not yet supported
        #     likelihood = torch.zeros_like(outputs)
        # change
        zerosLike = ops.ZerosLike()
        likelihood = zerosLike(outputs)


        # Convert back to input tensor shape
        # outputs = outputs.reshape(shape)
        # outputs = outputs.permute(3, 0, 1, 2).contiguous()
        #
        # likelihood = likelihood.reshape(shape)
        # likelihood = likelihood.permute(3, 0, 1, 2).contiguous()
        # change
        outputs = outputs.reshape(shape)
        transpose = ops.Transpose()
        outputs = transpose(outputs, (3, 0, 1, 2))

        likelihood = likelihood.reshape(shape)
        likelihood = transpose(likelihood, (3, 0, 1, 2))

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        N, C, H, W = size
        # indexes = torch.arange(C).view(1, -1, 1, 1)
        # change
        indexes = Tensor(np.arange(C)).reshape((1, -1, 1, 1))
        indexes = indexes.int()
        return indexes.repeat((N, 1, H, W))

    # 未用, 未修改
    # def compress(self, x):
    #     indexes = self._build_indexes(x.size())
    #     medians = self._medians().detach().view(1, -1, 1, 1)
    #     return super().compress(x, indexes, medians)   #默认index，指向当前像素通道号
    #
    # def decompress(self, strings, size):
    #     output_size = (len(strings), self._quantized_cdf.size(0), size[0],
    #                    size[1])
    #     indexes = self._build_indexes(output_size)  #熵编码（霍夫曼编码）的索引？字典索引，用来恢复图像（码字-》原信息）
    #     medians = self._medians().detach().view(1, -1, 1, 1)
    #     return super().decompress(strings, indexes, medians)


# 未用, 未修改
# class GaussianConditional(EntropyModel):
#     r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
#     S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
#     hyperprior" <https://arxiv.org/abs/1802.01436>`_.
#
#     This is a re-implementation of the Gaussian conditional layer in
#     *tensorflow/compression*. See the `tensorflow documentation
#     <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`_.
#     """
#     def __init__(self,
#                  scale_table,
#                  *args,
#                  scale_bound=0.11,
#                  tail_mass=1e-9,
#                  **kwargs):
#         super().__init__(*args, **kwargs)
#
#         if not isinstance(scale_table, (type(None), list, tuple)):
#             raise ValueError(
#                 f'Invalid type for scale_table "{type(scale_table)}"')
#
#         if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
#             raise ValueError(
#                 f'Invalid scale_table length "{len(scale_table)}"')
#
#         if scale_table and \
#                 (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
#             raise ValueError(f'Invalid scale_table "({scale_table})"')
#
#         self.register_buffer(
#             'scale_table',
#             self._prepare_scale_table(scale_table)
#             if scale_table else torch.Tensor())
#
#         self.register_buffer(
#             'scale_bound',
#             torch.Tensor([float(scale_bound)])
#             if scale_bound is not None else None)
#
#         self.tail_mass = float(tail_mass)
#         if scale_bound is None and scale_table:
#             self.lower_bound_scale = LowerBound(self.scale_table[0])
#         elif scale_bound > 0:
#             self.lower_bound_scale = LowerBound(scale_bound)
#         else:
#             raise ValueError('Invalid parameters')
#
#     @staticmethod
#     def _prepare_scale_table(scale_table):
#         return torch.Tensor(tuple(float(s) for s in scale_table))
#
#     def _standardized_cumulative(self, inputs):
#         # type: (Tensor) -> Tensor
#         half = float(0.5)
#         const = float(-(2**-0.5))
#         # Using the complementary error function maximizes numerical precision.
#         return half * torch.erfc(const * inputs)
#
#     @staticmethod
#     def _standardized_quantile(quantile):
#         return scipy.stats.norm.ppf(quantile)
#
#     def update_scale_table(self, scale_table, force=False):
#         # Check if we need to update the gaussian conditional parameters, the
#         # offsets are only computed and stored when the conditonal model is
#         # updated.
#         if self._offset.numel() > 0 and not force:
#             return
#         self.scale_table = self._prepare_scale_table(scale_table)
#         self.update()
#
#     def update(self):
#         multiplier = -self._standardized_quantile(self.tail_mass / 2)
#         pmf_center = torch.ceil(self.scale_table * multiplier).int()
#         pmf_length = 2 * pmf_center + 1
#         max_length = torch.max(pmf_length).item()
#
#         samples = torch.abs(
#             torch.arange(max_length).int() - pmf_center[:, None])
#         samples_scale = self.scale_table.unsqueeze(1)
#         samples = samples.float()
#         samples_scale = samples_scale.float()
#         upper = self._standardized_cumulative((.5 - samples) / samples_scale)
#         lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
#         pmf = upper - lower
#
#         tail_mass = 2 * lower[:, :1]
#
#         quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
#         quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
#                                          max_length)
#         self._quantized_cdf = quantized_cdf
#         self._offset = -pmf_center
#         self._cdf_length = pmf_length + 2
#
#     def _likelihood(self, inputs, scales, means=None):
#         # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
#         half = float(0.5)
#
#         if means is not None:
#             values = inputs - means
#         else:
#             values = inputs
#
#         scales = self.lower_bound_scale(scales)  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用
#
#         values = torch.abs(values)
#         upper = self._standardized_cumulative((half - values) / scales)
#         lower = self._standardized_cumulative((-half - values) / scales)
#         likelihood = upper - lower
#
#         return likelihood
#
#     def construct(self, inputs, scales, means=None):
#         # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
#         outputs = self._quantize(inputs,
#                                  'noise' if self.training else 'dequantize',
#                                  means)
#         likelihood = self._likelihood(outputs, scales, means)
#         if self.use_likelihood_bound:
#             likelihood = self.likelihood_lower_bound(likelihood)
#         return outputs, likelihood
#
#     def build_indexes(self, scales):
#         scales = self.lower_bound_scale(scales)
#         indexes = scales.new_full(scales.size(),
#                                   len(self.scale_table) - 1).int()
#         for s in self.scale_table[:-1]:
#             indexes -= (scales <= s).int()
#         return indexes  #统计数量


#GMM - ywz
class GaussianMixtureConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`_.
    """
    def __init__(self,
                 K,
                 scale_table=None,
                 mean_table=None,
                 weight_table=None,
                 *args,
                 scale_bound=0.11,
                 tail_mass=1e-9,
                 **kwargs):
        super().__init__(*args, **kwargs)

        #ywz for mixture numbers:K
        self.K = K

        if scale_table and \
                (scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        # self.register_buffer(
        #     'scale_table',
        #     self._prepare_scale_table(scale_table)
        #     if scale_table else torch.Tensor())
        #
        # self.register_buffer(
        #     'scale_bound',
        #     torch.Tensor([float(scale_bound)])
        #     if scale_bound is not None else None)
        # change TODO BY J:  no register in mindspore
        self.scale_table = self._prepare_scale_table(scale_table) if scale_table else None

        self.scale_bound = Tensor([float(scale_bound)], mindspore.float32) if scale_bound is not None else None

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            self.lower_bound_scale = LowerBound(self.scale_table[0])
        elif scale_bound > 0:
            self.lower_bound_scale = LowerBound(scale_bound)
        else:
            raise ValueError('Invalid parameters')

    @staticmethod
    def _prepare_scale_table(scale_table):
        # return torch.Tensor(tuple(float(s) for s in scale_table))
        # change
        return Tensor(tuple(float(s) for s in scale_table), mindspore.float32)

    def _standardized_cumulative(self, inputs):
        # type: (Tensor) -> Tensor
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        erfc = ops.Erfc() # TODO BY J: ops.Erfc not support
        return half * (const * inputs)

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return
        self.scale_table = self._prepare_scale_table(scale_table)
        self.update()

    def update(self):
        # multiplier = -self._standardized_quantile(self.tail_mass / 2)
        # pmf_center = torch.ceil(self.scale_table * multiplier).int()
        # pmf_length = 2 * pmf_center + 1
        # max_length = torch.max(pmf_length).item()
        # change
        multiplier = -self._standardized_quantile(self.tail_mass / 2)
        ceil = ops.Ceil()
        pmf_center = ceil(self.scale_table * multiplier).int()
        pmf_length = 2 * pmf_center + 1
        max_length = pmf_length.max()

        # samples = torch.abs(
        #     torch.arange(max_length).int() - pmf_center[:, None])
        # samples_scale = self.scale_table.unsqueeze(1)
        # samples = samples.float()
        # samples_scale = samples_scale.float()
        # upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        # lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        # pmf = upper - lower
        # change
        abs = ops.Abs()
        samples = abs(
            Tensor(mindspore.numpy.arange(max_length), mindspore.float32).int() - pmf_center[:, None])
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = Tensor(len(pmf_length), max_length + 2, mindspore.float32)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length,
                                         max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2


    #GMM
    def _likelihood(self, inputs, scales, means=None,weights=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor

        likelihood = None
        #获取M
        # M = inputs.size()[1] #通道数  除了inputs，另外三个的通道数均为M*K!!
        # change
        M = inputs.shape[1]  # 通道数  除了inputs，另外三个的通道数均为M*K!!

        # add_k  = 0
        for k in range(self.K):
            half = float(0.5)

            values = inputs - means[:,(M*k):((k+1)*M)]

            temp_scales = self.lower_bound_scale(scales[:,(M*k):((k+1)*M)])  #scales是方差，所以下方在标准正态基础上，直接/scales，起到方差作用

            # values = torch.abs(values)
            # change
            abs = ops.Abs()
            values = abs(values)
            upper = self._standardized_cumulative((half - values) / temp_scales)
            lower = self._standardized_cumulative((-half - values) / temp_scales)
            if likelihood==None:
                likelihood = (upper - lower)*weights[:,(M*k):((k+1)*M)] #指定对应的M个通道
            else:
                likelihood += (upper - lower)*weights[:,(M*k):((k+1)*M)] #指定对应的M个通道

            # if likelihood==None:
            #     likelihood = (upper - lower)*1/self.K #指定对应的M个通道
            # else:
            #     likelihood += (upper - lower)*1/self.K #指定对应的M个通道


        return likelihood

    #GMM
    def construct(self, inputs, scales, means=None,weights=None):
        # type: (Tensor, Tensor, Optional[Tensor]) -> Tuple[Tensor, Tensor]
        outputs = self._quantize(inputs,
                                 'noise' if self.training else 'dequantize',
                                 means=None) #changed: means=None
        #GMM
        likelihood = self._likelihood(outputs, scales, means,weights)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def build_indexes(self, scales):
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(),
                                  len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes  #统计数量


if __name__ == '__main__':
    # test1
    # a = EntropyBottleneck(128)
    # zeros = ops.Zeros()
    # x = zeros((16, 128, 4, 4), mindspore.float32)
    # out, likeli = a.construct(x)
    # print(out.shape, likeli.shape)

    # test2
    GMC = GaussianMixtureConditional(K = 5)
    zeros = ops.Zeros()
    y = zeros((16, 192, 16, 16), mindspore.float32)
    g1 = zeros((16, 960, 16, 16), mindspore.float32)
    g2 = zeros((16, 960, 16, 16), mindspore.float32)
    g3 = zeros((16, 960, 1, 1), mindspore.float32)
    y1_hat, y1_likelihoods = GMC.construct(y, g1, g2, g3)
    print(y1_hat.shape, y1_likelihoods.shape)
