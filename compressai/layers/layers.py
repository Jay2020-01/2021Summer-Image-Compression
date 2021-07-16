import mindspore.nn as nn
import mindspore.ops as ops

from .gdn import GDN


class MaskedConv2d(nn.Conv2d):
    r"""Masked 2D convolution implementation, mask future "unseen" pixels.
    Useful for building auto-regressive network components.

    Introduced in `"Conditional Image Generation with PixelCNN Decoders" <https://arxiv.org/abs/1606.05328>`_.

    Inherits the same arguments as a `nn.Conv2d`. Use `mask_type='A'` for the
    first layer (which also masks the "current pixel"), `mask_type='B'` for the following
    layers.
    """
    def __init__(self, *args, mask_type='A', **kwargs):
        super().__init__(*args, **kwargs)

        if mask_type not in ('A', 'B'):
            raise ValueError(f'Invalid "mask_type" value "{mask_type}"')

        # self.register_buffer('mask', torch.ones_like(self.weight.data))
        # change
        self.register_buffer('mask', ops.OnesLike(self.weight.data))
        _, _, h, w = self.mask.size()
        self.mask[:, :, h // 2, w // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, h // 2 + 1:] = 0

    def forward(self, x):
        # TODO(begaintj): weight assigment is not supported by torchscript
        # original TODO, not by J
        self.weight.data *= self.mask
        return super().forward(x)


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    # return nn.Sequential(
    #     nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1),
    #     # TODO BY J : no pixel shuffle in mindspore
    #     nn.PixelShuffle(r))
    # change
    return nn.SequentialCell([nn.Conv2d(in_ch, out_ch * r**2, kernel_size=3, padding=1),
                              #     # TODO BY J : no pixel shuffle in mindspore
                              #     nn.PixelShuffle(r))
                              ])


def conv1x1(in_ch, out_ch, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Cell):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
        #change
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1:
            self.downsample = conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Cell):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """
    def __init__(self, in_ch, out_ch, upsample=2):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
        # change
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Cell):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        # self.leaky_relu = nn.LeakyReLU(inplace=True)
        # change
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(out_ch, out_ch)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class AttentionBlock(nn.Cell):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """
    def __init__(self, N):
        super().__init__()

        class ResidualUnit(nn.Cell):
            """Simple residual unit."""
            def __init__(self):
                super().__init__()
                # self.conv = nn.Sequential(
                #     conv1x1(N, N // 2),
                #     nn.ReLU(inplace=True),
                #     conv3x3(N // 2, N // 2),
                #     nn.ReLU(inplace=True),
                #     conv1x1(N // 2, N),
                # )
                # change
                self.conv = nn.SequentialCell([
                    conv1x1(N, N // 2),
                    nn.ReLU(),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(),
                    conv1x1(N // 2, N),
                ])
                # self.relu = nn.ReLU(inplace=True)
                # change
                self.relu = nn.ReLU()

            def forward(self, x):
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        # self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(),
        #                             ResidualUnit())
        #
        # self.conv_b = nn.Sequential(
        #     ResidualUnit(),
        #     ResidualUnit(),
        #     ResidualUnit(),
        #     conv1x1(N, N),
        # )
        # change

        self.conv_a = nn.SequentialCell([ResidualUnit(), ResidualUnit(),
                                    ResidualUnit()])

        self.conv_b = nn.SequentialCell([
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        ])

    def forward(self, x):
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        # out = a * torch.sigmoid(b)
        # change
        out = a * nn.Sigmoid(b)
        out += identity
        return out