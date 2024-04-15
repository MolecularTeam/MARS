import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearWithConstraint(nn.Linear):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *config, max_norm=1, **kwconfig):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*config, **kwconfig)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class ConvSamePad2d(nn.Module):
    """
    extend nn.Conv2d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, bias=True):
        super(ConvSamePad2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        # self.bias = bias
        self.conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
            bias=bias)

    def forward(self, x):
        if type(self.kernel_size) != int:
            kernel_size = self.kernel_size
        else:
            kernel_size = (self.kernel_size, self.kernel_size)
        if type(self.stride) != int:
            stride = self.stride
        else:
            stride = (self.stride, self.stride)
        # net = x
        _, _, h, w = x.size()
        # Compute weight padding size
        out_dim = (w + stride[1] - 1) // stride[1]
        p = max(0, (out_dim - 1) * stride[1] + kernel_size[1] - w)
        pad_1 = p // 2
        pad_2 = p - pad_1
        w_pad_size = (pad_1, pad_2)
        # Compute height padding size
        out_dim = (h + stride[0] - 1) // stride[0]
        p = max(0, (out_dim - 1) * stride[0] + kernel_size[0] - h)
        pad_1 = p // 2
        pad_2 = p - pad_1
        h_pad_size = (pad_1, pad_2)
        # Pad
        x_pad = F.pad(x, w_pad_size + h_pad_size, "constant", 0)
        # Conv
        out = self.conv(x_pad)
        return out
