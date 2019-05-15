# Stacked Hourglass Network based on github.com/princeton-vl/pose-hg-train.
# Copyright 2019 Daniel Rose.
# Licensed under the MIT license (http://opensource.org/licenses/MIT)
# This file may not be copied, modified, or distributed
# except according to those terms.

from torch import nn
import torch.nn.functional as F

from maskrcnn_benchmark.layers import BatchNorm2d
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.utils.registry import Registry


class HourglassNet(nn.Module):
    """ Stacked Hourglass Network. """
    def __init__(self, cfg, in_channels):
        super(HourglassNet, self).__init__()

        # Backbone body stages to be fed into the HGN. For FPN-backbones,
        # the body (e.g. R-50) outputs 4 stages, otherwise it outputs 1 stage.
        self.stages: [int] = cfg.MODEL.HGN.INPUT_STAGES
        self.num_stages = len(self.stages)

        # Number of hourglass modules in stack
        stack_size_for_stages: [int] = cfg.MODEL.HGN.STACK_SIZE_FOR_STAGES

        # Number of consecutive residual modules at each depth level
        num_modules_for_stages: [int] = cfg.MODEL.HGN.NUM_MODULES_FOR_STAGES

        # Number of nested Residuals per Hourglass
        hg_depth_for_stages: [int] = cfg.MODEL.HGN.HG_DEPTH_FOR_STAGES

        self.modules = []

        for i in range(self.num_stages):
            name = "stack" + str(i)
            module = HourglassStack(
                cfg,
                stack_size_for_stages[i],
                num_modules_for_stages[i],
                hg_depth_for_stages[i],
                in_channels * 2**self.stages[i]
            )
            self.add_module(name, module)
            self.modules.append(name)

    def forward(self, x):
        for i in range(self.num_stages):
            stage = self.stages[i]
            x[stage] = getattr(self, self.modules[i])(x[stage])
        return x


class HourglassStack(nn.Module):
    def __init__(self, cfg, stack_size, num_modules, hg_depth, in_channels):
        super(HourglassStack, self).__init__()

        self.stack = []

        for i in range(stack_size):
            name = "hg" + str(i)
            module = Hourglass(cfg, num_modules, hg_depth, in_channels)
            self.add_module(name, module)
            self.stack.append(name)

    def forward(self, x):
        for hg in self.stack:
            x = getattr(self, hg)(x)
        return x


class Hourglass(nn.Module):
    def __init__(self, cfg, num_modules, hg_depth, in_channels):
        super(Hourglass, self).__init__()

        # Translate string name to implementation
        residual_module = _RESIDUAL_MODULES[cfg.MODEL.HGN.RES_FUNC]

        self.modules = []

        for i in range(1, 4):
            name = "depth" + str(hg_depth) + "_res" + str(i)
            module = nn.Sequential(
                *[residual_module(in_channels, in_channels) for _ in range(num_modules)]
            )
            self.add_module(name, module)
            self.modules.append(name)

        # Construct hourglass waist recursively
        self.waist = (
            Hourglass(cfg, num_modules, hg_depth - 1, in_channels) if hg_depth > 1
            else nn.Sequential(
                *[residual_module(in_channels, in_channels) for _ in range(num_modules)]
            )
        )

    def forward(self, x):
        up1 = getattr(self, self.modules[0])(x)
        low1 = nn.MaxPool2d(kernel_size=2, stride=2)(getattr(self, self.modules[1])(x))
        low2 = self.waist(low1)
        up2 = F.interpolate(
            getattr(self, self.modules[2])(low2),
            size=(up1.shape[2], up1.shape[3])
        )
        return up1 + up2


class BaseResidual(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func):
        super(BaseResidual, self).__init__()

        self.conv_block = ConvBlock(in_channels, out_channels, norm_func)
        self.skip = (
            lambda x: x if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        skip = self.skip(x)
        x = self.conv_block(x)
        x += skip
        return x


class ResidualWithBN(BaseResidual):
    def __init__(self, in_channels, out_channels):
        super(ResidualWithBN, self).__init__(
            in_channels,
            out_channels,
            norm_func=BatchNorm2d
        )


class ResidualWithGN(BaseResidual):
    def __init__(self, in_channels, out_channels):
        super(ResidualWithGN, self).__init__(
            in_channels,
            out_channels,
            norm_func=group_norm
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_func):
        super(ConvBlock, self).__init__()

        self.norm1 = norm_func(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=1
        )
        self.norm2 = norm_func(out_channels // 2)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 2,
            out_channels=out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm3 = norm_func(out_channels // 2)
        self.conv3 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = F.relu_(self.norm1(x))
        x = self.conv1(x)
        x = F.relu_(self.norm2(x))
        x = self.conv2(x)
        x = F.relu_(self.norm3(x))
        x = self.conv3(x)
        return x


_RESIDUAL_MODULES = Registry({
    "ResidualWithBN": ResidualWithBN,
    "ResidualWithGN": ResidualWithGN,
})
