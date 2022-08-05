from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule
from typing import Optional


class attention2d(BaseModule):
    def __init__(
        self,
        ip_channels,
        squeeze_ratio,
        K,
        temperature,
        init_cfg: Optional[dict] = None,
    ):
        """
        k - number of kernels for the convolution layer - Usually fixed
        squeeze_ratio - From SE Net - From higher dimension to C -> C/r -> C
        """
        super(attention2d, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if ip_channels == 3:
            hidden_channels = K
        else:
            hidden_channels = (ip_channels * squeeze_ratio) + 1

        # Bias = False To reduce parameters ?
        # self.fc1 = nn.Conv2d(ip_channels, hidden_channels, 1, bias=False)
        # self.fc2 = nn.Conv2d(hidden_channels, K, 1, bias=True)
        self.fc1 = build_conv_layer(
            None, ip_channels, int(hidden_channels), 1, bias=False
        )
        self.fc2 = build_conv_layer(None, int(hidden_channels), K, 1, bias=True)
        self.temperature = temperature

    def update_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
        if self.temperature < 0:
            self.temperature = 1

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = x.squeeze()
        # temperature initially high to make it uniform
        return F.softmax(x / self.temperature, dim=1)


class Dynamic_Conv2D(BaseModule):
    def __init__(
        self,
        ip_channels,
        output_channels,
        kernel_size,
        squeeze_ratio=0.25,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        K=6,
        temperature=34,
        init_cfg: Optional[dict] = None,
    ):
        super(Dynamic_Conv2D, self).__init__(init_cfg)
        self.ip_channels = ip_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(self.ip_channels, squeeze_ratio, K, temperature)

        self.weight = nn.parameter.Parameter(
            data=torch.randn(
                K, output_channels, ip_channels // groups, kernel_size, kernel_size
            ),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.parameter.Parameter(data=torch.zeros(K, output_channels))
        else:
            self.bias = None

    def update_temperature(self):
        self.attention.update_temperature()

    def forward(self, x):
        sf_attention = self.attention(x)
        batch, ch, h, w = x.shape
        x = x.view(1, -1, h, w)  # x - 1, (B x ch), h, w
        # weight - K , (op_ch x ip_xh x k_size x k_size)
        weight = self.weight.view(self.K, -1)

        aggregated_weight = torch.mm(sf_attention, weight)
        aggregated_weight = aggregated_weight.view(
            batch * self.output_channels,
            self.ip_channels // self.groups,
            self.kernel_size,
            self.kernel_size,
        )
        if self.bias is not None:
            aggregate_bias = torch.mm(sf_attention, self.bias).view(-1)
            output = F.conv2d(
                x,
                weight=aggregated_weight,
                bias=aggregate_bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch,
            )
        else:
            output = F.conv2d(
                x,
                weight=aggregated_weight,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups * batch,
            )

        output = output.view(
            batch, self.output_channels, output.shape[-2], output.shape[-1]
        )

        return output
