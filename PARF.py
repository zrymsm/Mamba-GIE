# --------------------------------------------------------
# PARF-References：PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers
# Link: https://github.com/XuJiacong/PIDNet/blob/main/models/model_utils.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class PARF(nn.Module):
    def __init__(self, in_chan, mid_chan, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PARF, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan,
                      kernel_size=1, bias=False),
            BatchNorm(mid_chan)
        )
        self.f_y = nn.Sequential(
            nn.Conv2d(in_chan, mid_chan,
                      kernel_size=1, bias=False),
            BatchNorm(mid_chan)
        )
        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_chan, in_chan,
                          kernel_size=1, bias=False),
                BatchNorm(in_chan)
            )
        if after_relu:
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                            mode='bilinear', align_corners=False)
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))

        y = F.interpolate(y, size=[input_size[2], input_size[3]],
                          mode='bilinear', align_corners=False)

        #此处明确需要增强Low-level Feature
        x = sim_map * x + (1 - sim_map) * y

        x=x.permute(0, 2, 3, 1)

        return x


