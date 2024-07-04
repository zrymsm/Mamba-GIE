# --------------------------------------------------------
# SDFF-References：DEA-Net: Single image dehazing based on detail enhanced convolution and content-guided attention
# Link: https://github.com/cecret3350/DEA-Net
# --------------------------------------------------------

import torch
from torch import nn
from einops.layers.torch import Rearrange

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(1)  # 添加BatchNorm层
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim // reduction),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=False),
            nn.BatchNorm2d(dim)  # 添加BatchNorm层
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class PixelAttention(nn.Module):
    def __init__(self, dim):
        super(PixelAttention, self).__init__()
        self.pa2 = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=False),
            nn.BatchNorm2d(dim)  # 添加BatchNorm层
        )
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, pattn1):
        B, C, H, W = x.shape
        x = x.unsqueeze(dim=2)
        pattn1 = pattn1.unsqueeze(dim=2)
        x2 = torch.cat([x, pattn1], dim=2)
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class SDFF(nn.Module):
    def __init__(self, dim, reduction=8):
        super(SDFF, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PixelAttention(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(2 * dim, 2 * dim, 1, bias=False),
            nn.BatchNorm2d(2 * dim)  # 添加BatchNorm层
        )
        self.sigmoid = nn.Sigmoid()

        


    def forward(self, x, y):
        
        initial_add = x + y
        cattn = self.ca(initial_add)
        sattn = self.sa(initial_add)
        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial_add, pattn1))

        #Remove skip
        result0 =  pattn2 * x
        result1 =  (1 - pattn2) * y

        #Cat+Conv
        result = torch.cat([result0, result1], dim=1)
        result = self.conv(result)


        

        return result






