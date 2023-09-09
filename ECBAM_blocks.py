import torch
from torch import nn
import math
from torch.nn import functional as F

class ECALayer(nn.Module):
    """ECA layer implementation."""
    def __init__(self):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        m = self.max_pool(x)
        m = self.conv(m.squeeze(-1).transpose(-1, -2))
        m = m.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y+m)
        return y.expand_as(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.LeakyReLU()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print("空间平均注意力", avg_out.shape)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print("空间最大注意力", max_out.shape)
        x = torch.cat([avg_out, max_out], dim=1)
        # print(x.shape)
        x = self.conv1(x)
        return self.sigmoid(x)


class ECBAM(nn.Module):
    def __init__(self):
        super(ECBAM, self).__init__()
        self.ca = ECALayer()
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


if __name__ == '__main__':
    img = torch.randn(1, 198, 100, 100)   # (16,32,20,20)
    net = ECBAM()
    print(net)
    out = net(img)
    print(out.size())