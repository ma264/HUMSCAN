import torch
from torch import nn
import math
from torch.nn import functional as F


class ECALayer(nn.Module):
    """ECA layer implementation."""
    def __init__(self, kernel_size):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
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
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        # assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
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
    def __init__(self, kernel_size, kernel_size1):
        super(ECBAM, self).__init__()
        self.ca = ECALayer(kernel_size)
        self.sa = SpatialAttention(kernel_size1)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class MultiECBAM(nn.Module):
    def __init__(self, Channel):
        super(MultiECBAM, self).__init__()
        self.channel = Channel
        self.ecbam1 = nn.Sequential(
            ECBAM(1, 1),
        )
        self.ecbam2 = nn.Sequential(
            ECBAM(1, 3),
        )
        self.ecbam3 = nn.Sequential(
            ECBAM(1, 5),
        )
        self.ecbam4 = nn.Sequential(
            ECBAM(1, 7),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(Channel * 7, Channel, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        # a1 = self.ecbam1(x)
        # a2 = self.ecbam2(x)
        # a3 = self.ecbam3(x)
        # a4 = self.ecbam4(x)
        # y = torch.cat([a2, a3, a4, x], dim=1)
        # y = self.conv(y)

        a2 = self.ecbam2(x)
        m1 = torch.cat([a2, x], dim=1)
        a3 = self.ecbam3(m1)
        m2 = torch.cat([a3, x], dim=1)
        a4 = self.ecbam4(m2)
        y = torch.cat([a2, a3, a4, x], dim=1)
        y = self.conv(y)

        # a2 = self.ecbam2(x)
        # a3 = self.ecbam3(a2)
        # a4 = self.ecbam4(a3)
        # y = torch.cat([a2, a3, a4, x], dim=1)
        # y = self.conv(y)
        return y


if __name__ == '__main__':
    img = torch.randn(1, 198, 100, 100)   # (16,32,20,20)
    net = MultiECBAM(198)
    print(net)
    out = net(img)
    print(out.size())

