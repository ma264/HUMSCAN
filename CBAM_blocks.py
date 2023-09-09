import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out =self.shared_MLP(self.avg_pool(x))
        # print("平均池化", avg_out.shape)
        max_out =self.shared_MLP(self.max_pool(x))
        # print("最大池化", max_out.shape)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
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


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


if __name__ == '__main__':
    img = torch.randn(1, 198, 100, 100)   # (16,32,20,20)
    net = CBAM(198)
    print(net)
    out = net(img)
    print(out.size())
