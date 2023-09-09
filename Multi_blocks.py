import torch
from torch import nn
import math
from torch.nn import functional as F


class Multi(nn.Module):
    """ECA layer implementation."""
    def __init__(self, input_Channel):
        super(Multi, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU(),
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU(),
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU(),
            nn.Conv2d(input_Channel, input_Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(input_Channel),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(input_Channel * 4, input_Channel, kernel_size=1),
        )

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        c = self.conv3(x)
        y = torch.cat([a, b, c, x], dim=1)
        y = self.conv4(y)
        return y


if __name__ == '__main__':
    img = torch.randn(1, 198, 100, 100)   # (16,32,20,20)
    net = Multi(198)
    print(net)
    out = net(img)
    print(out.size())
