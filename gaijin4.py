from torch import nn
import torch
import torch.nn.functional as F
from CBAM_blocks import ChannelAttention, SpatialAttention, CBAM
from ECBAM_blocks import ECBAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PGMSU(nn.Module):
    def __init__(self, P, Channel, z_dim, col):
        super(PGMSU, self).__init__()
        self.P = P
        self.Channel = Channel
        self.col = col
        self.z_dim = z_dim
        # encoder_m 端元解混网络编码器
        self.layer1 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.cbam1 = ECBAM(Channel)

        self.layer3 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.layer4 = nn.Conv2d(Channel, z_dim, kernel_size=3, stride=1, padding=1)
        self.layer5 = nn.Conv2d(Channel, z_dim, kernel_size=3, stride=1, padding=1)

        # 丰度编码器
        # encoder_m 端元解混网络编码器
        self.layer6 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.cbam2 = ECBAM(Channel)

        self.layer8 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(Channel, Channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
            nn.LeakyReLU(0.2),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(Channel, P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(Channel),
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(z_dim, 16 * P, kernel_size=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(0.2)
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(16 * P, 64 * P, kernel_size=1),
            nn.BatchNorm2d(64 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer23 = nn.Sequential(
            nn.Conv2d(64 * P, Channel * P, kernel_size=1),
        )

    def encoder_m(self, x):
        # resnet
        h1 = self.layer1(x)
        h1 = self.layer2(h1)
        h1 = self.cbam2(h1)
        h = h1 + x
        h1 = self.layer3(h)
        mu = self.layer4(h1)
        log_var = self.layer5(h1)
        return mu, log_var

    # 对端元加偏置
    def reparameterize(self, mu, log_var):  # 对端元
        std = (log_var * 0.01).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def encoder_a(self, x):
        h = self.layer6(x)
        h = self.layer7(h)
        h = self.cbam2(h)
        h = h + x
        h = self.layer8(h)
        h = self.layer9(h)
        print(h.shape)
        a = self.layer10(h)
        a = F.softmax(a, dim=1)
        return a

    def decoder(self, z):
        h1 = self.layer11(z)
        h1 = self.layer12(h1)
        h1 = self.layer13(h1)
        em = torch.sigmoid(h1)  # 将端元约束到0-1内
        return em

    def forward(self, input):
        mu, log_var = self.encoder_m(input)
        z = self.reparameterize(mu, log_var)
        a = self.encoder_a(input)

        em = self.decoder(z)
        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor  # @ 矩阵乘法
        y_hat = torch.squeeze(y_hat, dim=1)

        a = a.permute(2, 3, 0, 1)
        a = a.reshape(self.col * self.col, self.P)
        # mu = mu.permute(2, 3, 0, 1)
        # mu = mu.reshape(self.col * self.col, self.z_dim)
        #
        # log_var = log_var.permute(2, 3, 0, 1)
        # log_var = log_var.reshape(self.col * self.col, self.z_dim)
        return y_hat, mu, log_var, a, em_tensor


if __name__ == '__main__':
    P, Channel, z_dim = 5, 200, 4
    col = 10
    device = 'cpu'
    model = PGMSU(P, Channel, z_dim, col)
    input = torch.randn(1, Channel, col, col)
    y_hat, mu, log_var, a, em_tensor = model(input)
    print('shape of y_hat: ', y_hat.shape)
    print(mu.shape)
    print(log_var.shape)
    print(a.shape)
    print(em_tensor.shape)

