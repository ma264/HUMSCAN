from torch import nn
import torch
import torch.nn.functional as F
from ECBAM1_blocks import MultiECBAM, ECBAM
from ECBAM_blocks import ECBAM
from CBAM_blocks import CBAM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PGMSU(nn.Module):
    def __init__(self, P, Channel, z_dim, col):
        super(PGMSU, self).__init__()
        self.P = P
        self.Channel = Channel
        self.col = col
        self.z_dim = z_dim
        # encoder z  端元解混网络编码器
        self.layer1 = nn.Sequential(
            nn.Conv2d(Channel, 32 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * P),
            nn.ReLU(),
        )
        # self.cbam1 = CBAM(32 * P)
        # self.cbam1 = ECBAM()
        self.cbam1 = MultiECBAM(32 * P)

        self.layer2 = nn.Sequential(
            nn.Conv2d(32 * P, 16 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * P),
            nn.ReLU(),
        )

        # self.cbam2 = CBAM(16 * P)
        # self.cbam2 = ECBAM()
        self.cbam2 = MultiECBAM(16 * P)

        self.layer3 = nn.Sequential(
            nn.Conv2d(16 * P, 4 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.ReLU(),
        )

        # self.cbam3 = CBAM(4 * P)
        # self.cbam3 = ECBAM()
        self.cbam3 = MultiECBAM(4 * P)

        self.layer4 = nn.Conv2d(4 * P, z_dim, kernel_size=1)
        self.layer5 = nn.Conv2d(4 * P, z_dim, kernel_size=1)
        # self.layer4 = nn.Linear(4 * P, z_dim)
        # self.layer5 = nn.Linear(4 * P, z_dim)

        # encoder_a
        self.layer6 = nn.Sequential(
            nn.Conv2d(Channel, 32 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * P),
            nn.ReLU(),
        )

        # self.cbam6 = CBAM(32 * P)
        # self.cbam6 = ECBAM()
        self.cbam6 = MultiECBAM(32 * P)

        self.layer7 = nn.Sequential(
            nn.Conv2d(32 * P, 16 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * P),
            nn.ReLU(),
        )

        # self.cbam7 = CBAM(16 * P)
        # self.cbam7 = ECBAM()
        self.cbam7 = MultiECBAM(16 * P)

        self.layer8 = nn.Sequential(
            nn.Conv2d(16 * P, 4 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.ReLU(),
        )

        # self.cbam8 = CBAM(4 * P)
        # self.cbam8 = ECBAM()
        self.cbam8 = MultiECBAM(4 * P)

        self.layer9 = nn.Sequential(
            nn.Conv2d(4 * P, 4 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4 * P),
            nn.ReLU(),
        )

        # self.cbam9 = CBAM(4 * P)
        # self.cbam9 = ECBAM()
        self.cbam9 = MultiECBAM(4 * P)

        # self.layer10 = nn.Linear(4 * P, P)
        self.layer10 = nn.Conv2d(4 * P, P, kernel_size=3, stride=1, padding=1)

        # decoder
        # self.layer11 = nn.Sequential(
        #     nn.Linear(z_dim, 16 * P),
        #     nn.BatchNorm1d(16 * P),
        #     nn.ReLU()
        # )
        #
        # self.layer12 = nn.Sequential(
        #     nn.Linear(16 * P, 64 * P),
        #     nn.BatchNorm1d(64 * P),
        #     nn.ReLU(),
        # )
        #
        # self.layer13 = nn.Sequential(
        #     nn.Linear(64 * P, Channel * P),
        # )

        self.layer11 = nn.Sequential(
            nn.Conv2d(z_dim, 16 * P, kernel_size=1),
            nn.BatchNorm2d(16 * P),
            nn.ReLU(),
        )

        self.layer12 = nn.Sequential(
            nn.Conv2d(16 * P, 64 * P, kernel_size=1),
            nn.BatchNorm2d(64 * P),
            nn.ReLU(),
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(64 * P, Channel * P, kernel_size=1),
            # nn.BatchNorm2d(Channel * P),
            # nn.ReLU()
        )

    def encoder_m(self, x):
        # resnet
        # h1 = self.layer1(x)
        # h2 = self.cbam1(h1)
        # h = h1 + h2
        # h1 = self.layer2(h)
        # h2 = self.cbam2(h1)
        # h = h1 + h2
        # h1 = self.layer3(h)
        # h2 = self.cbam3(h1)
        # h = h1 + h2

        h = self.layer1(x)
        h = self.cbam1(h)
        h = self.layer2(h)
        h = self.cbam2(h)
        h = self.layer3(h)
        h = self.cbam3(h)
        # h = h.permute(2, 3, 0, 1)
        # m, n, p, q = h.shape
        # h = torch.reshape(h, (m * n, p * q))
        mu = self.layer4(h)
        log_var = self.layer5(h)
        return mu, log_var

    # 对端元加偏置
    def reparameterize(self, mu, log_var):  # 对端元
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def encoder_a(self, x):
        # h1 = self.layer6(x)
        # h2 = self.cbam6(h1)
        # h = h1 + h2
        # h1 = self.layer7(h)
        # h2 = self.cbam7(h1)
        # h = h1 + h2
        # h1 = self.layer8(h)
        # h2 = self.cbam8(h1)
        # h = h1 + h2
        # h1 = self.layer9(h)
        # h2 = self.cbam9(h1)
        # h = h1 + h2

        # h = self.layer6(x)
        # h = self.cbam6(h)
        # h = self.layer7(h)
        # h = self.cbam7(h)
        # h = self.layer8(h)
        # h = self.cbam8(h)

        h = self.layer1(x)
        h = self.cbam1(h)
        h = self.layer2(h)
        h = self.cbam2(h)
        h = self.layer3(h)
        h = self.cbam3(h)
        h = self.layer9(h)
        h = self.cbam9(h)
        # h = h.permute(2, 3, 0, 1)
        # m, n, p, q = h.shape
        # h = torch.reshape(h, (m * n, p * q))
        a = self.layer10(h)
        a = F.softmax(a, dim=1)
        return a

    def decoder(self, z):
        # h1 = z.permute(2, 3, 0, 1)
        # m, n, p, q = h1.shape
        # h1 = h1.reshape(m * n, p * q)
        h1 = self.layer11(z)
        h1 = self.layer12(h1)
        h1 = self.layer13(h1)
        h1 = h1.permute(2, 3, 0, 1)
        m, n, p, q = h1.shape
        h1 = h1.reshape(m * n, p * q)
        em = torch.sigmoid(h1)  # 将端元约束到0-1内
        return em

    def forward(self, input):
        mu, log_var = self.encoder_m(input)
        z = self.reparameterize(mu, log_var)
        a = self.encoder_a(input)
        a = a.permute(2, 3, 0, 1)
        a = a.reshape(self.col * self.col, self.P)
        em = self.decoder(z)
        # em = em.permute(2, 3, 0, 1)
        # em = em.reshape(self.col * self.col, self.P * self.Channel)
        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor  # @ 矩阵乘法
        y_hat = torch.squeeze(y_hat, dim=1)

        mu = mu.permute(2, 3, 0, 1)
        mu = mu.reshape(self.col * self.col, self.z_dim)

        log_var = log_var.permute(2, 3, 0, 1)
        log_var = log_var.reshape(self.col * self.col, self.z_dim)
        return y_hat, mu, log_var, a, em_tensor


if __name__ == '__main__':
    P, Channel, z_dim = 5, 200, 4
    col = 10
    device = 'cpu'
    model = PGMSU(P, Channel, z_dim, col)
    input = torch.randn(1, Channel, col, col)
    y_hat, mu, log_var, a, em = model(input)
    print('shape of y_hat: ', y_hat.shape)
    print(mu.shape)
    print(log_var.shape)
    print(a.shape)
    print(em.shape)

