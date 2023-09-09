from torch import nn
import torch
import torch.nn.functional as F
import transformer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PGMSU(nn.Module):
    def __init__(self, P, Channel, z_dim, col, patch, dim):
        super(PGMSU, self).__init__()
        self.P = P
        self.Channel = Channel
        self.col = col
        self.z_dim = z_dim
        self.dim = dim
        self.pathch = patch
        # encoder z
        self.layer1 = nn.Sequential(
            nn.Conv2d(Channel, 32 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32 * P, 16 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16 * P, (dim*z_dim)//patch**2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d((dim*z_dim)//patch**2, momentum=0.5),
            # nn.LeakyReLU(0.2),
        )

        self.layer15 = transformer.ViT(image_size=col, patch_size=patch, dim=(dim * z_dim), depth=3,
                                       heads=8, mlp_dim=12, pool='cls')

        self.upscale1 = nn.Sequential(
            nn.Linear(dim, col ** 2),
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1, stride=1, padding=0),
        )

        self.layer14 = nn.Sequential(
            nn.Conv2d(z_dim, z_dim, kernel_size=1, stride=1, padding=0),
        )

        # # encoder a 软分类 获得丰度估计
        # self.layer9 = nn.Sequential(
        #     nn.Conv1d(1, 3, kernel_size=5),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(3, 6, kernel_size=4),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(6, 12, kernel_size=5),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #     nn.Conv1d(12, 24, kernel_size=4),
        #     nn.MaxPool1d(kernel_size=2, stride=2),
        #
        # )
        # self.fc1 = nn.Linear(24 * 9, 100)
        # self.bn1 = nn.BatchNorm1d(100)
        # self.fc2 = nn.Linear(100, P)

        # encoder b 软分类 获得丰度估计
        self.layer4 = nn.Sequential(
            nn.Conv2d(Channel, 32 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(32 * P, 16 * P, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(16 * P, (dim*P)//patch**2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d((dim*P)//patch**2, momentum=0.5),
        )

        self.layer11 = transformer.ViT(image_size=col, patch_size=patch, dim=(dim*P), depth=3,
                                       heads=8, mlp_dim=12, pool='cls')

        self.upscale = nn.Sequential(
            nn.Linear(dim, col ** 2),
        )

        # self.layer12 = nn.Sequential(
        #     nn.Conv2d(4 * P, P, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(0.2),
        #     nn.BatchNorm2d(P),
        #     nn.Dropout(0.2),
        # )
        # self.fc6 = nn.Linear(4*P, P)

        # decoder
        self.layer6 = nn.Sequential(
            nn.Conv2d(z_dim, 32 * P, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32 * P),
            # nn.Dropout(0.2),
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(32 * P, 64 * P, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64 * P),
            # nn.Dropout(0.2),
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(64 * P, Channel * P, kernel_size=1),
            # nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(Channel * P),
            # nn.Dropout(0.2),
        )

        self.fc5 = nn.Linear(Channel * P, Channel*P)

        self.layer17 = nn.Sequential(
            nn.Linear(z_dim, 32 * P),
            nn.BatchNorm1d(32 * P),
            nn.LeakyReLU(0.2)
        )

        self.layer18 = nn.Sequential(
            nn.Linear(32 * P, 64 * P),
            nn.BatchNorm1d(64 * P),
            nn.LeakyReLU(0.2),
        )

        self.layer19 = nn.Sequential(
            nn.Linear(64 * P, Channel * P),
        )

    def encoder_z(self, x):
        h1 = self.layer1(x)
        h1 = self.layer2(h1)
        h1 = self.layer3(h1)
        h1 = self.layer15(h1)
        h1 = h1.view(1, self.z_dim, -1)
        h1 = self.upscale1(h1).view(1, self.z_dim, self.col, self.col)
        mu = self.layer13(h1)
        log_var = self.layer14(h1)
        return mu, log_var

     # 丰度编码器
    # def encoder_a(self, x):
    #     x = x.permute(2, 3, 0, 1)
    #     x = torch.reshape(x, (self.col*self.col, 1, self.Channel))
    #     a = self.layer9(x)
    #     m, n, p = a.shape
    #     a = torch.reshape(a, (m, n*p))
    #     a = self.fc1(a)
    #     a = self.bn1(a)
    #     a = self.fc2(a)
    #     a = F.softmax(a, dim=1)  # 满足ASC ANC限制
    #     return a

        # 丰度编码器
    def encoder_b(self, x):
        a = self.layer4(x)
        a = self.layer5(a)
        a = self.layer10(a)
        a = self.layer11(a)
        # a = self.layer12(a)
        a = a.view(1, self.P, -1)
        a = self.upscale(a).view(1, self.P, self.col, self.col)
        a = a.permute(2, 3, 0, 1)
        a = a.reshape(-1, self.P)
        a = F.softmax(a, dim=1)  # 满足ASC ANC限制
        return a

    # 对端元加偏置
    def reparameterize(self, mu, log_var):   # 对端元
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.shape, device=device)
        return mu + eps * std

    def decoder(self, z):
        h1 = self.layer6(z)
        h1 = self.layer7(h1)
        h1 = self.layer8(h1)
        h1 = h1.permute(2, 3, 0, 1)
        m, n, p, q = h1.shape
        h1 = h1.reshape(m*n, p*q)
        h1 = self.fc5(h1)
        em = torch.sigmoid(h1)  # 将端元约束到0-1内

        # h1 = z.permute(2, 3, 0, 1)
        # m, n, p, q = h1.shape
        # h1 = h1.reshape(m * n, p * q)
        # h1 = self.layer17(h1)
        # h1 = self.layer18(h1)
        # h1 = self.layer19(h1)
        # em = torch.sigmoid(h1)  # 将端元约束到0-1内
        return em

    def forward(self, inputs):
        mu, log_var = self.encoder_z(inputs)
        a = self.encoder_b(inputs)
        # reparameterization trick 重新参数化技巧
        z = self.reparameterize(mu, log_var)
        em = self.decoder(z)
        em_tensor = em.view([-1, self.P, self.Channel])
        a_tensor = a.view([-1, 1, self.P])
        y_hat = a_tensor @ em_tensor  # @ 矩阵乘法
        y_hat = torch.squeeze(y_hat, dim=1)

        mu = mu.permute(2, 3, 0, 1)
        mu = mu.reshape(self.col*self.col, self.z_dim)

        log_var = log_var.permute(2, 3, 0, 1)
        log_var = log_var.reshape(self.col * self.col, self.z_dim)
        return y_hat, mu, log_var, a, em_tensor


if __name__ == '__main__':
    P, Channel, z_dim = 5, 156, 4
    col = 95
    device = 'cpu'
    patch = 5
    dim = 200
    model = PGMSU(P, Channel, z_dim, col, patch, dim)
    input = torch.randn(1, Channel, col, col)
    y_hat, mu, log_var, a, em = model(input)
    print('shape of y_hat: ', y_hat.shape)
    print(mu.shape)
    print(log_var.shape)
    print(a.shape)
    print(em.shape)

