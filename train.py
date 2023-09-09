import os
import numpy as np
import torch.utils
import torch.utils.data
from torch import nn
import scipy.io as scio
import time
from gaijin7 import PGMSU
from loadhsi import loadhsi
import random
from hyperVca import hyperVca
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from utils import compute_rmse, compute_sad, compute_sad1
seed = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PYTHONHASHSEED'] = str(seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if not torch.cuda.is_available():
            print('WARNING: You have a CUDA device, so you should probably run with --cuda')
        else:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


set_seed(seed)
tic = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cases = ['samson', 'jasper', 'apex']
case = cases[0]
# load data
Y, A_true, EM_true, EM_VCA, P, Channel, col = loadhsi(case)  # Y原始高光谱图像数据   A_true 真实丰度图  P端元数
print(Y.shape, A_true.shape, EM_true.shape, EM_VCA.shape, P, Channel, col)
print(EM_true.shape)
abundance_GT = torch.from_numpy(A_true).float()  # true abundance
original_HSI = torch.from_numpy(Y).float()  # 原始图像
original_HSI = torch.reshape(original_HSI, (Channel, col, col))
original_HSI = original_HSI.unsqueeze(0)
abundance_GT = torch.reshape(abundance_GT, (P, col, col))
print(abundance_GT.shape, original_HSI.shape)
if case == 'samson':
    lambda_rec = 1
    lambda_vca = 2
    lambda_kl = 1  # 越大越好
    lambda_sad = 1
    lambda_vol = 6
    patch = 5
    dim = 200
    batchsz = 1  # 批处理次数 1000
    N = col * col
    lr = 5e-4  # 学习率
    epochs = 1000  # 迭代次数
    z_dim = 4  # 潜向量数
    index = [1, 0, 2]
    index1 = [0, 1, 2]
    path = 'bundles_samson.mat'

if case == 'jasper':
    lambda_rec = 1
    lambda_vca = 5
    lambda_kl = 0.001  # 越小越好
    lambda_sad = 3  # 越大越好
    lambda_vol = 7  # 越大越好
    patch = 5
    dim = 200
    batchsz = 1
    N = col * col
    lr = 5e-4  # 学习率
    epochs = 1000  # 迭代次数
    z_dim = 3  # 潜向量数
    index = [2, 1, 0, 3]
    index1 = [3, 1, 2, 0]
    path = 'bundles_ridge.mat'


if case == 'apex':
    lambda_rec = 1
    lambda_vca = 5
    lambda_kl = 0.01
    lambda_sad = 5
    lambda_vol = 7
    patch = 10
    dim = 200
    batchsz = 1  # 批处理次数 1000
    N = col * col
    lr = 5e-4  # 学习率
    epochs = 2000  # 迭代次数
    z_dim = 4  # 潜向量数
    index = [3, 1, 0, 2]
    index1 = [0, 1, 2, 3]
    path = 'bundles_apex.mat'

# 创建数据保存文件夹
model_weights = './PGMSU_weight/'
output_path = './PGMSU_out/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
if not os.path.exists(model_weights):
    os.makedirs(model_weights)
model_weights += 'PGMSU.pt'


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
    elif classname.find('Conv2D') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# 数据准备
train_db = torch.utils.data.TensorDataset(original_HSI)
train_db = torch.utils.data.DataLoader(original_HSI, batch_size=batchsz, shuffle=True)

# Vca端元估计
# EM_VCA, _, _ = hyperVca(Y.T, P)
# EM = EM.T
# EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
# CM = EM
# EM = torch.tensor(EM).to(device)
# for i in range(P):
#     plt.subplot(2, 2, i+1)
#     plt.plot(CM[0, i, :].T, 'r', linewidth=0.5)
#     plt.axis([0, len(CM[0, i, :]), 0, 1])
# plt.show()

EM_bundles = scio.loadmat(path)
EMr1 = EM_bundles['bundleLibs']
EM = np.mean(EMr1, axis=0)
EM = EM[index, :]
A = EM
EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
EM = torch.tensor(EM.astype(np.float32)).to(device)
print("EM", EM.shape)  # 端元束抓取 (1, 4, 198)
sad_err, mean_sad = compute_sad(A.T, EM_true)
# print(mean_sad, sad_err[0], sad_err[1], sad_err[2], sad_err[3])
#
# EM_VCA = EM_VCA[:, index1]
# sad_err, mean_sad = compute_sad(EM_VCA, EM_true)
# print(mean_sad, sad_err[0], sad_err[1], sad_err[2])

# EM = EM_VCA.T
# EM = np.reshape(EM, [1, EM.shape[0], EM.shape[1]]).astype('float32')
# CM = EM
# EM = torch.tensor(EM).to(device)
# for i in range(P):
#     plt.subplot(2, 2, i+1)
#     plt.plot(A[i, :].T, 'r', linewidth=0.5, label='S_VCA')
#     plt.plot(EM_true[:, i], 'b', linewidth=0.5, label='True')
#     plt.plot(EM_VCA[:, i], 'y', linewidth=0.5, label='VCA')
#     plt.axis([0, len(EM[0, i, :]), 0, 1])
#     plt.legend()
# plt.show()

model = PGMSU(P, Channel, z_dim, col).to(device)
# model = PGMSU(P, Channel, z_dim, col, patch, dim).to(device)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=4e-5)
losses = []
print('Start training!')
for epoch in range(epochs):
    model.train()
    for step, y in enumerate(train_db):
        y = y.to(device)
        y_hat, mu, log_var, a, em_tensor = model(y)
        # MSE
        y = y.permute(2, 3, 0, 1)
        y = y.reshape(col*col, Channel)
        loss_rec = ((y_hat - y) ** 2).sum() / y.shape[0]

        # ssim
        # loss_ssim = ssim(y_hat, y)

        # KL散度
        kl_div = -0.5 * (log_var + 1 - mu ** 2 - log_var.exp())
        kl_div = kl_div.sum() / y.shape[0]
        # KL balance of VAE
        kl_div = torch.max(kl_div, torch.tensor(0.2).to(device))

        if epoch < epochs // 2:
            # pre-train process 预训练处理  前100次训练 为了稳定分解结果并加快训练过程，在训练前采用预训练阶段。具体来说，我们使用VCA提取的端元来约束生成模型
            loss_vca = (em_tensor - EM).square().sum() / y.shape[0]
            loss = lambda_rec * loss_rec + lambda_kl * kl_div + lambda_vca * loss_vca
        else:
            # training process 训练过程
            # constrain 1 min_vol of EMs 端元的限制1 min_vol
            em_bar = em_tensor.mean(dim=1, keepdim=True)  # 求tensor每行的平均值
            loss_minvol = ((em_tensor - em_bar) ** 2).sum() / y.shape[0] / P / Channel

            # constrain 2 SAD for same materials  相同材料的限制 2 SAD
            em_bar = em_tensor.mean(dim=0, keepdim=True)  # [1,4,198] [1,z_dim,Channel]  求tensor每列的平均值
            aa = (em_tensor * em_bar).sum(dim=2)
            em_bar_norm = em_bar.square().sum(dim=2).sqrt()
            em_tensor_norm = em_tensor.square().sum(dim=2).sqrt()

            sad = torch.acos(aa / (em_bar_norm + 1e-6) / (em_tensor_norm + 1e-6))
            loss_sad = sad.sum() / y.shape[0] / P
            loss =lambda_rec * loss_rec + lambda_kl * kl_div + lambda_vol * loss_minvol + lambda_sad * loss_sad

        optimizer.zero_grad()
        loss.backward()
        # 函数的主要目的是对encoder.parameters()里的所有参数的梯度进行规范化,防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()

    losses.append(loss.detach().cpu().numpy())
    if (epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), model_weights)
        scio.savemat(output_path + 'loss.mat', {'loss': losses})
        print('epoch:', epoch+1, ' save results!')

toc = time.time()
print('time elapsed:', toc - tic)

# 验证
model.eval()
with torch.no_grad():
    y_hat, mu, log_var, A, em_hat = model(original_HSI.to(device))
    print(em_hat.shape, A.shape)   # 端元数据 (10000, 4, 198)
    A_hat = A.cpu().numpy().T  # (4, 10000) 模型丰度
    A_true = A_true.reshape(P, N)   # (4, 10000) 真实丰度
    dev = np.zeros([P, P])
    for i in range(P):
        for j in range(P):
            dev[i, j] = np.mean((A_hat[i, :] - A_true[j, :]) ** 2)
    pos = np.argmin(dev, axis=0)

    A_hat = A_hat[pos, :]
    em_hat = em_hat[:, pos, :]
    print(A_hat[0, :].shape, A_true.shape)
    # sum = 0
    # for i in range(P):
    #     armse_a1 = np.sqrt(np.mean(A_hat[i, :] - A_true[i, :]) ** 2)
    #     sum = sum + armse_a1
    #     print("丰度", armse_a1)
    # armse_a = sum/P

    A = np.reshape(A_hat, [P, col, col])
    A_True = np.reshape(A_true, [P, col, col])
    print(A.shape, A_true.shape)
    class_rmse, mean_rmse = compute_rmse(A, A_True)
    # print(mean_rmse, class_rmse[0], class_rmse[1], class_rmse[2], class_rmse[3])
    print(mean_rmse, class_rmse[0], class_rmse[1], class_rmse[2])
    # Y_hat = y_hat.cpu().numpy()
    # Y = Y.T
    # print(Y_hat.shape, Y.shape)
    # # 计算均值根误差——重构图像
    # armse_y = np.mean(np.sqrt(np.mean((Y_hat - Y) ** 2, axis=1)))
    # norm_y = np.sqrt(np.sum(Y ** 2, 1))
    # norm_y_hat = np.sqrt(np.sum(Y_hat ** 2, 1))
    # asad_y = np.mean(np.arccos(np.sum(Y_hat * Y, 1) / norm_y / norm_y_hat))

    EM_hat = em_hat.cpu().numpy()
    # 对于端元计算均方根误差
    print(EM_hat.shape, EM_true.shape)
    # armse_m = np.mean(np.mean(np.sqrt(np.mean((EM_hat - EM_true.T) ** 2, axis=2)), axis=1))
    # norm_m = np.sqrt(np.sum(EM_true.T**2, 1))
    # norm_m_hat = np.sqrt(np.sum(EM_hat**2, 2))
    # asad_m = np.mean(np.mean(np.arccos(np.sum(EM_hat*EM_true.T, 2)/norm_m_hat/norm_m), axis=1))
    scio.savemat(output_path + 'apex_ECBAM.mat', {'EM': em_hat.data.cpu().numpy(), 'A': A_hat.T, 'Y_hat': y_hat.cpu().numpy()})
    # for i in range(P):
    #     norm_m = np.sqrt(np.sum(EM_true[:, i].T ** 2, 0))
    #     norm_m_hat = np.sqrt(np.sum(EM_hat[:, i, :] ** 2, 1))
    #     asad_m1 = np.mean(np.arccos(np.sum(EM_hat[:, i, :] * EM_true[:, i].T, 1) / norm_m_hat / norm_m), axis=0)
    #     print(i, "端元:", asad_m1)

    # sad_err, mean_sad = compute_sad1(EM_hat, EM_true)
    # print(mean_sad, sad_err[0], sad_err[1], sad_err[2], sad_err[3])

    sad_err, mean_sad = compute_sad1(EM_hat, EM_true)
    print(mean_sad, sad_err[0], sad_err[1], sad_err[2])

    SS = np.mean(EM_hat, axis=0).T
    print(SS.shape)

    # sad_err, mean_sad = compute_sad(SS, EM_true)
    # print(mean_sad, sad_err[0], sad_err[1], sad_err[2], sad_err[3])

    sad_err, mean_sad = compute_sad(SS, EM_true)
    print(mean_sad, sad_err[0], sad_err[1], sad_err[2])

# print('*' * 70)
# print('RESULTS:')
# print('armse_a:', armse_a)
# print('armse_Y', armse_y, 'asad_Y', asad_y)
# print('armse_M', armse_m, 'asad_M', asad_m)
