import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from loadhsi import loadhsi
import torch
from sklearn.decomposition import PCA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cases = ['jasper', 'samson', 'apex']
case = cases[2]

model_weights = './PGMSU_weight/'
model_weights += 'PGMSU.pt'

output_path = './PGMSU_out/'
Y, A_true, EM_true, EM_VCA, P, Channel, col = loadhsi(case)
nCol = col
nRow = col
nband = Y.shape[0]
N = Y.shape[1]
Channel = Y.shape[0]
z_dim = 4

# plot 1: trainging loss
loss = scio.loadmat(output_path + 'loss.mat')['loss']
plt.loglog(loss[0])
plt.savefig(output_path+'loss.png')
plt.show()
# 端元 丰度 编码器输出像素重建
EM_hat = scio.loadmat(output_path + 'apex_ECBAM.mat')['EM']
A_hat = scio.loadmat(output_path + 'apex_ECBAM.mat')['A']
Y_hat = scio.loadmat(output_path + 'apex_ECBAM.mat')['Y_hat']
print(EM_hat.shape, A_hat.shape, Y.shape)

# 数据转换   重点
A_hat = np.reshape(A_hat, (nRow, nCol, P))
B = np.zeros((P, nRow, nCol))
for i in range(P):
    B[i] = A_hat[:, :, i]
A_hat = B
A_true = A_true.reshape([P, -1])
A_hat = A_hat.reshape([P, -1])

A_true = A_true.reshape([P, nCol, nRow])
A_hat = A_hat.reshape([P, nCol, nRow])
# plot 2 : Abundance maps 丰度图
fig = plt.figure()
for i in range(1, P + 1):
    # 网络计算的丰度图
    plt.subplot(2, P, i)
    aaa = plt.imshow(A_hat[i - 1], cmap='jet')
    aaa.set_clim(vmin=0, vmax=1)
    plt.axis('off')
    # 实际的丰度图
    plt.subplot(2, P, i + P)
    aaa = plt.imshow(A_true[i - 1], cmap='jet')
    aaa.set_clim(vmin=0, vmax=1)
    plt.axis('off')

plt.subplot(2, P, 1)
plt.ylabel('PGMSU')
plt.subplot(2, P, 1 + P)
plt.ylabel('reference GT')
plt.show()

EM_mean = np.mean(EM_hat, axis=0)
print(EM_mean.shape)
for i in range(P):
    plt.subplot(2, 2, i+1)
    plt.plot(EM_mean[i, :].T, 'r', linewidth=0.5)
    plt.plot(EM_true[:, i], 'b', linewidth=0.5)
    plt.axis([0, len(EM_mean[i, :]), 0, 1])
plt.show()

plt.figure()
for i in range(P):
    plt.subplot(2, (P + 1) // 2, i + 1)
    plt.plot(EM_hat[::50, i, :].T, 'c', linewidth=0.5)
    plt.xlabel('$\it{Bands}$', fontdict={'fontsize': 16})
    plt.ylabel('$\it{Reflectance}$', fontdict={'fontsize': 16})
    plt.axis([0, len(EM_hat[0, i, :]), 0, 1])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.axis('off')

pca = PCA(2)
y2d = pca.fit_transform(Y.T)
print(y2d.shape)
plt.figure()
plt.scatter(y2d[:, 0], y2d[:, 1], 5, label='Pixel data')
P = EM_hat.shape[1]

for i in range(P):
    em2d = pca.transform(np.squeeze(EM_hat[:, i, :]))
    plt.scatter(em2d[:, 0], em2d[:, 1], 5, label='EM #' + str(i + 1))

plt.legend()
plt.title('Scatter plot of mixed pixels and EMs')
# plt.savefig('em2d')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
