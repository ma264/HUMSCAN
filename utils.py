import numpy as np


def compute_rmse(x_true, x_pre):
    w, h, c = x_true.shape
    class_rmse = [0] * w
    for i in range(w):
        class_rmse[i] = np.sqrt(((x_true[i, :, :] - x_pre[i, :, :]) ** 2).sum() / (c * h))
    # mean_rmse = np.sqrt(((x_true - x_pre) ** 2).sum() / (w * h * c))
    mean_rmse = np.mean(class_rmse)
    return class_rmse, mean_rmse

# 单端元
def compute_sad(inp, target):
    p = inp.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        inp_norm = np.linalg.norm(inp[:, i], 2)
        tar_norm = np.linalg.norm(target[:, i], 2)
        summation = np.matmul(inp[:, i].T, target[:, i])
        sad_err[i] = np.arccos(summation / (inp_norm * tar_norm))
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad

# 多端元
def compute_sad1(inp, target):
    p = target.shape[-1]
    sad_err = [0] * p
    for i in range(p):
        norm_m = np.sqrt(np.sum(target[:, i].T ** 2, 0))
        norm_m_hat = np.sqrt(np.sum(inp[:, i, :] ** 2, 1))
        sad_err[i] = np.mean(np.arccos(np.sum(inp[:, i, :] * target[:, i].T, 1) / norm_m_hat / norm_m), axis=0)
    mean_sad = np.mean(sad_err)
    return sad_err, mean_sad

