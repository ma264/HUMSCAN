import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt


def loadhsi(case):
    '''
    :input: case: for different datasets,
                 'toy' and 'usgs' are synthetic datasets
    :return: Y : HSI data of size [Bands,N]
             A_ture : Ground Truth of abundance map of size [P,N]
             P : nums of endmembers signature
    '''
    if case == 'jasper':
        file = './dataset/jasper_dataset.mat'
        data = scio.loadmat(file)
        Channel = 198
        P = 4
        col = 100

    elif case == 'samson':
        file = './dataset/samson_dataset.mat'
        data = scio.loadmat(file)
        Channel = 156
        P = 3
        col = 95
    elif case == 'dc':
        file = './dataset/dc_dataset.mat'
        data = scio.loadmat(file)
        Channel = 191
        P = 6
        col = 290

    elif case == 'apex':
        file = './dataset/apex_dataset.mat'
        data = scio.loadmat(file)
        Channel = 285
        P = 4
        col = 110
    Y = data['Y']
    A_true = data['A']
    EM_true = data['M']
    EM_VCA = data['M1']

    return Y, A_true, EM_true, EM_VCA, P, Channel, col


if __name__ == '__main__':
    cases = ['dc', 'jasper', 'samson', 'apex']
    case = cases[0]

    Y, A_true, EM_true, EM_VCA, P, Channel, col = loadhsi(case)
    print(Y.shape, A_true.shape, EM_true.shape, EM_VCA.shape)
