import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

case = ['ridge', 'samson', 'apex']
case = case[0]
if case == 'ridge':
    data = scio.loadmat('dataset/jasper_dataset.mat')
    Y = data['Y']
    Y = Y / np.max(Y)  # 归一化
    image = np.reshape(Y, [198, 100, 100])
    # image = image[:, 0:300, 0:300]
    image = np.swapaxes(image, 0, 2)  # 将0维与2维进行交换
    image = image[:, :, [30, 20, 10]] * 3  # [100,100,3]
    n_segments = 200
    compactness = 10.
    Channel = 198
    P = 4
    W = 100
    H = 100
    times = 250
    subNum = 200
    seg_file = 'segments_ridge.mat'

elif case == 'samson':
    data = scio.loadmat('dataset/samson_dataset.mat')
    Y = data['Y']
    Y = Y / np.max(Y)
    image = np.reshape(Y, [156, 95, 95])
    image = np.swapaxes(image, 0, 2)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image[:, :, [30, 20, 10]] * 1.5
    n_segments = 200
    compactness = 10.
    P = 3
    Channel = 156
    W = 95
    H = 95
    times = 250
    subNum = 200
    seg_file = 'segments_samson.mat'

if case == 'apex':
    data = scio.loadmat('dataset/apex_dataset.mat')
    Y = data['Y']
    Y = Y / np.max(Y)  # 归一化
    image = np.reshape(Y, [285, 110, 110])
    # image = image[:, 0:300, 0:300]
    image = np.swapaxes(image, 0, 2)  # 将0维与2维进行交换
    image = image[:, :, [30, 20, 10]] * 3  # [100,100,3]
    n_segments = 600
    compactness = 20.
    Channel = 285
    P = 4
    W = 110
    H = 110
    times = 250
    subNum = 200
    seg_file = 'segments_apex.mat'

# n_segments 可选分割输出图像中标签的(大致)数量。
# compactness控制颜色和空间之间的平衡，约高越方块，和图关系密切，最好先确定指数级别，再微调   compactness=S/N
segments = slic(image, n_segments=n_segments, start_label=1, max_num_iter=100, compactness=compactness)

print(segments.shape)
print(np.max(segments), np.min(segments))
scio.savemat(seg_file, {'segments': segments})

fig = plt.figure("Superpixels -- %d segments" % (400))
plt.subplot(131)
plt.title('image')
plt.imshow(image)
plt.subplot(132)
plt.title('segments')
plt.imshow(segments)
plt.subplot(133)
plt.title('image and segments')
plt.imshow(mark_boundaries(image, segments, color=(0, 0, 0)))
plt.show()
