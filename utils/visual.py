import colorsys
import os
import pickle
import random

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from umap import UMAP
import wandb
import seaborn as sns


# sample to 10000
def sample_10000(labels, h1_data, h2_data, h3_data):
    if not (labels == h1_data == h2_data == h3_data):
        raise ValueError("The lengths of the four arrays are inconsistent。")

    if labels.shape[0] > 10000:
        idx = np.random.choice(labels.shape[0], 10000, replace=False)
        labels = labels[idx]
        h1_data = h1_data[idx]
        h2_data = h2_data[idx]
        h3_data = h3_data[idx]
    return labels, h1_data, h2_data, h3_data


def print_dict_structure(d, indent=0):
    for key in d:
        print('  ' * indent + str(key))
        if isinstance(d[key], dict):
            print_dict_structure(d[key], indent + 1)


# ########################################################################################################################################

USD_file = 'path_to_***_dataset'

# ########################################################################################################################################

print('USD_file:', USD_file)

path = f'./save_Embedding/SWaT/'
with open(path + USD_file, 'rb') as file:
    ours_feature_data = pickle.load(file)

print('ours_feature_data type:', type(ours_feature_data))
print('ours_feature_data structure:')
print_dict_structure(ours_feature_data, indent=2)

ours_train_h_data = ours_feature_data['train']['features']
ours_train_h_label = ours_feature_data['train']['labels']
original_data = ours_feature_data['train']['original']
print('ours_train_h_data:', ours_train_h_data.shape, 'ours_train_h_label:', ours_train_h_label.shape)
print('original_data:', original_data.shape)

pca = PCA(n_components=100)
ours_train_h_pca = pca.fit_transform(ours_train_h_data)
print('finished pca')

perplex = 200
niter = 1000
seed = 0

der = TSNE(n_components=2, random_state=seed, perplexity=perplex, n_iter=niter)
ours_train_h_tsne = der.fit_transform(ours_train_h_pca)
print('finished umap or tsne')

colors = ['#1f77b4', '#ff7f0e']
Labels = ['Normal', 'Anomaly']

# 设置字体大小
plt.rcParams.update({'font.size': 20})  # 或者使用 plt.rc('font', size=14)

# 创建一个图和轴
fig = plt.figure(figsize=(10, 10))

plt.scatter(ours_train_h_tsne[:, 0], ours_train_h_tsne[:, 1], c=[colors[label] for label in ours_train_h_label], s=1)

def search_poing(target_point):
    # 计算每个点与目标点的距离
    distances = np.linalg.norm(ours_train_h_tsne - target_point, axis=1)

    # 获取距离最近的几个点的索引
    # 例如，我们这里找到距离最近的5个点
    num_neighbors = 20
    nearest_indices = np.argsort(distances)[:num_neighbors]

    # 输出这些点的索引、坐标和对应的距离
    nearest_points = ours_train_h_tsne[nearest_indices]
    nearest_distances = distances[nearest_indices]
    print('-' * 80)
    print('target point:', target_point)
    for i, index in enumerate(nearest_indices):
        point = nearest_points[i]
        dist = nearest_distances[i]
        print(f"Index: {index}, Point: {point}, Distance: {dist}")
    print('-' * 80)

    return nearest_indices


# %% 目标点坐标
points = np.array([[-60, -40], [-18, -68], [-35, 38], [-25, 65]])

marked_Labels = ['A', 'B', 'C', 'D']

# # 特别标出四个点并加标签
# for i, (x, y) in enumerate(points):
#     plt.text(x, y, marked_Labels[i], fontsize=20)

plt.tight_layout()

wandb.log({f'visual': wandb.Image(fig)})

# 保存图形为PDF文件
pdf_filename = f'visual.png'
plt.savefig(pdf_filename, format='png')

# 上传PDF文件到wandb
wandb.save(pdf_filename)

point1_index = search_poing(points[0])
point2_index = search_poing(points[1])
point3_index = search_poing(points[2])
point4_index = search_poing(points[3])

marked_point_index = []

marked_point_index.append(point1_index[0])
marked_point_index.append(point2_index[0])
marked_point_index.append(point3_index[0])
marked_point_index.append(point4_index[0])
marked_point_index.append(point1_index[-1])
marked_point_index.append(point2_index[-1])
marked_point_index.append(point3_index[-1])
marked_point_index.append(point4_index[-1])

arrays = []
for i, index in enumerate(marked_point_index):
    print('-' * 60)
    image_data = original_data[index].reshape(51, 60)
    image_label = ours_train_h_label[index]
    print('image_data:', i, index, image_data, image_data.shape, 'label:', image_label)
    arrays.append(image_data)
    # print('image_data:', image_data)

# 创建基于特定颜色的渐变色映射
base_color = '#1f77b4'
cmap = LinearSegmentedColormap.from_list("custom_cmap", ['#ff7f0e', base_color])

# 创建一个画布和8个子图，排版为2行4列
fig1, axs = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
# 调整子图间距
plt.subplots_adjust(wspace=0.04)

for ax, data in zip(axs.flat, arrays):
    # 绘制heatmap
    im = ax.imshow(data, cmap=cmap)
    ax.set_xlabel('Time')  # 设置x轴标签
    ax.set_ylabel('sensor')  # 设置y轴标签

# 添加一个统一的colorbar
cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8)

# 保存图形为PDF文件
pdf_filename = f'heatmap.png'
plt.savefig(pdf_filename, format='png')

plt.close(fig)

