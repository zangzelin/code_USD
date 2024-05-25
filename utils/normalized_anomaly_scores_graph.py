import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import wandb


def print_dict_structure(d, indent=0):
    for key in d:
        print('  ' * indent + str(key))
        if isinstance(d[key], dict):
            print_dict_structure(d[key], indent + 1)

path = ''
USD_file = 'path_to_***_dataset'
MTGFLOW_file = 'path_to_***_dataset'
GANF_file = 'path_to_***_dataset'
# ############################################################################################################################


# %% USD
print('-' * 80)
with open(path + USD_file, 'rb') as file:
    USD_feature_data = pickle.load(file)
print('USD_feature_data type:', type(USD_feature_data))
print('USD_feature_data structure:')
print_dict_structure(USD_feature_data, indent=2)

USD__label = USD_feature_data['test']['labels']
USD_log_prob = USD_feature_data['test']['log_prob']
print("USD__label:", USD__label.shape)
print("USD_log_prob:", USD_log_prob.shape)

num_data = USD_log_prob.shape[0]
sorted_probs_reshaped = np.sort(USD_log_prob)
start_value = sorted_probs_reshaped[int(num_data * 0.05)]
end_value = sorted_probs_reshaped[int(num_data * 0.95)]
not_outliers = (USD_log_prob > start_value) & (USD_log_prob < end_value)
not_outliers = not_outliers.reshape(-1)
USD_labels = USD__label[not_outliers]
USD_log_prob = USD_log_prob[not_outliers]
# 将一维数组重塑为二维数组
USD_probs_reshaped = USD_log_prob.reshape(-1, 1)
print('-' * 80)

# %% MTGFLOW
with open(path + MTGFLOW_file, 'rb') as file:
    MTGFLOW_feature_data = pickle.load(file)
print('MTGFLOW_feature_data type:', type(MTGFLOW_feature_data))
print('MTGFLOW_feature_data structure:')
print_dict_structure(MTGFLOW_feature_data, indent=2)

MTGFLOW__label = MTGFLOW_feature_data['test']['labels']
MTGFLOW_log_prob = MTGFLOW_feature_data['test']['log_prob']
print("MTGFLOW__label:", MTGFLOW__label.shape)
print("MTGFLOW_log_prob:", MTGFLOW_log_prob.shape)

num_data = MTGFLOW_log_prob.shape[0]
sorted_probs_reshaped = np.sort(MTGFLOW_log_prob)
start_value = sorted_probs_reshaped[int(num_data * 0.05)]
end_value = sorted_probs_reshaped[int(num_data * 0.95)]
not_outliers = (MTGFLOW_log_prob > start_value) & (MTGFLOW_log_prob < end_value)
not_outliers = not_outliers.reshape(-1)
MTGFLOW_labels = MTGFLOW__label[not_outliers]
MTGFLOW_log_prob = MTGFLOW_log_prob[not_outliers]
# 将一维数组重塑为二维数组
MTGFLOW_probs_reshaped = MTGFLOW_log_prob.reshape(-1, 1)
print('-' * 80)

# %% GANF
with open(path + GANF_file, 'rb') as file:
    GANF_feature_data = pickle.load(file)
print('GANF_feature_data type:', type(GANF_feature_data))
print('GANF_feature_data structure:')
print_dict_structure(GANF_feature_data, indent=2)

GANF__label = GANF_feature_data['test']['labels']
GANF_log_prob = GANF_feature_data['test']['log_prob']
print("GANF__label:", GANF__label.shape)
print("GANF_log_prob:", GANF_log_prob.shape)

num_data = GANF_log_prob.shape[0]
sorted_probs_reshaped = np.sort(GANF_log_prob)
start_value = sorted_probs_reshaped[int(num_data * 0.05)]
end_value = sorted_probs_reshaped[int(num_data * 0.95)]
not_outliers = (GANF_log_prob > start_value) & (GANF_log_prob < end_value)
not_outliers = not_outliers.reshape(-1)
GANF_labels = GANF__label[not_outliers]
GANF_log_prob = GANF_log_prob[not_outliers]
# 将一维数组重塑为二维数组
GANF_probs_reshaped = GANF_log_prob.reshape(-1, 1)
print('-' * 80)

# %%
scaler = StandardScaler()

# 对probs数组进行归一化
USD_normalized_probs = scaler.fit_transform(USD_probs_reshaped).flatten()
MTGFLOW_normalized_probs = scaler.fit_transform(MTGFLOW_probs_reshaped).flatten()
GANF_normalized_probs = scaler.fit_transform(GANF_probs_reshaped).flatten()

# 根据标签将分开
USD_data_normal = [prob for label, prob in zip(USD_labels, USD_normalized_probs) if label == 0]
USD_data_anomaly = [prob for label, prob in zip(USD_labels, USD_normalized_probs) if label == 1]

MTGFLOW_data_normal = [prob for label, prob in zip(MTGFLOW_labels, MTGFLOW_normalized_probs) if label == 0]
MTGFLOW_data_anomaly = [prob for label, prob in zip(MTGFLOW_labels, MTGFLOW_normalized_probs) if label == 1]

GANF_data_normal = [prob for label, prob in zip(GANF_labels, GANF_normalized_probs) if label == 0]
GANF_data_anomaly = [prob for label, prob in zip(GANF_labels, GANF_normalized_probs) if label == 1]

USD_data_normal = np.array(USD_data_normal).reshape(-1, 1)
USD_data_anomaly = np.array(USD_data_anomaly).reshape(-1, 1)
MTGFLOW_data_normal = np.array(MTGFLOW_data_normal).reshape(-1, 1)
MTGFLOW_data_anomaly = np.array(MTGFLOW_data_anomaly).reshape(-1, 1)
GANF_data_normal = np.array(GANF_data_normal).reshape(-1, 1)
GANF_data_anomaly = np.array(GANF_data_anomaly).reshape(-1, 1)

# 创建MinMaxScaler实例
scaler = MinMaxScaler()

# 转换数据
normalized_USD_data_normal = scaler.fit_transform(USD_data_normal)
normalized_USD_data_anomaly = scaler.fit_transform(USD_data_anomaly)

normalized_MTGFLOW_data_normal = scaler.fit_transform(MTGFLOW_data_normal)
normalized_MTGFLOW_data_anomaly = scaler.fit_transform(MTGFLOW_data_anomaly)

normalized_GANF_data_normal = scaler.fit_transform(GANF_data_normal)
normalized_GANF_data_anomaly = scaler.fit_transform(GANF_data_anomaly)

normalized_USD_data_normal = normalized_USD_data_normal.ravel()
normalized_USD_data_anomaly = normalized_USD_data_anomaly.ravel()
normalized_MTGFLOW_data_normal = normalized_MTGFLOW_data_normal.ravel()
normalized_MTGFLOW_data_anomaly = normalized_MTGFLOW_data_anomaly.ravel()
normalized_GANF_data_normal = normalized_GANF_data_normal.ravel()
normalized_GANF_data_anomaly = normalized_GANF_data_anomaly.ravel()

# %% 设置字体大小
plt.rcParams.update({'font.size': 16})

# 创建一个图像和两个子图，设置sharex和sharey参数
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

# 调整子图间距
plt.subplots_adjust(wspace=0.04)

# 在第1个子图上绘制直方图
ax1.hist(x=[normalized_GANF_data_normal, normalized_GANF_data_anomaly], bins=40, color=['#1f77b4', '#ff7f0e'], density=True,
         histtype='bar', rwidth=0.8, label=['Normal', 'Anomaly'])
ax1.set_title('GANF(AUC=79.6%)', fontsize=18)  # ap = 19.7%  -
# 添加图例
ax1.legend(fontsize='x-small', frameon=True, edgecolor='gray', handlelength=2, handletextpad=0.5, markerscale=0.8)
# 添加轴标签
ax1.set_xlabel('Anomaly Score', fontsize=16)
ax1.set_ylabel('Density', fontsize=16)

# 在第2个子图上绘制直方图
ax2.hist(x=[normalized_MTGFLOW_data_normal, normalized_MTGFLOW_data_anomaly], bins=40, color=['#1f77b4', '#ff7f0e'], density=True,
         histtype='bar', rwidth=0.8, label=['Normal', 'Anomaly'])
ax2.set_title('MTGFLOW(AUC=83.2%)', fontsize=18)  # ap = 35.6%
# 添加图例
ax2.legend(fontsize='x-small', frameon=True, edgecolor='gray', handlelength=2, handletextpad=0.5, markerscale=0.8)
# 添加轴标签
ax2.set_xlabel('Anomaly Score', fontsize=16)

# 在第3个子图上绘制直方图
ax3.hist(x=[normalized_USD_data_normal, normalized_USD_data_anomaly], bins=40, color=['#1f77b4', '#ff7f0e'], density=True,
         histtype='bar',
         rwidth=0.8, label=['Normal', 'Anomaly'])
ax3.set_title('USD(AUC=91.6%)', fontsize=18)  # ap = 40.0%
# 添加图例，尽量小型化
ax3.legend(fontsize='x-small', frameon=True, edgecolor='gray', handlelength=2, handletextpad=0.5, markerscale=0.8)
# 添加轴标签
ax3.set_xlabel('Anomaly Score', fontsize=16)

# 调用tight_layout函数，使图像贴边
plt.tight_layout()


# 保存图形为PDF文件
pdf_filename = "normalized anomaly scores.pdf"
fig.savefig(pdf_filename, format='pdf')


plt.close(fig)
