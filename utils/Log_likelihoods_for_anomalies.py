import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

import wandb


def find_ones_segments(labels):
    one_start = []
    one_end = []
    in_segment = False
    start_index = None

    # 遍历数组，同时记录索引
    for i, label in enumerate(labels):
        if label == 1 and not in_segment:
            # 当遇到 '1' 并且之前不在连续段中时，标记连续段开始
            in_segment = True
            start_index = i
        elif label == 0 and in_segment:
            # 当遇到 '0' 并且之前在连续段中时，标记连续段结束
            one_start.append(start_index)
            one_end.append(i - 1)
            in_segment = False

    # 如果数组以 '1' 结尾，确保收尾
    if in_segment:
        one_start.append(start_index)
        one_end.append(len(labels) - 1)

    return one_start, one_end


train_split = 0.6

root = "Data/input/SWaT_Dataset_Attack_v0.csv"
data = pd.read_csv(root, sep=';', low_memory=False)
Timestamp = pd.to_datetime(data["Timestamp"], format="mixed")
data["Timestamp"] = Timestamp
data = data.set_index("Timestamp")
labels = [int(l != 'Normal') for l in data["Normal/Attack"].values]
for i in list(data):
    data[i] = data[i].apply(lambda x: str(x).replace(",", "."))
data = data.drop(["Normal/Attack"], axis=1)
data = data.astype(float)
n_sensor = len(data.columns)
# %%
feature = data.iloc[:, :51]
scaler = StandardScaler()
norm_feature = scaler.fit_transform(feature)

norm_feature = pd.DataFrame(norm_feature, columns=data.columns, index=Timestamp)
norm_feature = norm_feature.dropna(axis=1)

train_df = norm_feature.iloc[:int(train_split * len(data))]
train_label = labels[:int(train_split * len(data))]
print('trainset size', train_df.shape, 'anomaly ration', sum(train_label) / len(train_label))

# test_df = norm_feature.iloc[int(train_split * len(data)):]
# test_label = labels[int(train_split * len(data)):]

test_df = norm_feature.iloc[int(0.8 * len(data)):]
test_label = labels[int(0.8 * len(data)):]
print('testset size', test_df.shape, 'anomaly ration', sum(test_label) / len(test_label))


def print_dict_structure(d, indent=0):
    for key in d:
        print('  ' * indent + str(key))
        if isinstance(d[key], dict):
            print_dict_structure(d[key], indent + 1)


path =''


USD_file = 'path_to_***_dataset.pkl'
MTGFLOW_file = 'path_to_***_dataset.pkl'

print('trainsplit:', train_split)

with open(path + USD_file, 'rb') as file:
    USD_feature_data = pickle.load(file)
print('USD_feature_data type:', type(USD_feature_data))
print('USD_feature_data structure:')
print_dict_structure(USD_feature_data, indent=2)

USD_label = USD_feature_data['test']['labels']
USD_log_prob = USD_feature_data['test']['log_prob']
USD_index = USD_feature_data['test']['idx']
print("USD_label:", USD_label.shape)
print("USD_log_prob:", USD_log_prob.shape)
print("USD_index:", USD_index.shape)

# 平移变换以保证所有值为正
USD_shifted_prob = USD_log_prob - np.min(USD_log_prob) + 1

# 应用对数变换
USD_log_transformed_prob = np.log(USD_shifted_prob)

print("USD_log_transformed_prob:", USD_log_transformed_prob.shape)

with open(path + MTGFLOW_file, 'rb') as file:
    MTGFLOW_feature_data = pickle.load(file)
print('MTGFLOW_feature_data type:', type(MTGFLOW_feature_data))
print('MTGFLOW_feature_data structure:')
print_dict_structure(MTGFLOW_feature_data, indent=2)

MTGFLOW_label = MTGFLOW_feature_data['test']['labels']
MTGFLOW_log_prob = MTGFLOW_feature_data['test']['log_prob']
MTGFLOW_index = MTGFLOW_feature_data['test']['idx']
print("MTGFLOW_label:", MTGFLOW_label.shape)
print("MTGFLOW_log_prob:", MTGFLOW_log_prob.shape)
print("MTGFLOW_index:", MTGFLOW_index.shape)


def choose_time_range(label, prob, index):
    combined = list(zip(label, prob, index))
    combined_sorted = sorted(combined, key=lambda x: x[2])
    # 解包排序后的列表以获取排序后的三个数组
    labels_sorted, prob_sorted, idx_sorted = zip(*combined_sorted)
    labels_sorted_list = list(labels_sorted)
    prob_sorted_list = list(prob_sorted)
    idx_sorted_list = list(idx_sorted)

    one_start, one_end = find_ones_segments(labels_sorted_list)

    print('one_start:\n', one_start, len(one_start))
    print('one_end:\n', one_end, len(one_end))
    print('=' * 80)
    for i in range(len(one_start)):
        print('one_start:', one_start[i], 'one_end:', one_end[i])
        anomaly_time_start = test_df.index[idx_sorted_list[one_start[i]]]
        anomaly_time_end = test_df.index[idx_sorted_list[one_end[i]]]
        print(f'anomaly_time_start{i}:', anomaly_time_start, '-->', f'anomaly_time_end:{i}', anomaly_time_end)

    print('time lengthe:  7649   -  3028', test_df.index[idx_sorted_list[7649]] - test_df.index[idx_sorted_list[3028]])
    print('=' * 80)

    return labels_sorted_list, prob_sorted_list, idx_sorted_list, one_start, one_end


print('=' * 80)
USD_labels_sorted_list, USD_prob_sorted_list, USD_idx_sorted_list, USD_one_start, USD_one_end = choose_time_range(USD_label,
                                                                                                                  USD_log_transformed_prob,
                                                                                                                  USD_index)
print('-' * 80)
MTGFLOW_labels_sorted_list, MTGFLOW_prob_sorted_list, MTGFLOW_idx_sorted_list, MTGFLOW_one_start, MTGFLOW_one_end = choose_time_range(
    MTGFLOW_label, MTGFLOW_log_prob, MTGFLOW_index)

print('=' * 80)

subidx_range_start = 7553
subidx_range_end = 7793

USD_choose_idx = USD_idx_sorted_list[subidx_range_start:subidx_range_end + 1]
USD_choose_likelihoods = USD_prob_sorted_list[subidx_range_start:subidx_range_end + 1]
USD_choose_lables = USD_labels_sorted_list[subidx_range_start:subidx_range_end + 1]
USD_time_range = test_df.index[USD_choose_idx]
print('subidx_range_start,subidx_range_end:', subidx_range_start, subidx_range_end)
print('USD_choose_idx:', type(USD_choose_idx), len(USD_choose_idx))
print('start time:', test_df.index[USD_choose_idx[0]], 'end time:', test_df.index[USD_choose_idx[-1]])
print('USD_time_range:', len(USD_time_range), type(USD_time_range))

MTGFLOW_choose_idx = MTGFLOW_idx_sorted_list[subidx_range_start:subidx_range_end + 1]
MTGFLOW_choose_likelihoods = MTGFLOW_prob_sorted_list[subidx_range_start:subidx_range_end + 1]
MTGFLOW_choose_lables = MTGFLOW_labels_sorted_list[subidx_range_start:subidx_range_end + 1]
MTGFLOWtime_range = test_df.index[MTGFLOW_choose_idx]
print('subidx_range_start,subidx_range_end:', subidx_range_start, subidx_range_end)
print('USD_choose_idx:', type(USD_choose_idx), len(USD_choose_idx))
print('start time:', test_df.index[USD_choose_idx[0]], 'end time:', test_df.index[USD_choose_idx[-1]])
print('MTGFLOWtime_range:', len(MTGFLOWtime_range), type(MTGFLOWtime_range))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# # 添加整个画布的标题
# fig.suptitle(f'{subidx_range_start} --> {subidx_range_end}', fontsize=30)

# 调整子图间距
plt.subplots_adjust(wspace=0.04)
# 散点图1
axs[0].scatter(MTGFLOWtime_range, MTGFLOW_choose_likelihoods, s=5, color='#1f77b4')
print('=' * 80)
# 高亮特定时间段
for i in range(len(MTGFLOW_one_start)):
    if subidx_range_start <= MTGFLOW_one_start[i] <= subidx_range_end and subidx_range_start <= MTGFLOW_one_end[i] <= \
            subidx_range_end:
        Highlight_time_start = test_df.index[MTGFLOW_idx_sorted_list[MTGFLOW_one_start[i]]]
        Highlight_time_end = test_df.index[MTGFLOW_idx_sorted_list[MTGFLOW_one_end[i]]]
        print('Highlight_time_start:', Highlight_time_start, '-->', 'Highlight_time_end:', Highlight_time_end)
        axs[0].axvspan(pd.Timestamp(Highlight_time_start), pd.Timestamp(Highlight_time_end), color='#d62728', alpha=0.1)
print('=' * 80)
# 设置日期格式
axs[0].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # 每internal分钟显示一个刻度
axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置时间的格式
# 设置 x 和 y 轴的标签
axs[0].set_xlabel('Time', fontsize=20)
axs[0].set_ylabel('Prediction', fontsize=20)
axs[0].set_title('MTGFLOW', fontsize=24)
# 设置刻度标签文字大小
axs[0].set_xticklabels(axs[0].get_xticklabels(), fontsize=20)
axs[0].set_yticklabels(axs[0].get_yticks(), fontsize=20)
# 设置y轴的刻度格式，只保留一位小数
axs[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# 自动格式化日期标签以避免它们重叠
# fig.autofmt_xdate()

# 散点图2
axs[1].scatter(USD_time_range, USD_choose_likelihoods, s=5, color='#1f77b4')
print('=' * 80)
# 高亮特定时间段
for i in range(len(USD_one_start)):
    if subidx_range_start <= USD_one_start[i] <= subidx_range_end and subidx_range_start <= USD_one_end[i] <= subidx_range_end:
        Highlight_time_start = test_df.index[USD_idx_sorted_list[USD_one_start[i]]]
        Highlight_time_end = test_df.index[USD_idx_sorted_list[USD_one_end[i]]]
        print('Highlight_time_start:', Highlight_time_start, '-->', 'Highlight_time_end:', Highlight_time_end)
        axs[1].axvspan(pd.Timestamp(Highlight_time_start), pd.Timestamp(Highlight_time_end), color='#d62728', alpha=0.1)
print('=' * 80)
# 设置日期格式
axs[1].xaxis.set_major_locator(mdates.MinuteLocator(interval=10))  # 每internal分钟显示一个刻度
axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置时间的格式
# 设置 x 和 y 轴的标签
axs[1].set_xlabel('Time', fontsize=20)
# axs[1].set_ylabel('Prediction', fontsize=20)
axs[1].set_title('USD', fontsize=24)
# 设置刻度标签文字大小
axs[1].set_xticklabels(axs[1].get_xticklabels(), fontsize=20)
axs[1].set_yticklabels(axs[1].get_yticks(), fontsize=20)
# 设置y轴的刻度格式，只保留一位小数
axs[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# 自动格式化日期标签以避免它们重叠
# fig.autofmt_xdate()

plt.tight_layout()


# 保存图形为PDF文件
pdf_filename = "Log_likelihoods_for_anomalies.pdf"
fig.savefig(pdf_filename, format='pdf')
print('end..')


plt.close()


