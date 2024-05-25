import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import wandb
import seaborn as sns


def print_dict_structure(d, indent=0):
    for key in d:
        print('  ' * indent + str(key))
        if isinstance(d[key], dict):
            print_dict_structure(d[key], indent + 1)


file_name = ''
Datasets = ['SWaT', 'Wadi', 'PSM', 'MSL']
colors = ['#bcbd22', '#2ca02c', '#17becf', '#1f77b4', '#9467bd', '#d62728']
models = ['DROCC', 'DeepSAD', 'USAD', 'GANF', 'MTGFLOW', 'USD']

# 准备画布
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))  # 2行4列

# for dataset in Datasets:
# 循环处理每个数据集
for col, dataset in enumerate(Datasets):
    print('-' * 80)
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'

    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        DROCC_file = pickle.load(file)
    print('DROCC_file type:', type(DROCC_file))
    print('DROCC_file structure:')
    print_dict_structure(DROCC_file, indent=2)
    DROCC_y_test = DROCC_file['y_test']
    DROCC_y_pred = DROCC_file['y_pred']
    print('DROCC_y_test:', DROCC_y_test.shape)
    print('DROCC_y_pred:', DROCC_y_pred.shape)
    print('-' * 80)

    # DeepSAD
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'
    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        DeepSAD_file = pickle.load(file)
    print('DeepSAD_file type:', type(DeepSAD_file))
    print('DeepSAD_file structure:')
    print_dict_structure(DeepSAD_file, indent=2)
    DeepSAD_y_test = DeepSAD_file['y_test']
    DeepSAD_y_pred = DeepSAD_file['y_pred']
    print('DeepSAD_y_test:', DeepSAD_y_test.shape)
    print('DeepSAD_y_pred:', DeepSAD_y_pred.shape)
    print('-' * 80)

    # USAD
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'
    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        USAD_file = pickle.load(file)
    print('USAD_file type:', type(USAD_file))
    print('USAD_file structure:')
    print_dict_structure(USAD_file, indent=2)
    USAD_y_test = USAD_file['y_test']
    USAD_y_pred = USAD_file['y_pred']
    print('USAD_y_test:', USAD_y_test.shape)
    print('USAD_y_pred:', USAD_y_pred.shape)
    print('-' * 80)

    # GANF
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'
    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        GANF_file = pickle.load(file)
    print('GANF_file type:', type(GANF_file))
    print('GANF_file structure:')
    print_dict_structure(GANF_file, indent=2)
    GANF_y_test = GANF_file['label']
    GANF_y_pred = GANF_file['loss_test']
    print('GANF_y_test:', GANF_y_test.shape)
    print('GANF_y_pred:', GANF_y_pred.shape)
    print('-' * 80)

    # MTGFLOW
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'
    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        MTGFLOW_file = pickle.load(file)
    print('MTGFLOW_file type:', type(MTGFLOW_file))
    print('MTGFLOW_file structure:')
    print_dict_structure(MTGFLOW_file, indent=2)
    MTGFLOW_y_test = MTGFLOW_file['label']
    MTGFLOW_y_pred = MTGFLOW_file['loss_test']
    print('MTGFLOW_y_test:', MTGFLOW_y_test.shape)
    print('MTGFLOW_y_pred:', MTGFLOW_y_pred.shape)
    print('-' * 80)

    # USD
    if dataset == 'SWaT':
        file_name = 'path_to_***_dataset'
    elif dataset == 'Wadi':
        file_name = 'path_to_***_dataset'
    elif dataset == 'PSM':
        file_name = 'path_to_***_dataset'
    elif dataset == 'MSL':
        file_name = 'path_to_***_dataset'
    data_file = f'./saved_loss/{dataset}/{file_name}'
    with open(data_file, 'rb') as file:
        USD_file = pickle.load(file)
    print('USD_file type:', type(USD_file))
    print('USD_file structure:')
    print_dict_structure(USD_file, indent=2)
    USD_y_test = USD_file['label']
    USD_y_pred = USD_file['loss_test']
    print('USD_y_test:', USD_y_test.shape)
    print('USD_y_pred:', USD_y_pred.shape)
    print('-' * 80)

    y_true = {
        "DROCC": DROCC_y_test,
        "DeepSAD": DeepSAD_y_test,
        "USAD": USAD_y_test,
        "GANF": GANF_y_test,
        "MTGFLOW": MTGFLOW_y_test,
        "USD": USD_y_test

    }

    y_scores = {
        "DROCC": DROCC_y_pred,
        "DeepSAD": DeepSAD_y_pred,
        "USAD": USAD_y_pred,
        "GANF": GANF_y_pred,
        "MTGFLOW": MTGFLOW_y_pred,
        "USD": USD_y_pred

    }
    # 绘制ROC曲线
    for (model, scores), color in zip(y_scores.items(), colors):
        fpr, tpr, _ = roc_curve(y_true[model], scores)
        roc_auc = auc(fpr, tpr)
        axs[0, col].plot(fpr, tpr, label=f'{model}(AUC = {roc_auc:.2f})', color=color)
        axs[0, col].plot([0, 1], [0, 1], 'k--')  # 对角线

    axs[0, col].set_title(f'{dataset}', fontsize=20)
    axs[0, col].set_xlim([0, 1])  # 设置x轴的范围
    axs[0, col].set_ylim([0, 1])  # 设置y轴的范围
    axs[0, col].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 设置x轴刻度格式
    axs[0, col].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 设置y轴刻度格式
    axs[0, col].set_xlabel('FPR', fontsize=18)
    axs[0, col].set_ylabel('TPR', fontsize=18)
    axs[0, col].legend(loc="lower right", fontsize=10)
    axs[0, col].tick_params(axis='both', which='major', labelsize=18)  # 设置刻度字体大小

    # 设置子图为正方形
    axs[0, col].set_aspect('equal', adjustable='box')

    # 绘制Precision-Recall曲线
    for (model, scores), color in zip(y_scores.items(), colors):
        precision, recall, _ = precision_recall_curve(y_true[model], scores)
        pr_auc = auc(recall, precision)
        axs[1, col].plot(recall, precision, label=f'{model}(PR = {pr_auc:.2f})', color=color)
    # axs[1, col].set_title(f'{dataset}', fontsize=24)
    axs[1, col].set_xlim([0, 1])  # 设置x轴的范围A
    axs[1, col].set_ylim([0, 1])  # 设置y轴的范围
    axs[1, col].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 设置x轴刻度格式
    axs[1, col].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))  # 设置y轴刻度格式
    axs[1, col].set_xlabel('Recall', fontsize=18)
    axs[1, col].set_ylabel('Precision', fontsize=18)
    axs[1, col].legend(loc="lower left", fontsize=10)
    axs[1, col].tick_params(axis='both', which='major', labelsize=18)  # 设置刻度字体大小

    # 设置子图为正方形
    axs[1, col].set_aspect('equal', adjustable='box')

# 调整布局
plt.tight_layout()

# 保存图形为PDF文件
pdf_filename = f'ROC_PR.pdf'
plt.savefig(pdf_filename, format='pdf')

