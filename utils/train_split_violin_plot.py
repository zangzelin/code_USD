# libraries

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

import wandb

Datasets = ['SWaT', 'WADI', 'PSM', 'MSL', 'SMD']

# 创建子图并设置网格比例
fig, axs = plt.subplots(1, 5, figsize=(25, 6))

# 设置颜色映射
palette = {
    'MTGFLOW': '#1f77b4',
    'USD': '#ff7f0e'
}

Datas = Datas_5seed
for i, dataset in enumerate(Datasets):
    if dataset in ['SWaT', 'WADI', 'PSM', 'MSL', 'SMD']:
        # 将数组合并成DataFrame
        df = pd.DataFrame({
            'train split': ['0.6'] * len(Datas['MTGFLOW'][dataset]['0.6']) + ['0.6'] * len(Datas['USD'][dataset]['0.6']) +
                           ['0.65'] * len(Datas['MTGFLOW'][dataset]['0.65']) + ['0.65'] * len(Datas['USD'][dataset]['0.65']) +
                           ['0.7'] * len(Datas['MTGFLOW'][dataset]['0.7']) + ['0.7'] * len(Datas['USD'][dataset]['0.7']) +
                           ['0.75'] * len(Datas['MTGFLOW'][dataset]['0.75']) + ['0.75'] * len(Datas['USD'][dataset]['0.75']) +
                           ['0.8'] * len(Datas['MTGFLOW'][dataset]['0.8']) + ['0.8'] * len(Datas['USD'][dataset]['0.8']),
            'Subgroup': ['MTGFLOW'] * len(Datas['MTGFLOW'][dataset]['0.6']) + ['USD'] * len(Datas['USD'][dataset]['0.6']) +
                        ['MTGFLOW'] * len(Datas['MTGFLOW'][dataset]['0.65']) + ['USD'] * len(Datas['USD'][dataset]['0.65']) +
                        ['MTGFLOW'] * len(Datas['MTGFLOW'][dataset]['0.7']) + ['USD'] * len(Datas['USD'][dataset]['0.7']) +
                        ['MTGFLOW'] * len(Datas['MTGFLOW'][dataset]['0.75']) + ['USD'] * len(Datas['USD'][dataset]['0.75']) +
                        ['MTGFLOW'] * len(Datas['MTGFLOW'][dataset]['0.8']) + ['USD'] * len(Datas['USD'][dataset]['0.8']),
            'AUROC%': Datas['MTGFLOW'][dataset]['0.6'] + Datas['USD'][dataset]['0.6'] +
                      Datas['MTGFLOW'][dataset]['0.65'] + Datas['USD'][dataset]['0.65'] +
                      Datas['MTGFLOW'][dataset]['0.7'] + Datas['USD'][dataset]['0.7'] +
                      Datas['MTGFLOW'][dataset]['0.75'] + Datas['USD'][dataset]['0.75'] +
                      Datas['MTGFLOW'][dataset]['0.8'] + Datas['USD'][dataset]['0.8']
        })

        axs[i].set_title(dataset, fontsize=24)
        # 绘制小提琴图并应用颜色映射
        sns.violinplot(x="train split", y='AUROC%', hue='Subgroup', data=df, ax=axs[i], split=False, linewidth=1.5,
                       palette=palette, legend=False, saturation=1, alpha=1)
        axs[i].get_legend().remove()  # 移除子图内的图例
        # 设置轴标签文字大小
        axs[i].set_xlabel('train split', fontsize=20)  # 设置横轴标题和文字大小
        axs[i].set_ylabel('AUROC%', fontsize=20)  # 设置竖轴标题和文字大小

        # 设置刻度标签文字大小
        axs[i].set_xticklabels(axs[i].get_xticklabels(), fontsize=20)
        axs[i].set_yticklabels(axs[i].get_yticks(), fontsize=20)

# 调整子图间的间隔，留出顶部空间用于放置图例
plt.subplots_adjust(top=0.8)

# 在最后一个子图的上方添加图例
legend_handles = [Patch(color=palette['MTGFLOW'], label='MTGFLOW'), Patch(color=palette['USD'], label='USD')]
axs[-1].legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.3), fontsize=20, ncol=len(legend_handles),
               frameon=True)
plt.tight_layout()

# 保存图形为PDF文件
pdf_filename = "trainsplit violin.pdf"
fig.savefig(pdf_filename, format='pdf')



plt.close()

