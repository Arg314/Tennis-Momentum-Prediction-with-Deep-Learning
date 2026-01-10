# 基于深度学习的 ATP 网球比赛动量预测

利用 RNN、LSTM、GRU 和 Transformer 模型，对职业网球比赛中的“动量”变化进行建模与预测。

## 项目简介

本项目旨在通过点对点（point-by-point）的比赛序列数据，建模并预测职业网球比赛中动量（momentum）的转移。我们从原始 ATP 比赛日志出发，经过清洗与特征工程，构建结构化的时间序列数据集，并训练多种深度学习模型（RNN、LSTM、GRU、Transformer）来预测短期内是否会发生动量变化。

清洗后的数据集已公开托管在 Hugging Face Datasets Hub，便于复现与社区共享。

- Hugging Face 数据集地址：  
your-username/atp-momentum-prediction


## 项目结构

tennis-momentum/
├── src/                     # 核心代码
│   ├── data_preprocessing.py  # 数据加载与清洗
│   ├── feature_engineering.py # 特征构造与序列化
│   ├── models.py            # 所有模型定义（LSTM/GRU/Transformer等）
│   ├── train.py             # 训练逻辑
│   ├── evaluate.py          # 评估指标计算
│   └── utils.py             # 工具函数
├── data/                    # 本地数据
│   ├── raw/                 # 原始数据
│   └── processed/           # 清洗后数据
├── trained_models/          # 训练好的模型权重（.pth）
├── config.yaml              # 超参数配置文件
├── main.py                  # 主程序入口
├── app/                     # 可视化仪表盘（Streamlit）
│   └── dashboard.py
├── notebooks/               # 探索性数据分析
├── README.md                # ← 当前文档
└── requirements.txt         # 依赖库列表

## 数据集说明

数据来源
- 原始数据来自https://www.kaggle.com/datasets/colinparker/pointbypoint-bo3-tennis-data-2011-2017
- 覆盖 13,050 场 ATP 男子单打比赛（2011–2017 年）。

预处理流程
- 解析每一分的结果（如 'S'=发球直接得分，'R'=接发球回球，'A'=Ace 球等）
- 基于胜率波动定义“动量转移”标签（binary label）
- 过滤无效或不完整比赛
- 编码球员信息、发球方状态
- 将比赛切分为固定长度的滑动窗口序列（例如最近 20 分）

清洗结果
- 原始数据：13,050 场比赛  
- 清洗后有效数据：13,050 场比赛（无缺失，全部保留）

字段说明（清洗后 Parquet 文件）
字段名   类型   说明
match_id   str   比赛唯一标识符

server1, server2   str   两位球员姓名

parsed_points   list[str]   点序列（如 ['S', 'R', 'D', ...]）

n_points   int   本场比赛总分数

momentum_labels   list[int]   动量标签（0/1，1 表示发生动量转移）

模型架构

本项目实现了以下 5 种神经网络模型：
模型名称   结构特点   输出
TennisRNN   2 层 RNN + 标量特征融合   单步动量概率

TennisLSTM   2 层 LSTM + 标量特征融合   单步动量概率

TennisGRU   2 层 GRU + 标量特征融合   单步动量概率

TennisTransformer   Transformer 编码器（d_model=32, nhead=2）   单步动量概率

MultiStepLSTM   LSTM + 多步预测头（horizon=3）   未来 3 步动量概率

所有模型均将序列特征（历史得分序列）与标量特征（如当前局分、发球优势等）进行融合，提升预测能力。

## 快速开始

1. 安装依赖
pip install -r requirements.txt

2. 单模型训练与评估（演示模式，仅用前 300 场）
python main.py --model LSTM

3. 批量训练多个模型（使用全部 13,050 场比赛）
python main.py --models RNN LSTM GRU Transformer --use_full_data

4. 从 Hugging Face 加载数据集
from datasets import load_dataset

自动下载并加载
dataset = load_dataset("your-username/atp-momentum-prediction")
df = dataset["train"].to_pandas()

print(df.head())

5. 启动可视化仪表盘（可选）
streamlit run app/dashboard.py

评估指标

- step_1_acc：下一拍动量预测准确率
- auc：ROC 曲线下面积（衡量分类器整体性能）
- f1：F1 分数（适用于标签不平衡场景）
