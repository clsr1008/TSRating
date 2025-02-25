import json
import torch
from momentfm import MOMENTPipeline
import argparse
import pandas as pd
from trainer import PairwiseDataset, ScoreModel, train_model, evaluate_model
from torch.utils.data import random_split, Subset
import random
import numpy as np

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)  # 设置 Python 随机数种子
    np.random.seed(seed)  # 设置 NumPy 随机数种子
    torch.manual_seed(seed)  # 设置 PyTorch 随机数种子
    torch.cuda.manual_seed(seed)  # 如果使用 GPU，设置 CUDA 随机数种子
    torch.backends.cudnn.deterministic = True  # 确保 CUDA 使用确定性操作
    torch.backends.cudnn.benchmark = False  # 禁用非确定性优化算法

# 加载 MOMENT 模型
def load_moment_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={'task_name': 'embedding'}
    )
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters in the encoder: {num_params}")
    model.init()  # 初始化模型
    model.eval()  # 切换到推理模式
    return model

# 从 JSONL 文件中加载数据
def load_jsonl_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 筛选置信度高的样本
def filter_pairwise_data(pairwise_df):
    """
    筛选比较数据集中置信度高的样本
    条件: comparisons_avg 小于 0.25 或大于 0.75
    """
    filtered_df = pairwise_df[
        (pairwise_df["comparisons_avg"] >= 0) &
        (pairwise_df["comparisons_avg"] <= 1) &
        ((pairwise_df["comparisons_avg"] <= 0.25) | (pairwise_df["comparisons_avg"] >= 0.75))
    ]

    ratio = len(filtered_df) / len(pairwise_df) * 100

    # 获取过滤后的样本编号（Excel 中的行号，从 1 开始）
    # filtered_indices = filtered_df.index + 1  # 将 Pandas 索引转为从 1 开始的编号
    print(f"Filtered dataset: {len(filtered_df)} samples remaining ({ratio:.2f}% of original dataset).")
    # print(f"Filtered sample indices: {filtered_indices.tolist()}")  # 打印编号列表

    return filtered_df

# 处理时序数据并进行表征学习
def process_time_series(model, data):
    embeddings_dict = {}  # 用于存储 index 和对应的 embedding

    input_tensors = []
    indices = []

    # 提取所有 input_arr 和 index
    for record in data:
        input_arr = record.get("input_arr")
        index = record.get("index")
        if input_arr is not None and index is not None:
            input_tensors.append(input_arr)
            indices.append(index)

    # 将所有样本堆叠成一个 Tensor，形状为 [batch_size, n_channels, context_length]
    input_tensor = torch.tensor(input_tensors, dtype=torch.float32).unsqueeze(1)  # 单通道
    print(f"Input tensor shape: {input_tensor.shape}")  # 应为 [batch_size, n_channels, context_length]

    # 使用 MOMENT 模型提取表征
    output = model(x_enc=input_tensor)

    # 提取表征特征
    embeddings = output.embeddings  # 提取 embeddings
    print(f"Embeddings shape: {embeddings.shape}")  # 应为 [batch_size, n_channels, context_length]

    # 将每个 index 和对应的 embedding 存入字典
    for idx, embedding in zip(indices, embeddings):
        embeddings_dict[idx] = embedding  # 直接存储 PyTorch Tensor 格式

    return embeddings_dict


def create_subset_dataset(full_dataset, retain_ratio=0.5):
    if not (0 < retain_ratio <= 1):
        raise ValueError("retain_ratio 必须在 (0, 1] 范围内")
    total_size = len(full_dataset)
    retain_size = int(total_size * retain_ratio)
    indices = torch.randperm(total_size).tolist()[:retain_size]  # 转为整数列表
    return Subset(full_dataset, indices)

# 主函数
def main(jsonl_path, pairwise_path, output_model_path):
    # Step 0: 设置随机种子
    set_seed(42)

    # Step 1: 加载 MOMENT 模型
    model = load_moment_model()

    # Step 2: 加载 JSONL 数据
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} records from {jsonl_path}")

    # Step 3: 使用 MOMENT 模型生成特征
    embeddings_dict = process_time_series(model, data)
    print(f"Generated embeddings for {len(embeddings_dict)} records")

    # Step 4: 加载 pairwise 数据集
    pairwise_df = pd.read_excel(pairwise_path)
    print(f"Loaded pairwise dataset with {len(pairwise_df)} pairs.")

    # Step 5: 筛选置信度高的样本
    pairwise_df = filter_pairwise_data(pairwise_df)

    # Step 6: 构建数据集
    dataset = PairwiseDataset(embeddings_dict, pairwise_df)
    # 使用函数随机保留 50% 的样本
    # dataset = create_subset_dataset(full_dataset, retain_ratio=0.9)
    # print(len(dataset))

    # 按 80% 训练集和 20% 测试集划分
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Step 7: 初始化评分模型
    input_dim = next(iter(embeddings_dict.values())).shape[0]
    score_model = ScoreModel(input_dim)

    # Step 8: 训练模型
    score_model = train_model(score_model, train_dataset, epochs=20, batch_size=64, lr=0.005)

    # Step 9: 测试模型
    evaluate_model(score_model, test_dataset)

    # Step 10: 保存模型参数
    torch.save(score_model.state_dict(), output_model_path)
    print(f"Model parameters saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="时间序列模型评分程序")
    parser.add_argument("--jsonl_path", type=str, help="输入 JSONL 文件路径")
    parser.add_argument("--pairwise_path", type=str, help="输入 pairwise 数据集路径")
    parser.add_argument("--output_model_path", type=str, help="输出模型路径")

    args = parser.parse_args()

    args.jsonl_path = "../middleware/traffic/blocks.jsonl"
    args.pairwise_path = "../middleware/traffic/pairwise_trend.xlsx"
    args.output_model_path = "../middleware/traffic/rater_trend.pth"

    main(
        jsonl_path=args.jsonl_path,
        pairwise_path=args.pairwise_path,
        output_model_path=args.output_model_path
    )
