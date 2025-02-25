import numpy as np
import pandas as pd
import json
from opendataval.dataloader import DataFetcher
from opendataval.dataval import DataOob, DataShapley, KNNShapley
from sklearn.base import BaseEstimator, clone
import argparse
from opendataval.model import RegressionSkLearnWrapper
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from timeinf import calc_linear_time_inf
from tqdm import tqdm
import random
import time

def split_time_series_blocks(series, block_length):
    """将时间序列分成完全重叠的块."""
    step = 1  # 完全重叠时步长为 1
    blocks = [series[i:i + block_length] for i in range(0, len(series) - block_length + 1, step)]
    return np.array(blocks)


def evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """对每个训练块评估并打分（分类任务版本）."""

    # 确保标签是整数类型，不需要 one-hot 编码
    fetcher = DataFetcher.from_data_splits(X_train, Y_train, X_val, Y_val, X_test, Y_test, one_hot=False)

    # 分类模型替换为逻辑回归
    pred_model = RegressionSkLearnWrapper(LinearRegression)

    # 初始化结果字典
    scores = {"DataOob": [], "DataShapley": [], "KNNShapley": [], "TimeInf": []}

    # DataOob
    start_time = time.time()
    data_oob = DataOob(num_models=1000).train(fetcher=fetcher, pred_model=pred_model)
    end_time = time.time()
    scores["DataOob"] = data_oob.data_values
    print(f"DataOob method execution time: {end_time - start_time:.4f} seconds")

    # DataShapley
    start_time = time.time()
    data_shapley = DataShapley(gr_threshold=1.1, max_mc_epochs=100, min_models=100).train(
        fetcher=fetcher, pred_model=pred_model
    )
    end_time = time.time()
    scores["DataShapley"] = data_shapley.data_values
    print(f"DataShapley method execution time: {end_time - start_time:.4f} seconds")

    # KNNShapley
    start_time = time.time()
    knn_shapley = KNNShapley(k_neighbors=0.1 * len(X_train)).train(fetcher=fetcher)
    end_time = time.time()
    scores["KNNShapley"] = knn_shapley.data_values
    print(f"KNNShapley method execution time: {end_time - start_time:.4f} seconds")


    # TimeInf: Compute time-based influences using LOO approximation
    lr = LinearRegression().fit(X_train, Y_train)
    beta = lr.coef_
    b = lr.intercept_
    try:
        inv_hess = len(X_train) * np.linalg.inv(X_train.T @ X_train)
    except:
        inv_hess = len(X_train) * np.linalg.pinv(X_train.T @ X_train)
    params = (beta, b, inv_hess)
    start_time = time.time()
    time_block_loos = []
    for i in tqdm(range(len(X_train)), total=len(X_train), desc="Compute LOO"):
        val_influences = 0
        for j in range(len(X_val)):
            val_influences += calc_linear_time_inf(i, j, X_train, Y_train, X_val, Y_val, params)
        time_block_loos.append(val_influences / len(X_val))
    time_block_loos = np.array(time_block_loos)

    scores["TimeInf"] = time_block_loos
    end_time = time.time()
    print(f"TimeInf method execution time: {end_time - start_time:.4f} seconds")

    return scores



def main(file_path, column_name, start_idx, end_idx, output_file, block_length):
    # 读取数据
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    series = data[column_name].iloc[start_idx:end_idx].values

    # 按 7:1:2 划分数据
    train_size = int(len(series) * 0.7)
    valid_size = int(len(series) * 0.1)
    train_series = series[:train_size]
    val_series = series[train_size:train_size + valid_size]
    test_series = series[train_size + valid_size:]

    # 切分训练集为块
    train_blocks = split_time_series_blocks(train_series, block_length=block_length)  # 完全重叠
    X_train = train_blocks[:, :-1]
    Y_train = train_blocks[:, -1]

    # 验证集和测试集
    val_blocks = split_time_series_blocks(val_series, block_length=block_length)  # 完全重叠
    test_blocks = split_time_series_blocks(test_series, block_length=block_length)  # 完全重叠

    X_val = val_blocks[:, :-1]
    Y_val = val_blocks[:, -1]
    X_test = test_blocks[:, :-1]
    Y_test = test_blocks[:, -1]

    print("训练集大小:", X_train.shape)
    print("验证集大小:", X_val.shape)
    print("测试集大小:", X_test.shape)

    # 评估每个块
    scores = evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    # 为每个块的下标分配评分
    block_scores = {method: [] for method in scores.keys()}
    for method, values in scores.items():
        block_scores[method] = values


    print("评估完成，每个块的评分:")
    for method, values in block_scores.items():
        print(f"{method}: {values}")
        print(len(values))

    # 保存结果到 JSONL 文件
    # 读取原始 JSONL 文件内容
    with open(output_file, "r") as jsonl_file:
        original_data = [json.loads(line) for line in jsonl_file]

    # 追加新评分字段
    for idx, record in enumerate(original_data):
        record.update({
            "DataOob": float(block_scores["DataOob"][idx]),
            "DataShapley": float(block_scores["DataShapley"][idx]),
            "KNNShapley": float(block_scores["KNNShapley"][idx]),
            "TimeInf": float(block_scores["TimeInf"][idx]),
        })

    # 写入更新后的数据到新 JSONL 文件
    with open(output_file, "w") as jsonl_file:
        jsonl_file.writelines(json.dumps(record) + "\n" for record in original_data)

    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估时间序列块的程序")
    parser.add_argument("--file_path", type=str, help="时间序列数据文件路径")
    parser.add_argument("--column_name", type=str, help="列名")
    parser.add_argument("--start_idx", type=int, help="起始索引")
    parser.add_argument("--end_idx", type=int, help="结束索引")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="输出的 JSONL 文件路径")
    parser.add_argument("--block_length", type=int, default=128, help="块长度")
    args = parser.parse_args()

    args.file_path = "../datasets/traffic/traffic.csv"
    args.column_name = "OT"
    args.start_idx = 4000
    args.end_idx = 8000
    args.block_length = 128
    args.output_file = "../middleware/traffic/annotation.jsonl"

    block_scores = main(args.file_path, args.column_name, args.start_idx, args.end_idx, args.output_file, args.block_length)
