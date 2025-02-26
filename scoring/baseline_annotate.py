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
    """Split the time series into completely overlapping blocks."""
    step = 1
    blocks = [series[i:i + block_length] for i in range(0, len(series) - block_length + 1, step)]
    return np.array(blocks)


def evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """Evaluate and score each training block."""
    fetcher = DataFetcher.from_data_splits(X_train, Y_train, X_val, Y_val, X_test, Y_test, one_hot=False)
    pred_model = RegressionSkLearnWrapper(LinearRegression)
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
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    series = data[column_name].iloc[start_idx:end_idx].values

    train_size = int(len(series) * 0.7)
    valid_size = int(len(series) * 0.1)
    train_series = series[:train_size]
    val_series = series[train_size:train_size + valid_size]
    test_series = series[train_size + valid_size:]

    train_blocks = split_time_series_blocks(train_series, block_length=block_length)
    X_train = train_blocks[:, :-1]
    Y_train = train_blocks[:, -1]

    val_blocks = split_time_series_blocks(val_series, block_length=block_length)
    test_blocks = split_time_series_blocks(test_series, block_length=block_length)

    X_val = val_blocks[:, :-1]
    Y_val = val_blocks[:, -1]
    X_test = test_blocks[:, :-1]
    Y_test = test_blocks[:, -1]


    scores = evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    block_scores = {method: [] for method in scores.keys()}
    for method, values in scores.items():
        block_scores[method] = values


    print("Evaluation complete, scores for each block:")
    for method, values in block_scores.items():
        print(f"{method}: {values}")
        print(len(values))

    with open(output_file, "r") as jsonl_file:
        original_data = [json.loads(line) for line in jsonl_file]
    for idx, record in enumerate(original_data):
        record.update({
            "DataOob": float(block_scores["DataOob"][idx]),
            "DataShapley": float(block_scores["DataShapley"][idx]),
            "KNNShapley": float(block_scores["KNNShapley"][idx]),
            "TimeInf": float(block_scores["TimeInf"][idx]),
        })
    with open(output_file, "w") as jsonl_file:
        jsonl_file.writelines(json.dumps(record) + "\n" for record in original_data)

    print(f"results have been saved to {output_file}")


if __name__ == "__main__":
    # Example usage with replaceable parameters
    parser = argparse.ArgumentParser(description="Program for evaluating time series blocks")
    parser.add_argument("--file_path", type=str, help="Path to the time series data file")
    parser.add_argument("--column_name", type=str, help="Column name")
    parser.add_argument("--start_idx", type=int, help="Start index")
    parser.add_argument("--end_idx", type=int, help="End index")
    parser.add_argument("--output_file", type=str, default="results.jsonl", help="Path to the output JSONL file")
    parser.add_argument("--block_length", type=int, default=128, help="Block length")

    args = parser.parse_args()

    args.file_path = "../datasets/traffic/traffic.csv"  # to be changed
    args.column_name = "OT"  # to be changed
    args.start_idx = 4000  # to be changed
    args.end_idx = 8000  # to be changed
    args.block_length = 128  # to be changed
    args.output_file = "../middleware/traffic/annotation.jsonl"  # to be changed

    block_scores = main(args.file_path, args.column_name, args.start_idx, args.end_idx, args.output_file, args.block_length)
