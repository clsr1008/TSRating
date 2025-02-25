import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def plot_blocks_from_jsonl(file_path, indices, output_dir):
    """
    根据索引列表绘制 JSONL 文件中的对应数据块，并保存到指定目录。

    参数:
        file_path (str): JSONL 文件的路径。
        indices (list): 需要绘制的块的索引列表（如 [3, 7, 15]）。
        output_dir (str): 保存图片的目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
    indices_set = set(indices)  # 使用集合加速查找
    found_indices = set()  # 记录已找到的索引

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            block_index = data['index']  # 读取当前数据的索引

            if block_index in indices_set:
                input_arr = np.array(data['input_arr'])

                # 检查 input_arr 的形状并保存图像
                if input_arr.ndim == 1:  # 确保是一维数据
                    plt.figure()
                    plt.plot(input_arr)  # 使用折线图绘制
                    plt.title(f"Block at index {block_index}")
                    plt.xlabel("Step")
                    plt.ylabel("Value")

                    # 保存图片
                    output_path = os.path.join(output_dir, f"block_{block_index}.png")
                    plt.savefig(output_path)
                    plt.close()

                    print(f"Saved plot for block {block_index} at {output_path}")
                    found_indices.add(block_index)  # 记录已找到的索引

                else:
                    print(f"Data at index {block_index} is not a 1D array for plotting.")

                # 如果所有索引都已经找到，就提前结束循环，提高效率
                if found_indices == indices_set:
                    break

    # 检查是否有未找到的索引
    missing_indices = indices_set - found_indices
    if missing_indices:
        print(f"Warning: The following indices were not found in the file: {sorted(missing_indices)}")


def analyze_scores(input_jsonl_path, output_base_dir):
    """
    读取 JSONL 文件，获取评分最高的前 5 个索引和最低的后 5 个索引，并绘制对应的图。

    Args:
        input_jsonl_path (str): 输入的 JSONL 文件路径。
        output_base_dir (str): 输出图片的根目录。
    """
    print(f"Loading data from {input_jsonl_path}...")
    records = []

    with open(input_jsonl_path, "r") as infile:
        for line in infile:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line.strip())

    print(f"Loaded {len(records)} records.")

    # 需要分析的维度
    dimensions = ["trend_score", "frequency_score", "amplitude_score", "pattern_score"]

    for dimension in dimensions:
        print(f"\nAnalyzing {dimension}...")

        # 过滤掉没有该维度评分的数据，并进行排序
        valid_records = [r for r in records if r.get(dimension) is not None]
        sorted_records = sorted(valid_records, key=lambda x: x[dimension], reverse=True)

        # 获取前 5 和后 5 个索引
        top_20 = [r["index"] for r in sorted_records[:20]]
        bottom_20 = [r["index"] for r in sorted_records[-20:]]

        # 打印索引
        print(f"  Top 20 {dimension} indices: {top_20}")
        print(f"  Bottom 20 {dimension} indices: {bottom_20}")

        # 创建 `top_5` 和 `bottom_5` 目录
        top_20_dir = os.path.join(output_base_dir, dimension, "top_20")
        bottom_20_dir = os.path.join(output_base_dir, dimension, "bottom_20")
        os.makedirs(top_20_dir, exist_ok=True)
        os.makedirs(bottom_20_dir, exist_ok=True)

        # 画图并保存
        plot_blocks_from_jsonl(input_jsonl_path, top_20, top_20_dir)
        plot_blocks_from_jsonl(input_jsonl_path, bottom_20, bottom_20_dir)


if __name__ == "__main__":
    input_jsonl_path = "../middleware/electricity/annotation.jsonl"  # 输入文件路径
    output_base_dir = "../middleware/electricity/plots"  # 图片保存根目录

    analyze_scores(input_jsonl_path, output_base_dir)
