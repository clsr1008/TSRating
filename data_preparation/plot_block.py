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
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    indices_set = set(indices)  # 使用集合加速查找
    found_indices = set()  # 记录已找到的索引

    # 逐行读取 JSONL 文件
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


if __name__ == "__main__":
    # 指定 JSONL 文件路径、需要绘制的索引列表和输出目录
    file_path = "../middleware/electricity/annotation.jsonl"
    indices = [2547, 9, 74, 1, 59]  # 你要绘制的索引列表
    output_dir = "../middleware/electricity/plots"  # 图片保存目录

    # 调用函数绘制并保存图像
    plot_blocks_from_jsonl(file_path, indices, output_dir)
