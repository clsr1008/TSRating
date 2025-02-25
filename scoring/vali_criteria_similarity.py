import json
from scipy.stats import pearsonr
import numpy as np


def load_scores(file_path, aspect_keys):
    """
    加载 JSONL 文件中指定评分维度的分数。

    参数:
        file_path (str): 注释文件路径。
        aspect_keys (list): 评分维度列表。

    返回:
        dict: 包含每个维度分数的字典，key 为维度名称，value 为分数列表。
    """
    scores_dict = {aspect: [] for aspect in aspect_keys}
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            for aspect in aspect_keys:
                key = f"{aspect}_score"
                if key in record:
                    scores_dict[aspect].append(record[key])
    return scores_dict


def compare_aspects(file_path, aspects):
    """
    比较同一个文件中多个评分维度之间的 Pearson 相关系数。

    参数:
        file_path (str): 注释文件路径（JSONL 格式）。
        aspects (list): 评分维度列表。

    返回:
        np.ndarray: Pearson 相关系数矩阵。
    """
    # 加载文件中的所有维度分数
    scores_dict = load_scores(file_path, aspects)

    # 确保每个维度的分数长度一致
    scores_list = [scores_dict[aspect] for aspect in aspects]
    if not all(len(scores) == len(scores_list[0]) for scores in scores_list):
        raise ValueError("文件中的维度分数记录数不一致。")

    # 计算 Pearson 相关系数矩阵
    num_aspects = len(aspects)
    correlation_matrix = np.zeros((num_aspects, num_aspects))
    for i in range(num_aspects):
        for j in range(num_aspects):
            if i <= j:  # Pearson 矩阵是对称的，只计算上三角部分
                correlation, _ = pearsonr(scores_list[i], scores_list[j])
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

    return correlation_matrix


if __name__ == "__main__":
    # 文件路径
    file_path = "../middleware/traffic/annotation.jsonl"

    # 要比较的评分维度
    aspects = ["trend", "frequency", "amplitude", "pattern"]

    # 比较各维度之间的相关性
    correlation_matrix = compare_aspects(file_path, aspects)

    # 打印结果
    print("维度之间的 Pearson 相关系数矩阵：")

    # 设置列宽
    column_width = 12
    header = "".ljust(column_width) + "".join(aspect.ljust(column_width) for aspect in aspects)
    print(header)

    for i, aspect in enumerate(aspects):
        row = aspect.ljust(column_width) + "".join(f"{correlation_matrix[i, j]:.2f}".ljust(column_width) for j in range(len(aspects)))
        print(row)
