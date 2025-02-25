import json
from scipy.stats import spearmanr


def compare_scores(file1_path, file2_path, aspect="trend"):
    """
    比较两个注释文件中某个评分维度的分数相似性。

    参数:
        file1_path (str): 第一个注释文件的路径（JSONL 格式）。
        file2_path (str): 第二个注释文件的路径（JSONL 格式）。
        aspect (str): 需要比较的评分维度名称。

    返回:
        float: Spearman相关系数，用于衡量两个文件中该维度分数的相似性。
    """

    # Step 1: 加载两个 JSONL 文件
    def load_scores(file_path, aspect_key):
        scores = []
        with open(file_path, "r") as f:
            for line in f:
                record = json.loads(line.strip())
                if f"{aspect_key}_score" in record:
                    scores.append(record[f"{aspect_key}_score"])
        return scores

    # Step 2: 读取两个文件中的分数
    scores1 = load_scores(file1_path, aspect)
    scores2 = load_scores(file2_path, aspect)

    # 检查两个文件中记录数是否一致
    if len(scores1) != len(scores2):
        raise ValueError("两个文件中记录数不一致，无法比较。")

    # Step 3: 计算 Spearman 相关系数
    spearman_corr, _ = spearmanr(scores1, scores2)

    return spearman_corr


if __name__ == "__main__":
    # 文件路径
    file1_path = "../middleware/electricity/annotation_0.5.jsonl"
    file2_path = "../middleware/electricity/annotation.jsonl"

    # 比较评分维度的相似性
    aspect = "trend"  # 选择要比较的评分维度
    similarity = compare_scores(file1_path, file2_path, aspect)
    print(f"{aspect} 维度的 Spearman 相关系数为: {similarity}")
