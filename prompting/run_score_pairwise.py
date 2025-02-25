import subprocess
from datasets import Dataset
import json

def run_score_pairwise(input_path, output_path):
    # 构造命令
    command = [
        "python", "score_pairwise.py", input_path, output_path,
        # "-n", "10", #要从数据集中选取多少个样本进行比较
        "--num_examples_proportion", "1.0",
        "-k", "2",  #2 表示进行成对比较
        "--model", "gpt-4o-mini",
        "--generations", "20",
        "--template_file", "templates/pairwise_trend.txt",
        "--tokens_min", "512", #每个文本至少包含 512 个 tokens
        "--tokens_max", "1024", #每个文本最多包含 512 个 tokens
        "--ratio", "1.0", #有效pairwise数量是block数多少倍
        "--text_field", "input_str"  # 表示每个数据示例中包含文本数据的字段名
    ]

    if json_flag:  #原本运行没有这个标志，读取的是一个数据集而不是单个文件
        command.append("--json")

    if out_format:  #原本没有此标志，设置让输出简洁
        command.append("--flat_output_format")

    # 执行命令
    subprocess.run(command)

# 转换为 Excel 文件函数
def convert_to_excel(dataset_path, excel_path, jsonl_path):
    """
    根据 JSONL 文件中的索引数据，给 Excel 表格增加 block_a 和 block_b 列。

    参数:
        dataset_path (str): 原数据集的路径。
        excel_path (str): 生成的 Excel 文件路径。
        jsonl_path (str): JSONL 文件路径，提供 index 数据。
    """
    # 加载数据集
    dataset = Dataset.load_from_disk(dataset_path)

    # 转换为 Pandas DataFrame
    df = dataset.to_pandas()
    # 打开 JSONL 文件并读取索引数据
    with open(jsonl_path, "r") as f:
        # 读取 JSONL 文件并提取所有索引
        indices = []
        for line in f:
            data = json.loads(line)
            indices.append(data['index'])
    # 保存为 Excel 文件
    df.to_excel(excel_path, index=False)
    print(f"数据已保存为 Excel 文件: {excel_path}")

# 示例用法
if __name__ == "__main__":
    input_path = "../middleware/traffic/blocks.jsonl"  # 替换为你的输入路径
    output_path = "../middleware/traffic/pairwise_trend"  # 替换为你的输出路径

    json_flag = True  # 设置 json 标志为 True
    out_format = True
    run_score_pairwise(input_path, output_path)
    # 转换生成的数据集为 Excel 文件
    excel_output_path = "../middleware/traffic/pairwise_trend.xlsx"  # Excel 文件保存路径
    convert_to_excel(output_path, excel_output_path, input_path)
