import pandas as pd
import numpy as np
import random
import tiktoken
import dill as pickle
from serialize import SerializerSettings, serialize_arr
from sklearn.preprocessing import StandardScaler
from scalar import get_scaler, truncate_input
import json

seed = 42

random.seed(seed)
np.random.seed(seed)


def get_dataset_from_excel(file_path, column_name, start_idx, end_idx):
    """
    从 Excel 文件加载数据并提取指定列，采样指定范围的数据。

    参数:
        file_path (str): Excel 文件路径。
        column_name (str): 要提取的列名。
        start_idx (int): 数据开始的行索引。
        end_idx (int): 数据结束的行索引。

    返回:
        pd.Series: 提取的时间序列数据。
    """
    # 读取 Excel 文件
    df = pd.read_csv(file_path)
    # 提取指定列并采样指定范围的数据
    series = df[column_name].iloc[start_idx:end_idx]
    return pd.Series(series.values, index=pd.RangeIndex(len(series)))


def preprocess_data(train, settings, model, blocklength):
    """
    对训练和测试数据进行预处理，包括缩放、序列化和截断。
    """
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    # 将数据集转换为 Pandas Series 格式
    if not isinstance(train, pd.Series):
        train = pd.Series(train, index=pd.RangeIndex(len(train)))
    # input_arrs = train.values.reshape(-1, 1)
    # # # 为整个时间序列创建缩放器
    # scaler = StandardScaler()
    # scaler.fit(input_arrs)
    # # 对训练数据进行缩放
    # transformed_input_arrs = np.array(scaler.transform(input_arrs)).flatten()
    # 划分滑动窗口
    arr_slices = generate_sliding_windows(train.values, blocklength)
    # 序列化标准化后的训练数据
    str_slices = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in arr_slices]
    # 计算token数量
    num_tokens = count_tokens(str_slices, model)
    return arr_slices, str_slices, num_tokens


def count_tokens(input_strs, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)
    # Encode each string in the list and return the token IDs
    return [len(encoding.encode(text)) for text in input_strs]

def generate_sliding_windows(data, block_length):
    """
    根据给定的一维 numpy 数组和窗口长度，生成滑动窗口。
    """
    slices = []
    for i in range(len(data) - block_length + 1):
        slices.append(data[i:i + block_length])
    return slices


def get_dataset_by_name(file_path, column_name, start_idx, end_idx, prec, blocklength,
                        testfrac=0.3, model="gpt-4o-mini"):
    """
    根据文件路径和列名加载并预处理单个数据集。

    参数:
        file_path (str): 数据文件路径。
        column_name (str): 数据列名。
        start_idx (int): 数据起始行索引。
        end_idx (int): 数据结束行索引。
        testfrac (float): 测试集占比。
        settings: 序列化设置。
        alpha (float): 缩放器的 alpha 参数。
        beta (float): 缩放器的 beta 参数。
        basic (bool): 是否使用基础缩放器。
        model (str): 使用的模型名称。

    返回:
        dict: 包含训练集、测试集及预处理后数据的字典。
    """
    # 加载数据集
    series = get_dataset_from_excel(file_path, column_name, start_idx, end_idx)

    # 划分训练集和测试集
    splitpoint = int(len(series) * (1 - testfrac))
    train = series.iloc[:splitpoint]
    test = series.iloc[splitpoint:]

    settings = SerializerSettings(prec=prec)
    # 对数据进行预处理
    arr_slices, str_slices, num_tokens = preprocess_data(
        train=train, settings=settings, model=model, blocklength=blocklength
    )

    # 返回处理后的数据集
    return {
        "train": train,
        "test": test,
        "arr_slices": arr_slices,
        "str_slices": str_slices,
        "num_tokens": num_tokens
    }

def save_to_jsonl(input_arrs, input_strs, file_path, shuffle=False):
    """
    将 input_arrs 和 input_strs 转换为 JSONL 格式，并添加 index 字段。

    参数:
        input_arrs (list): numpy.ndarray 转换为列表后的输入数组。
        input_strs (list): 输入字符串列表。
        file_path (str): 保存的 JSONL 文件路径。
        shuffle (bool): 是否打乱数据顺序，默认为 False。
    """
    # 如果 shuffle 为 True，打乱数据顺序
    if shuffle:
        # 为每个元素分配原始的索引值
        indexed_data = [(index, arr, string) for index, (arr, string) in enumerate(zip(input_arrs, input_strs))]
        random.shuffle(indexed_data)  # 打乱数据
        # 一次性解包数据，恢复为原来的 3 个列表
        indices, input_arrs, input_strs = zip(*indexed_data)  # 解包为 indices, input_arrs, input_strs
    else:
        indices = list(range(len(input_arrs)))  # 如果不打乱，直接生成顺序索引

    # 保存 JSONL 文件
    with open(file_path, "w") as f:
        for index, (arr, string) in zip(indices, zip(input_arrs, input_strs)):
            # 创建 JSON 对象
            data = {
                "index": index,
                "input_arr": arr.tolist(),
                "input_str": string
            }
            # 写入文件，每行一个 JSON 对象
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    # 示例用法
    file_path = "../datasets/traffic/traffic.csv"
    column_name = "OT"
    start_idx = 4000
    end_idx = 8000
    block_length = 128
    prec = 4

    data = get_dataset_by_name(file_path, column_name, start_idx, end_idx, prec, block_length)
    print(f"Train shape: {data['train'].shape}, Test shape: {data['test'].shape}")
    print(f"Num of blocks: {len(data['arr_slices'])}")
    print(f"First input: {data['arr_slices'][0]}")
    print(type(data['arr_slices'][0]))
    print(f"First serialized input: {data['str_slices'][0]}")
    print(type(data['str_slices'][0]))
    print("滑动窗口最大的token数是", max(data['num_tokens']))

    save_to_jsonl(data['arr_slices'], data['str_slices'], "../middleware/traffic/blocks.jsonl")
    print("数据已经写入jsonl")
