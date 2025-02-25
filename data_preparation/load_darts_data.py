import darts.datasets
import pandas as pd
import numpy as np
import tiktoken
import random
import dill as pickle

from serialize import SerializerSettings, serialize_arr
from scalar import get_scaler, truncate_input
import json

seed = 42

random.seed(seed)
np.random.seed(seed)

def get_dataset(dsname):
    """加载指定的数据集并进行预处理"""
    darts_ds = getattr(darts.datasets, dsname)().load()
    if dsname == 'GasRateCO2Dataset':
        # 处理多变量数据集，仅选择某一列
        darts_ds = darts_ds[darts_ds.columns[1]]
    # 转换为 Pandas Series 格式
    series = darts_ds.pd_series()
    if dsname == 'SunspotsDataset':
        # 针对特定数据集进行下采样
        series = series.iloc[::4]
    if dsname == 'HeartRateDataset':
        # 针对特定数据集进行下采样
        series = series.iloc[::2]
    return series

def preprocess_data(train, settings, alpha, beta, basic, model, blocklength):
    """
    对训练和测试数据进行预处理，包括缩放、序列化和截断。
    """
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    # 将数据集转换为 Pandas Series 格式
    if not isinstance(train, pd.Series):
        train = pd.Series(train, index=pd.RangeIndex(len(train)))
    # 为整个时间序列创建缩放器
    scaler = get_scaler(train.values, alpha=alpha, beta=beta, basic=basic)
    # 对训练数据进行缩放
    input_arrs = train.values
    transformed_input_arrs = np.array(scaler.transform(input_arrs))
    # 划分滑动窗口
    arr_slices = generate_sliding_windows(transformed_input_arrs, blocklength)
    # 序列化标准化后的训练数据
    str_slices = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in arr_slices]
    # 确保输入序列不超过模型的上下文长度,暂时不需要截断
    # input_arrs, input_strs = zip(*[
    #     truncate_input(input_array, input_str, settings, model, test_len)
    #     for input_array, input_str in zip(input_arrs, input_strs)
    # ])
    # 计算token数量
    num_tokens = count_tokens(str_slices, model)
    return arr_slices, str_slices, scaler, num_tokens

def count_tokens(input_strs, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)
    # Encode each string in the list and return the token IDs
    return [len(encoding.encode(text)) for text in input_strs]

def generate_sliding_windows(data, block_length):
    """
    根据给定的一维 numpy 数组和窗口长度，生成滑动窗口。

    参数:
        data (numpy.ndarray): 输入的一维时间序列数据。
        block_length (int): 滑动窗口的长度。

    返回:
        list: 包含滑动窗口的列表，每个窗口是一个 numpy 数组。
    """
    slices = []
    for i in range(len(data) - block_length + 1):
        slices.append(data[i:i + block_length])
    return slices


def get_dataset_by_name(dsname, testfrac=0.2, settings=SerializerSettings(), alpha=0.95, beta=0.7, basic=True,
                        blocklength=25, model="gpt-4o-mini"):
    """
    根据数据集名称加载并预处理单个数据集。

    参数:
        dsname (str): 数据集名称。
        testfrac (float): 测试集占比。
        settings: 序列化设置。
        alpha (float): 缩放器的 alpha 参数。
        beta (float): 缩放器的 beta 参数。
        basic (bool): 是否使用基础缩放器。
        model (str): 使用的模型名称。

    返回:
        dict: 包含训练集、测试集及预处理后数据的字典。
    """
    datasets = [
        'AirPassengersDataset',
        'AusBeerDataset',
        'GasRateCO2Dataset',  # 多变量数据集
        'MonthlyMilkDataset',
        'SunspotsDataset',  # 非常大的数据集，需下采样
        'WineDataset',
        'WoolyDataset',
        'HeartRateDataset',
    ]
    # 加载数据集
    series = get_dataset(dsname)

    # 划分训练集和测试集
    splitpoint = int(len(series) * (1 - testfrac))
    train = series.iloc[:splitpoint]
    test = series.iloc[splitpoint:]

    # 对数据进行预处理
    arr_slices, str_slices, scaler, num_tokens = preprocess_data(
        train=train, settings=settings, alpha=alpha, beta=beta, basic=basic, model=model, blocklength=blocklength
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
    ds_name = "AirPassengersDataset"

    data = get_dataset_by_name(ds_name)
    print(f"Dataset: {ds_name}")
    print(f"Train shape: {data['train'].shape}, Test shape: {data['test'].shape}")
    print(f"Train: {data['train']}")
    print(f"First input: {data['arr_slices'][0]}")
    print(type(data['arr_slices'][0]))
    print(f"First serialized input: {data['str_slices'][0]}")
    print(type(data['str_slices'][0]))
    print("滑动窗口最大的token数是", max(data['num_tokens']))
    save_to_jsonl(data['arr_slices'], data['str_slices'], "../prompting/input_data/tsdata_example.jsonl")
    print("数据已经写入tsdata_example.jsonl")

    # 保存数据
    with open("air_passengers_data.pkl", "wb") as f:
        pickle.dump(data, f)
