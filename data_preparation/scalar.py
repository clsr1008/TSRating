import tiktoken
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from dataclasses import dataclass

STEP_MULTIPLIER = 1.2


@dataclass
class Scaler:
    """
    Represents a data scaler with transformation and inverse transformation functions.

    Attributes:
        transform (callable): Function to apply transformation.
        inv_transform (callable): Function to apply inverse transformation.
    """
    transform: callable = lambda x: x
    inv_transform: callable = lambda x: x


def get_scaler(history, alpha=0.95, beta=0.3, basic=False):
    """
    Generate a Scaler object based on given history data.

    Args:
        history (array-like): Data to derive scaling from.
        alpha (float, optional): Quantile for scaling. Defaults to .95.
        # Truncate inputs
        tokens = [tokeniz]
        beta (float, optional): Shift parameter. Defaults to .3.
        basic (bool, optional): If True, no shift is applied, and scaling by values below 0.01 is avoided. Defaults to False.

    Returns:
        Scaler: Configured scaler object.
    """
    history = history[~np.isnan(history)]  # 去除 NaN 值 #True
    if basic:  # 基础缩放
        q = np.maximum(np.quantile(np.abs(history), alpha), .01)
        # 取绝对值的分位数（alpha 指定，如 95%），并保证缩放因子 q 不低于 0.01
        def transform(x):
            return x / q

        def inv_transform(x):
            return x * q
        # 转换函数 transform 将数据除以 q，反向转换函数 inv_transform 乘以 q
    else:  # 带偏移缩放
        min_ = np.min(history) - beta * (np.max(history) - np.min(history))
        # 计算最小值 min_，偏移范围由 beta 控制
        q = np.quantile(history - min_, alpha)
        if q == 0:
            q = 1

        def transform(x):
            return (x - min_) / q

        def inv_transform(x):
            return x * q + min_
        # 转换函数 transform 执行偏移并除以 q，反向转换函数 inv_transform 恢复偏移
    return Scaler(transform=transform, inv_transform=inv_transform)
    # 返回一个 Scaler 对象，包含 transform 和 inv_transform 两个方法


def truncate_input(input_arr, input_str, settings, model, steps):
    """
    Truncate inputs to the maximum context length for a given model.

    Args:
        input (array-like): input time series.
        input_str (str): serialized input time series.
        settings (SerializerSettings): Serialization settings.
        model (str): Name of the LLM model to use.
        steps (int): Number of steps to predict.
    Returns:
        tuple: Tuple containing:
            - input (array-like): Truncated input time series.
            - input_str (str): Truncated serialized input time series.
    """
    context_length = context_lengths[model]
    # 根据模型名称获取对应的分词方式和允许的最大令牌数量
    input_str_chuncks = input_str.split(settings.time_sep)
    # 分割序列化输入
    for i in range(len(input_str_chuncks) - 1):  # 逐步截断
        truncated_input_str = settings.time_sep.join(input_str_chuncks[i:])
        # add separator if not already present
        if not truncated_input_str.endswith(settings.time_sep):
            truncated_input_str += settings.time_sep  ##将从第 i 块开始的内容重新拼接成字符串
        input_tokens = tokenize(truncated_input_str, model)
        num_input_tokens = len(input_tokens)
        avg_token_length = num_input_tokens / (len(input_str_chuncks) - i)
        num_output_tokens = avg_token_length * steps * STEP_MULTIPLIER
        if num_input_tokens + num_output_tokens <= context_length:  # 检查输入和输出总令牌数是否在上下文长度限制内
            truncated_input_arr = input_arr[i:]  # 如果条件满足，从当前截断点 i 处截断原始时间序列数组。
            break
    if i > 0:  # 截断警告
        print(f'Warning: Truncated input from {len(input_arr)} to {len(truncated_input_arr)}')
    return truncated_input_arr, truncated_input_str

def tokenize(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(str)

context_lengths = {
    'text-davinci-003': 4097,
    'gpt-3.5-turbo-instruct': 4097,
    'gpt-3.5-turbo': 4097,
    'gpt-4o-mini': 128000,
    'gpt-4o': 128000,
    'llama-7b': 4096,
    'llama-13b': 4096,
    'llama-70b': 4096,
    'llama-7b-chat': 4096,
    'llama-13b-chat': 4096,
    'llama-70b-chat': 4096,
}