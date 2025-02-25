import numpy as np
from serialize import SerializerSettings, serialize_arr
from load_forecast_data import save_to_jsonl


def generate_time_series_trend(num_significant=10, num_non_significant=10, length=25):
    """
    生成时间序列数据，其中部分具有显著趋势，部分无显著趋势，结果交替排列。

    参数:
        num_significant (int): 具有显著趋势的时间序列数量。
        num_non_significant (int): 无显著趋势的时间序列数量。
        length (int): 每条时间序列的长度。

    返回:
        list: 包含生成的时间序列的 numpy 数组列表，显著和不显著趋势交替排列。
        list: 对应的标签列表（"significant" 或 "non-significant"）。
    """

    def generate_significant_trend(length, start=0, slope=1, noise_level=1):
        x = np.arange(length)
        trend = start + slope * x
        noise = np.random.normal(0, noise_level, size=length)
        return trend + noise

    def generate_non_significant_trend(length, baseline=10, noise_level=1):
        noise = np.random.normal(0, noise_level, size=length)
        return baseline + noise

    # 生成显著趋势和非显著趋势的时间序列
    significant_trends = [
        generate_significant_trend(length, start=np.random.randint(0, 5), slope=np.random.uniform(0.2, 0.4)) for _ in
        range(num_significant)]
    non_significant_trends = [generate_non_significant_trend(length, baseline=np.random.randint(5, 15)) for _ in
                              range(num_non_significant)]

    # 交替排列
    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_trends[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_trends[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_frequency(num_significant=10, num_non_significant=10, length=25):
    """
    生成时间序列数据，其中部分具有显著的频率/周期性，部分无显著频率，结果交替排列。

    参数:
        num_significant (int): 具有显著频率的时间序列数量。
        num_non_significant (int): 无显著频率的时间序列数量。
        length (int): 每条时间序列的长度。

    返回:
        list: 包含生成的时间序列的 numpy 数组列表，显著和非显著频率交替排列。
        list: 对应的标签列表（"significant" 或 "non-significant"）。
    """

    def generate_significant_frequency(length, amplitude=3, frequency=0.2, noise_level=0.01):
        x = np.arange(length)
        signal = amplitude * np.sin(2 * np.pi * frequency * x)
        noise = np.random.normal(0, noise_level, size=length)
        return signal + noise

    def generate_non_significant_frequency(length, baseline=10, noise_level=3):
        noise = np.random.normal(0, noise_level, size=length)
        return baseline + noise

    # 生成显著频率和非显著频率的时间序列
    significant_frequencies = [
        generate_significant_frequency(length, amplitude=np.random.uniform(3, 4),
                                       frequency=np.random.uniform(0.1, 0.2))
        for _ in range(num_significant)]
    non_significant_frequencies = [generate_non_significant_frequency(length, baseline=0)
                                   for _ in range(num_non_significant)]

    # 交替排列
    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_frequencies[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_frequencies[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_amplitude(num_significant=10, num_non_significant=10, length=25):
    """
    生成时间序列数据，其中部分具有显著的振幅变化，部分无显著振幅变化，结果交替排列。

    参数:
        num_significant (int): 具有显著振幅的时间序列数量。
        num_non_significant (int): 无显著振幅的时间序列数量。
        length (int): 每条时间序列的长度。

    返回:
        list: 包含生成的时间序列的 numpy 数组列表，显著和非显著振幅交替排列。
        list: 对应的标签列表（"significant" 或 "non-significant"）。
    """

    def generate_significant_amplitude(length, base_frequency=0.2, amplitude=2, noise_level=0.1):
        x = np.arange(length)
        signal = amplitude * np.sin(2 * np.pi * base_frequency * x)
        noise = np.random.normal(0, noise_level, size=length)
        return signal + noise

    def generate_non_significant_amplitude(length, baseline=0, noise_level=1):
        return np.random.normal(baseline, noise_level, size=length)

    # 生成显著振幅和非显著振幅的时间序列
    significant_amplitudes = [
        generate_significant_amplitude(length, base_frequency=np.random.uniform(0.1, 0.2),
                                       amplitude=np.random.uniform(4, 5))
        for _ in range(num_significant)]
    non_significant_amplitudes = [generate_non_significant_amplitude(length) for _ in range(num_non_significant)]
    # 交替排列
    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_amplitudes[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_amplitudes[i])
            labels.append("non-significant")

    return time_series_data, labels


def generate_time_series_pattern(num_significant=10, num_non_significant=10, length=50):
    """
    生成时间序列数据，其中部分具有显著的模式/形状，部分无显著模式，结果交替排列。

    参数:
        num_significant (int): 具有显著模式的时间序列数量。
        num_non_significant (int): 无显著模式的时间序列数量。
        length (int): 每条时间序列的长度。

    返回:
        list: 包含生成的时间序列的 numpy 数组列表，显著和非显著模式交替排列。
        list: 对应的标签列表（"significant" 或 "non-significant"）。
    """

    def generate_significant_pattern(length, beta_0 = 10, beta_1 = 0.05, A_1 = 5, f_1 = 1 / 12, phi_1 = np.pi / 2, A_2 = 3, f_2 = 1 / 20, phi_2 = np.pi / 3, noise_level=0):
        x = np.arange(length)
        trend = beta_0 + beta_1 * x
        seasonality = A_1 * np.sin(2 * np.pi * f_1 * x + phi_1)
        cyclical = A_2 * np.cos(2 * np.pi * f_2 * x + phi_2)
        noise = np.random.normal(0, noise_level, size=length)
        return trend + seasonality + cyclical + noise

    def generate_non_significant_pattern(length, noise_level=6):
        return np.random.normal(10, noise_level, size=length)

    # 生成显著模式和非显著模式的时间序列
    significant_patterns = [
        generate_significant_pattern(length, beta_0 = np.random.randint(5, 15), beta_1 = np.random.uniform(0.01, 0.1), A_1 = np.random.uniform(5, 8), f_1 = np.random.uniform(0.08, 0.1), phi_1 = np.random.uniform(0, np.pi), A_2 = np.random.uniform(1, 2), f_2 = np.random.uniform(0.02, 0.04), phi_2 = np.random.uniform(0, np.pi))
        for _ in range(num_significant)]
    non_significant_patterns = [generate_non_significant_pattern(length) for _ in range(num_non_significant)]
    # 交替排列
    time_series_data = []
    labels = []
    for i in range(max(num_significant, num_non_significant)):
        if i < num_significant:
            time_series_data.append(significant_patterns[i])
            labels.append("significant")
        if i < num_non_significant:
            time_series_data.append(non_significant_patterns[i])
            labels.append("non-significant")

    return time_series_data, labels


if __name__ == "__main__":
    # 测试函数
    time_series_list, labels_list = generate_time_series_pattern(num_significant=200, num_non_significant=200, length=50)
    print(time_series_list[:4])
    print(labels_list[:4])  # 示例展示前两条数据及其标签
    str_slices = [serialize_arr(input_arr, SerializerSettings()) for input_arr in time_series_list]
    save_to_jsonl(time_series_list, str_slices, "synthesis/tsdata_synthesis.jsonl", False)
    print("数据已经写入tsdata_synthesis.jsonl")
