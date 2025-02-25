import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from uea import subsample, interpolate_missing
from load_forecast_data import count_tokens, save_to_jsonl
from serialize import SerializerSettings, serialize_arr



def load_cla_data(filepath):
    # 从 .ts 文件加载数据
    df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                               replace_missing_vals_with='NaN')
    # 将标签转换为分类类型
    labels = pd.Series(labels, dtype="category")
    class_names = labels.cat.categories
    labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # 标签编码为 int8

    # 检查每个时间序列的长度，确保每个样本和维度长度一致
    lengths = df.applymap(lambda x: len(x)).values  # 每个时间序列的长度 (num_samples, num_dimensions)

    horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))  # 检查同一个样本不同维度间的长度差异
    if np.sum(horiz_diffs) > 0:  # 若发现不同维度间长度不一致
        df = df.applymap(subsample)  # 对不一致的序列进行下采样

    # 再次计算长度，确保样本间维度长度一致
    lengths = df.applymap(lambda x: len(x)).values
    vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))  # 检查不同样本间维度的长度差异
    if np.sum(vert_diffs) > 0:  # 若发现样本间长度不一致
        max_seq_len = int(np.max(lengths[:, 0]))
    else:
        max_seq_len = lengths[0, 0]

    # 将时间序列数据展开，生成 (seq_len, feat_dim) 格式的数据
    df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                    .reset_index(drop=True)
                    .set_index(pd.Series(lengths[row, 0] * [row]))
                    for row in range(df.shape[0])), axis=0)

    # 对缺失值进行插值处理
    grp = df.groupby(by=df.index)
    df = grp.transform(interpolate_missing)

    return df, labels_df, max_seq_len


def process_time_series(time_series_list, model, settings):
    # 1. 将 time_series_list 拆分为 arr_slices（假设 arr_slices 是时间序列切片）
    arr_slices = [np.asarray(input_arr) for input_arr in time_series_list]  # 如果需要进行预处理，可以在这里添加

    # 2. 对每个 arr_slices 中的项进行序列化
    str_slices = [serialize_arr(input_arr, settings) for input_arr in arr_slices]

    # 3. 计算 token 数量
    num_tokens = count_tokens(str_slices, model)

    return arr_slices, str_slices, num_tokens


if __name__ == "__main__":
    filepath = "../datasets/MedicalImages/MedicalImages_TRAIN.ts"
    df, labels_df, max_seq_len = load_cla_data(filepath)
    time_series_list = [group['dim_0'].tolist() for _, group in df.groupby(df.index)]
    print(max_seq_len)
    print(len(time_series_list))
    prec = 6
    model = "gpt-4o-mini"
    settings = SerializerSettings(prec=prec)
    arr_slices, str_slices, num_tokens = process_time_series(time_series_list, model, settings)

    print("滑动窗口最大的token数是", max(num_tokens))
    save_to_jsonl(arr_slices, str_slices, "../middleware/MedicalImages/blocks.jsonl")
    print("数据已经写入jsonl")
