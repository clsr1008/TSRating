import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from uea import subsample, interpolate_missing
from load_forecast_data import count_tokens, save_to_jsonl
from serialize import SerializerSettings, serialize_arr


def load_cla_data(filepath):
    # load data from .ts file
    df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                               replace_missing_vals_with='NaN')
    # Convert labels to categorical type
    labels = pd.Series(labels, dtype="category")
    class_names = labels.cat.categories
    labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)

    # Check the length of each time series to ensure consistent sample and dimension length
    lengths = df.applymap(lambda x: len(x)).values  # Length of each time series (num_samples, num_dimensions)

    horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))  # Check length differences across dimensions for the same sample
    if np.sum(horiz_diffs) > 0:
        df = df.applymap(subsample)

    # Recalculate lengths to ensure consistent dimension length across samples
    lengths = df.applymap(lambda x: len(x)).values
    vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))  # Check length differences between samples across dimensions
    if np.sum(vert_diffs) > 0:
        max_seq_len = int(np.max(lengths[:, 0]))
    else:
        max_seq_len = lengths[0, 0]

    # Flatten the time series data to generate data in (seq_len, feat_dim) format
    df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                    .reset_index(drop=True)
                    .set_index(pd.Series(lengths[row, 0] * [row]))
                    for row in range(df.shape[0])), axis=0)

    # Perform interpolation for missing values
    grp = df.groupby(by=df.index)
    df = grp.transform(interpolate_missing)

    return df, labels_df, max_seq_len


def process_time_series(time_series_list, model, settings):
    arr_slices = [np.asarray(input_arr) for input_arr in time_series_list]
    str_slices = [serialize_arr(input_arr, settings) for input_arr in arr_slices]
    num_tokens = count_tokens(str_slices, model)

    return arr_slices, str_slices, num_tokens


if __name__ == "__main__":
    # Example usage with replaceable parameters
    filepath = "../datasets/MedicalImages/MedicalImages_TRAIN.ts"  # to be changed
    prec = 6  # to be changed
    model = "gpt-4o-mini"  # to be changed
    jsonl_path = "../middleware/MedicalImages/blocks.jsonl"  # to be changed

    df, labels_df, max_seq_len = load_cla_data(filepath)
    time_series_list = [group['dim_0'].tolist() for _, group in df.groupby(df.index)]  # can be changed to other dimensions
    settings = SerializerSettings(prec=prec)
    arr_slices, str_slices, num_tokens = process_time_series(time_series_list, model, settings)

    print("The maximum number of tokens in the sliding window is ", max(num_tokens))
    save_to_jsonl(arr_slices, str_slices, jsonl_path)
    print("The data has been written to JSONL.")
