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
        Load data from an Excel file and extract the specified column, sampling the data within the specified range.

        Parameters:
            file_path (str): Path to the Excel file.
            column_name (str): The name of the column to extract.
            start_idx (int): The row index where the data starts.
            end_idx (int): The row index where the data ends.

        Returns:
            pd.Series: The extracted time series data.
    """
    df = pd.read_csv(file_path)
    series = df[column_name].iloc[start_idx:end_idx]
    return pd.Series(series.values, index=pd.RangeIndex(len(series)))


def preprocess_data(train, settings, model, blocklength):
    """
        Preprocess training data, including division and serialization.
    """
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)
    if not isinstance(train, pd.Series):
        train = pd.Series(train, index=pd.RangeIndex(len(train)))
    # input_arrs = train.values.reshape(-1, 1)
    # scaler = StandardScaler()
    # scaler.fit(input_arrs)
    # transformed_input_arrs = np.array(scaler.transform(input_arrs)).flatten()
    arr_slices = generate_sliding_windows(train.values, blocklength)
    str_slices = [serialize_arr(scaled_input_arr, settings) for scaled_input_arr in arr_slices]
    num_tokens = count_tokens(str_slices, model)
    return arr_slices, str_slices, num_tokens


def count_tokens(input_strs, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)
    # Encode each string in the list and return the token IDs
    return [len(encoding.encode(text)) for text in input_strs]

def generate_sliding_windows(data, block_length):
    """
        Generate sliding windows based on the given one-dimensional array and window length.
    """
    slices = []
    for i in range(len(data) - block_length + 1):
        slices.append(data[i:i + block_length])
    return slices


def get_dataset_by_name(file_path, column_name, start_idx, end_idx, prec, blocklength,
                        testfrac=0.3, model="gpt-4o-mini"):
    """
        Load and preprocess the dataset based on the given parameters, splitting it into training and testing sets.

        Parameters:
            file_path (str): Path to the Excel file containing the dataset.
            column_name (str): The name of the column to extract from the dataset.
            start_idx (int): The starting index for extracting the data.
            end_idx (int): The ending index for extracting the data.
            prec (int): Precision for the data serialization.
            blocklength (int): The length of the data blocks for preprocessing.
            testfrac (float, optional): The fraction of the data to be used for testing (default is 0.3).
            model (str, optional): The model to be used for preprocessing (default is "gpt-4o-mini").

        Returns:
            dict: A dictionary containing the following keys:
                - "train": The training data as a pandas Series.
                - "test": The testing data as a pandas Series.
                - "arr_slices": The preprocessed data in array slices.
                - "str_slices": The preprocessed data in string slices.
                - "num_tokens": The number of tokens in the preprocessed data.
    """
    series = get_dataset_from_excel(file_path, column_name, start_idx, end_idx)

    splitpoint = int(len(series) * (1 - testfrac))
    train = series.iloc[:splitpoint]
    test = series.iloc[splitpoint:]

    settings = SerializerSettings(prec=prec)
    arr_slices, str_slices, num_tokens = preprocess_data(
        train=train, settings=settings, model=model, blocklength=blocklength
    )

    return {
        "train": train,
        "test": test,
        "arr_slices": arr_slices,
        "str_slices": str_slices,
        "num_tokens": num_tokens
    }

def save_to_jsonl(input_arrs, input_strs, file_path, shuffle=False):
    """
        Convert input_arrs and input_strs to JSONL format and add an index field.

        Parameters:
            input_arrs (list): A list of input arrays converted from numpy.ndarray.
            input_strs (list): A list of input strings.
            file_path (str): The path to save the JSONL file.
            shuffle (bool, optional): Whether to shuffle the data order (default is False).
    """
    if shuffle:
        indexed_data = [(index, arr, string) for index, (arr, string) in enumerate(zip(input_arrs, input_strs))]
        random.shuffle(indexed_data)
        indices, input_arrs, input_strs = zip(*indexed_data)
    else:
        indices = list(range(len(input_arrs)))

    with open(file_path, "w") as f:
        for index, (arr, string) in zip(indices, zip(input_arrs, input_strs)):
            data = {
                "index": index,
                "input_arr": arr.tolist(),
                "input_str": string
            }
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    # Example usage with replaceable parameters
    file_path = "../datasets/traffic/traffic.csv"  # to be changed
    column_name = "OT"  # to be changed
    start_idx = 4000  # to be changed
    end_idx = 8000  # to be changed
    block_length = 128   # to be changed
    prec = 4  # to be changed
    jsonl_path = "../middleware/traffic/blocks.jsonl"  # to be changed

    data = get_dataset_by_name(file_path, column_name, start_idx, end_idx, prec, block_length)
    print(f"Train shape: {data['train'].shape}, Test shape: {data['test'].shape}")
    print(f"Num of blocks: {len(data['arr_slices'])}")
    print(f"First input: {data['arr_slices'][0]}")
    print(type(data['arr_slices'][0]))
    print(f"First serialized input: {data['str_slices'][0]}")
    print(type(data['str_slices'][0]))
    print("The maximum number of tokens in the sliding window is ", max(data['num_tokens']))

    save_to_jsonl(data['arr_slices'], data['str_slices'], jsonl_path)
    print("The data has been written to JSONL.")
