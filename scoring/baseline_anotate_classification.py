import pandas as pd
import numpy as np
from sktime.datasets import load_from_tsfile_to_dataframe
from data_preparation.uea import subsample, interpolate_missing

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from opendataval.dataval import DataOob, DataShapley, KNNShapley
from opendataval.dataloader import DataFetcher
from opendataval.model import RegressionSkLearnWrapper
from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import RandomOverSampler, SMOTE

import json
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone
import time

def compute_loo_linear_approx(train_idx, val_idx, X_train, Y_train, X_val, Y_val, params):
    """
    Compute the Leave-One-Out (LOO) influence using a linear approximation.

    Parameters:
    - train_idx: int, index of the training point to remove.
    - val_idx: int, index of the validation point to evaluate.
    - X_train: np.ndarray, training features of shape (n_train, n_features).
    - Y_train: np.ndarray, training labels of shape (n_train,).
    - X_val: np.ndarray, validation features of shape (n_val, n_features).
    - Y_val: np.ndarray, validation labels of shape (n_val,).
    - params: tuple, (beta, b, inv_hess) where:
        - beta: np.ndarray, model coefficients.
        - b: float, model intercept.
        - inv_hess: np.ndarray, inverse Hessian approximation.

    Returns:
    - float, estimated LOO influence on validation loss.
    """
    beta, b, inv_hess = params

    # Training sample to be removed
    x_i = X_train[train_idx]
    y_i = Y_train[train_idx]

    # Compute the parameter update using influence functions
    # delta_beta = H^(-1) * x_i * (y_i - x_i @ beta - b)
    prediction_error = y_i - (x_i @ beta + b)
    delta_beta = inv_hess @ x_i * prediction_error

    # Validation sample to evaluate influence
    x_val = X_val[val_idx]

    # Compute influence on validation prediction
    val_influence = x_val @ delta_beta

    return val_influence

def evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """
    Evaluate and score each training block (classification task version).
    """
    fetcher = DataFetcher.from_data_splits(X_train, Y_train, X_val, Y_val, X_test, Y_test, one_hot=False)
    pred_model = RegressionSkLearnWrapper(RandomForestClassifier)
    scores = {"DataOob": [], "DataShapley": [], "KNNShapley": [], "TimeInf": []}

    # DataOob
    start_time = time.time()
    data_oob = DataOob(num_models=1000).train(fetcher=fetcher, pred_model=pred_model)
    end_time = time.time()
    scores["DataOob"] = data_oob.data_values
    print(f"DataOob method execution time: {end_time - start_time:.4f} seconds")

    # DataShapley
    start_time = time.time()
    data_shapley = DataShapley(gr_threshold=1.1, max_mc_epochs=100, min_models=100).train(
        fetcher=fetcher, pred_model=pred_model
    )
    end_time = time.time()
    scores["DataShapley"] = data_shapley.data_values
    print(f"DataShapley method execution time: {end_time - start_time:.4f} seconds")

    # KNNShapley
    start_time = time.time()
    knn_shapley = KNNShapley(k_neighbors=0.1 * len(X_train)).train(fetcher=fetcher)
    end_time = time.time()
    scores["KNNShapley"] = knn_shapley.data_values
    print(f"KNNShapley method execution time: {end_time - start_time:.4f} seconds")

    # TimeInf: Compute time-based influences using LOO approximation
    start_time = time.time()
    time_inf_scores = []
    params = (np.zeros(X_train.shape[1]), 0.0, np.linalg.pinv(X_train.T @ X_train))

    for train_idx in range(len(X_train)):
        influence_sum = 0
        for val_idx in range(len(X_val)):
            influence_sum += compute_loo_linear_approx(train_idx, val_idx, X_train, Y_train, X_val, Y_val, params)
        time_inf_scores.append(influence_sum / len(X_val))

    scores["TimeInf"] = time_inf_scores
    end_time = time.time()
    print(f"TimeInf method execution time: {end_time - start_time:.4f} seconds")

    return scores



def load_cla_data(filepath):
    df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                               replace_missing_vals_with='NaN')
    labels = pd.Series(labels, dtype="category")
    class_names = labels.cat.categories
    labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)

    lengths = df.applymap(lambda x: len(x)).values

    horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))
    if np.sum(horiz_diffs) > 0:
        df = df.applymap(subsample)

    lengths = df.applymap(lambda x: len(x)).values
    vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
    if np.sum(vert_diffs) > 0:
        max_seq_len = int(np.max(lengths[:, 0]))
    else:
        max_seq_len = lengths[0, 0]

    df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns})
                    .reset_index(drop=True)
                    .set_index(pd.Series(lengths[row, 0] * [row]))
                    for row in range(df.shape[0])), axis=0)

    grp = df.groupby(by=df.index)
    df = grp.transform(interpolate_missing)

    time_series_list = [group['dim_0'].tolist() for _, group in df.groupby(df.index)]

    time_series_arr = np.array([np.asarray(input_arr) for input_arr in time_series_list])
    labels_arr = np.array(labels_df.values.ravel())

    return time_series_arr, labels_arr


def save_scores_to_jsonl(scores, output_file):
    """ Assign scores to the index of each block and save them to a JSONL file. """
    block_scores = {method: [] for method in scores.keys()}
    for method, values in scores.items():
        block_scores[method] = values

    print("Evaluation complete, scores for each block:")
    for method, values in block_scores.items():
        print(f"{method}: {values}")
        print(len(values))

    try:
        with open(output_file, "r") as jsonl_file:
            original_data = [json.loads(line) for line in jsonl_file]

        for idx, record in enumerate(original_data):
            record.update({
                "DataOob": float(block_scores["DataOob"][idx]),
                "DataShapley": float(block_scores["DataShapley"][idx]),
                "KNNShapley": float(block_scores["KNNShapley"][idx]),
                "TimeInf": float(block_scores["TimeInf"][idx]),
            })

        with open(output_file, "w") as jsonl_file:
            jsonl_file.writelines(json.dumps(record) + "\n" for record in original_data)

        print(f"results have been saved to {output_file}")
    except Exception as e:
        print(f"error occurs while saving results: {e}")



if __name__ == "__main__":
    # Example usage with replaceable parameters
    train_filepath = "../datasets/MedicalImages/MedicalImages_TRAIN.ts"  # to be changed
    X_train, Y_train = load_cla_data(train_filepath)
    test_filepath = "../datasets/MedicalImages/MedicalImages_TEST.ts"  # to be changed
    X_test, Y_test = load_cla_data(test_filepath)
    output_file = "../middleware/MedicalImages/annotation.jsonl"  # to be changed

    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.2, random_state=42, stratify=Y_test)
    print("Train class distribution:", np.unique(Y_train, return_counts=True))
    print("Validation class distribution:", np.unique(Y_val, return_counts=True))
    print("Test class distribution:", np.unique(Y_test, return_counts=True))

    scores = evaluate_blocks(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    save_scores_to_jsonl(scores, output_file)
