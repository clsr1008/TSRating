import json
from scipy.stats import pearsonr
import numpy as np


def load_scores(file_path, aspect_keys):
    """
    Load the scores for specified evaluation dimensions from a JSONL file.

    Parameters:
        file_path (str): The path to the annotation file.
        aspect_keys (list): A list of evaluation dimensions.

    Returns:
        dict: A dictionary containing the scores for each dimension, where the key is the dimension name and the value is a list of scores.
    """
    scores_dict = {aspect: [] for aspect in aspect_keys}
    with open(file_path, "r") as f:
        for line in f:
            record = json.loads(line.strip())
            for aspect in aspect_keys:
                key = f"{aspect}_score"
                if key in record:
                    scores_dict[aspect].append(record[key])
    return scores_dict


def compare_aspects(file_path, aspects):
    """
    Compare the Pearson correlation coefficients between multiple evaluation dimensions in the same file.

    Parameters:
        file_path (str): The path to the annotation file (in JSONL format).
        aspects (list): A list of evaluation dimensions.

    Returns:
        np.ndarray: The Pearson correlation coefficient matrix.
    """
    scores_dict = load_scores(file_path, aspects)

    scores_list = [scores_dict[aspect] for aspect in aspects]
    if not all(len(scores) == len(scores_list[0]) for scores in scores_list):
        raise ValueError("The number of dimension score records in the file is inconsistent.")

    num_aspects = len(aspects)
    correlation_matrix = np.zeros((num_aspects, num_aspects))
    for i in range(num_aspects):
        for j in range(num_aspects):
            if i <= j:
                correlation, _ = pearsonr(scores_list[i], scores_list[j])
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation

    return correlation_matrix


if __name__ == "__main__":
    # Example usage with replaceable parameters
    file_path = "../middleware/traffic/annotation.jsonl"  # to be changed
    aspects = ["trend", "frequency", "amplitude", "pattern"]  # to be changed

    correlation_matrix = compare_aspects(file_path, aspects)

    # print results
    column_width = 12
    header = "".ljust(column_width) + "".join(aspect.ljust(column_width) for aspect in aspects)
    print(header)

    for i, aspect in enumerate(aspects):
        row = aspect.ljust(column_width) + "".join(f"{correlation_matrix[i, j]:.2f}".ljust(column_width) for j in range(len(aspects)))
        print(row)
