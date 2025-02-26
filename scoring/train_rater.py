import json
import torch
from momentfm import MOMENTPipeline
import argparse
import pandas as pd
from trainer import PairwiseDataset, ScoreModel, train_model, evaluate_model
from torch.utils.data import random_split, Subset
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_moment_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={'task_name': 'embedding'}
    )
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters in the encoder: {num_params}")
    model.init()
    model.eval()
    return model

def load_jsonl_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def filter_pairwise_data(pairwise_df):
    """
        Filter the high-confidence samples from the comparison dataset.
        Condition: comparisons_avg less than 0.25 or greater than 0.75
    """
    filtered_df = pairwise_df[
        (pairwise_df["comparisons_avg"] >= 0) &
        (pairwise_df["comparisons_avg"] <= 1) &
        ((pairwise_df["comparisons_avg"] <= 0.25) | (pairwise_df["comparisons_avg"] >= 0.75))
    ]

    ratio = len(filtered_df) / len(pairwise_df) * 100

    print(f"Filtered dataset: {len(filtered_df)} samples remaining ({ratio:.2f}% of original dataset).")
    # print(f"Filtered sample indices: {filtered_indices.tolist()}")

    return filtered_df


def process_time_series(model, data):
    embeddings_dict = {}

    input_tensors = []
    indices = []

    for record in data:
        input_arr = record.get("input_arr")
        index = record.get("index")
        if input_arr is not None and index is not None:
            input_tensors.append(input_arr)
            indices.append(index)

    input_tensor = torch.tensor(input_tensors, dtype=torch.float32).unsqueeze(1)
    print(f"Input tensor shape: {input_tensor.shape}")

    output = model(x_enc=input_tensor)

    embeddings = output.embeddings
    print(f"Embeddings shape: {embeddings.shape}")  # [batch_size, n_channels, context_length]

    for idx, embedding in zip(indices, embeddings):
        embeddings_dict[idx] = embedding

    return embeddings_dict


def create_subset_dataset(full_dataset, retain_ratio=0.5):
    if not (0 < retain_ratio <= 1):
        raise ValueError("retain_ratio must be in the range (0, 1]")
    total_size = len(full_dataset)
    retain_size = int(total_size * retain_ratio)
    indices = torch.randperm(total_size).tolist()[:retain_size]
    return Subset(full_dataset, indices)

def main(jsonl_path, pairwise_path, output_model_path):
    # Step 0: Set random seed
    set_seed(42)

    # Step 1: Load the MOMENT model
    model = load_moment_model()

    # Step 2: Load the JSONL data
    data = load_jsonl_data(jsonl_path)
    print(f"Loaded {len(data)} records from {jsonl_path}")

    # Step 3: Generate features using the MOMENT model
    embeddings_dict = process_time_series(model, data)
    print(f"Generated embeddings for {len(embeddings_dict)} records")

    # Step 4: Load the pairwise dataset
    pairwise_df = pd.read_excel(pairwise_path)
    print(f"Loaded pairwise dataset with {len(pairwise_df)} pairs.")

    # Step 5: Filter samples with high confidence
    pairwise_df = filter_pairwise_data(pairwise_df)

    # Step 6: Build the dataset
    dataset = PairwiseDataset(embeddings_dict, pairwise_df)

    # Split into 80% training set and 20% testing set
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Step 7: Initialize the scoring model
    input_dim = next(iter(embeddings_dict.values())).shape[0]
    score_model = ScoreModel(input_dim)

    # Step 8: Train the model (you can apply grid search for hyperparameter tuning)
    score_model = train_model(score_model, train_dataset, epochs=20, batch_size=64, lr=0.005)

    # Step 9: Test the model
    evaluate_model(score_model, test_dataset)

    # Step 10: Save the model parameters
    torch.save(score_model.state_dict(), output_model_path)
    print(f"Model parameters saved to {output_model_path}")

if __name__ == "__main__":
    # Example usage with replaceable parameters
    parser = argparse.ArgumentParser(description="Time series model scoring program")
    parser.add_argument("--jsonl_path", type=str, help="Input JSONL file path")
    parser.add_argument("--pairwise_path", type=str, help="Input pairwise dataset path")
    parser.add_argument("--output_model_path", type=str, help="Output model path")

    args = parser.parse_args()

    args.jsonl_path = "../middleware/traffic/blocks.jsonl"  # to be changed
    args.pairwise_path = "../middleware/traffic/pairwise_trend.xlsx"  # to be changed
    args.output_model_path = "../middleware/traffic/rater_trend.pth"  # to be changed

    main(
        jsonl_path=args.jsonl_path,
        pairwise_path=args.pairwise_path,
        output_model_path=args.output_model_path
    )
