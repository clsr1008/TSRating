import json
import torch
from momentfm import MOMENTPipeline
import argparse
import pandas as pd
from trainer import PairwiseDataset, ScoreModel, train_model, evaluate_model
from torch.utils.data import random_split, Subset
import random
import numpy as np
from maml import MetaLearner
from torch.utils.data import random_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load MOMENT model
def load_moment_model():
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-base",
        model_kwargs={'task_name': 'embedding'}
    )
    num_params = sum(p.numel() for p in model.encoder.parameters())
    print(f"Number of parameters in the encoder: {num_params}")
    model.init()  # Initialize model
    model.eval()  # Switch to inference mode
    return model

# Load data from a JSONL file
def load_jsonl_data(filepath):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# Filter high-confidence pairwise samples
def filter_pairwise_data(pairwise_df):
    """
    Filter high-confidence samples in the pairwise comparison dataset.
    Condition: comparisons_avg is less than 0.25 or greater than 0.75
    """
    filtered_df = pairwise_df[
        (pairwise_df["comparisons_avg"] >= 0) &
        (pairwise_df["comparisons_avg"] <= 1) &
        ((pairwise_df["comparisons_avg"] <= 0.25) | (pairwise_df["comparisons_avg"] >= 0.75))
    ]

    ratio = len(filtered_df) / len(pairwise_df) * 100
    print(f"Filtered dataset: {len(filtered_df)} samples remaining ({ratio:.2f}% of original dataset).")

    return filtered_df

# Process time series data and extract embeddings
def process_time_series(model, data):
    embeddings_dict = {}  # Stores index to embedding mappings

    input_tensors = []
    indices = []
    expected_len = 128  # Target sequence length

    # Extract input_arr and index from each record
    for record in data:
        input_arr = record.get("input_arr")
        index = record.get("index")
        if input_arr is not None and index is not None:
            # Pad or truncate to expected_len
            if len(input_arr) < expected_len:
                input_arr = input_arr + [0.0] * (expected_len - len(input_arr))
            elif len(input_arr) > expected_len:
                input_arr = input_arr[:expected_len]

            input_tensors.append(input_arr)
            indices.append(index)

    # Construct input tensor: [batch_size, 1, context_length]
    input_tensor = torch.tensor(input_tensors, dtype=torch.float32).unsqueeze(1)
    print(f"Input tensor shape: {input_tensor.shape}")  # 应为 [batch_size, 1, context_length]

    # Extract embeddings using MOMENT
    output = model(x_enc=input_tensor)

    embeddings = output.embeddings  # Get embeddings
    print(f"Embeddings shape: {embeddings.shape}")  # Should be [batch_size, feature_length]

    # Map each index to its corresponding embedding
    for idx, embedding in zip(indices, embeddings):
        embeddings_dict[idx] = embedding  # Store as PyTorch tensors

    return embeddings_dict



cached_datasets = {}
def load_or_cache_datasets(jsonl_paths, pairwise_paths, model, query_ratio, test_ratio):
    # Define a cache key using paths and model identity
    cache_key = (tuple(jsonl_paths), tuple(pairwise_paths), str(model))
    # Return cached datasets if available
    if cache_key in cached_datasets:
        print("Using cached datasets")
        return cached_datasets[cache_key]
    # Otherwise, load and cache new datasets
    else:
        task_datasets = build_task_datasets_moment(jsonl_paths, pairwise_paths, model, query_ratio, test_ratio)
        cached_datasets[cache_key] = task_datasets
        # Optionally save to disk
        # with open("task_datasets_cache.pkl", "wb") as f:
        #     pickle.dump(cached_datasets, f)
        print("Datasets loaded and cached")
        return task_datasets


def build_task_datasets_moment(jsonl_paths, pairwise_paths, model, query_ratio, test_ratio):
    task_datasets = []

    for jsonl_path, pairwise_path in zip(jsonl_paths, pairwise_paths):
        print(f"\n[Task] Loading from: {jsonl_path} + {pairwise_path}")

        # Step 1: Load JSONL data
        data = load_jsonl_data(jsonl_path)
        print(f"Loaded {len(data)} records.")

        # Step 2: Generate embeddings
        embeddings_dict = process_time_series(model, data)
        print(f"Generated embeddings for {len(embeddings_dict)} records")

        # Step 3: Load and filter pairwise comparison data
        pairwise_df = pd.read_excel(pairwise_path)
        pairwise_df = filter_pairwise_data(pairwise_df)
        print(f"Using {len(pairwise_df)} filtered pairs.")

        # Step 4: Construct the PairwiseDataset object
        dataset = PairwiseDataset(embeddings_dict, pairwise_df)

        # Step 5: Split into support/query/test subsets
        total_len = len(dataset)
        query_size = int(total_len * query_ratio)
        test_size = int(total_len * test_ratio)
        support_size = total_len - query_size - test_size
        support_set, query_set, test_set = random_split(dataset, [support_size, query_size, test_size])

        task_datasets.append({
            "support": support_set,
            "query": query_set,
            "test": test_set
        })

    return task_datasets


def main(jsonl_paths, pairwise_paths, output_model_path, query_ratio, test_ratio, meta_lr, inner_lr, inner_steps,
         meta_batch_size, data_batch_size, epochs, hidden_dim, num_layers):
    # Step 0: Set random seed
    set_seed(42)

    # Step 1: Load the MOMENT model
    model = load_moment_model()

    # Step 2: Build task datasets
    # task_datasets = build_task_datasets_moment(jsonl_paths, pairwise_paths, model, query_ratio=query_ratio, test_ratio=test_ratio)
    task_datasets = load_or_cache_datasets(jsonl_paths, pairwise_paths, model, query_ratio, test_ratio)

    # Step 3: Initialize MetaLearner
    input_dim = next(iter(task_datasets[0]["support"]))[0].shape[0]
    meta_learner = MetaLearner(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        meta_lr=meta_lr,
        inner_lr=inner_lr,
        inner_steps=inner_steps,
        device='cuda',
    )
    # Step 4: Train the meta-learning model
    meta_learner.meta_train(task_datasets, meta_batch_size=meta_batch_size, data_batch_size=data_batch_size, epochs=epochs)

    # Step 5: Save the trained model
    torch.save(meta_learner.meta_model.state_dict(), output_model_path)
    print(f"Meta model saved to {output_model_path}")

    # Step 6: Evaluate on the test sets
    print("Evaluating on test sets of each task...")
    all_accuracies = []
    for task_id, task in enumerate(task_datasets):
        print(f"Task {task_id}:")
        acc = evaluate_model(meta_learner.meta_model, task["test"], batch_size=data_batch_size)
        all_accuracies.append(acc)
    # Compute average accuracy
    avg_accuracy = sum(all_accuracies) / len(all_accuracies)

    print(f"\nAverage Accuracy across tasks: {avg_accuracy:.4f}")
    return avg_accuracy, meta_learner.meta_model



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time series model scoring script")
    parser.add_argument("--jsonl_paths", type=str, help="Path to input JSONL files")
    parser.add_argument("--pairwise_paths", type=str, help="Path to input pairwise dataset files")
    parser.add_argument("--output_model_path", type=str, help="Path to save the output model")

    parser.add_argument("--query_ratio", type=float, default=0.4)
    parser.add_argument("--test_ratio", type=float, default=0.2)

    parser.add_argument("--meta_lr", type=float, default=0.00025)
    parser.add_argument("--inner_lr", type=float, default=0.004)
    parser.add_argument("--inner_steps", type=int, default=14)

    parser.add_argument("--meta_batch_size", type=int, default=10)
    parser.add_argument("--data_batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=7)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)

    args = parser.parse_args()

    args.jsonl_paths = [  # to be changed
        "./middleware/meta/electricity_15min/blocks.jsonl",
        "./middleware/meta/electricity_weekly/blocks.jsonl",
        "./middleware/meta/monash_traffic/blocks.jsonl",
        # add more datasets
    ]
    args.pairwise_paths = [  # to be changed
        "./middleware/meta/electricity_15min/pairwise_trend.xlsx",
        "./middleware/meta/electricity_weekly/pairwise_trend.xlsx",
        "./middleware/meta/monash_traffic/pairwise_trend.xlsx",
        # add more datasets
    ]
    args.output_model_path = "./middleware/meta/rater_trend.pth"  # to be changed

    main(
        jsonl_paths=args.jsonl_paths,
        pairwise_paths=args.pairwise_paths,
        output_model_path=args.output_model_path,
        query_ratio=args.query_ratio,
        test_ratio=args.test_ratio,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_batch_size=args.meta_batch_size,
        data_batch_size = args.data_batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
