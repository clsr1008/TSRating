import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import copy
import os
import sys

from trainer import PairwiseDataset, evaluate_model, BradleyTerryLoss, ScoreModel

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(BASE_DIR)
# from scoring.trainer import ScoreModel

from meta_main import load_jsonl_data, process_time_series, filter_pairwise_data, set_seed, load_moment_model
import pandas as pd


def few_shot_finetune(model, dataset, few_shot_len=10, adaptation_steps=10, adaptation_lr=0.01, device='cuda'):
    model = copy.deepcopy(model).to(device)
    loss_fn = BradleyTerryLoss()
    support_len = min(few_shot_len, len(dataset) - 1)
    test_len = len(dataset) - support_len
    support_set, test_set = random_split(dataset, [support_len, test_len])

    support_loader = DataLoader(support_set, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=adaptation_lr)
    model.train()

    for _ in range(adaptation_steps):
        for emb_a, emb_b, p in support_loader:
            emb_a, emb_b, p = emb_a.to(device), emb_b.to(device), p.to(device)
            scores_a = model(emb_a)
            scores_b = model(emb_b)

            # use BradleyTerryLoss
            loss = loss_fn(scores_a, scores_b, p)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, test_set

# Load the saved scoring model
def load_model(model_path, input_dim, hidden_dim=256, num_layers=3):
    model = ScoreModel(input_dim, hidden_dim, num_layers)
    # model = ScoreModel(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

_dataset_cache = None  # Global cache variable
def main(jsonl_path, pairwise_path, model_path, adaptation_steps, adaptation_lr, few_shot_len, seed):
    global _dataset_cache
    set_seed(seed)

    # Step 1: Build dataset only on the first call
    if _dataset_cache is None:
        moment = load_moment_model()

        data = load_jsonl_data(jsonl_path)
        emb_dict = process_time_series(moment, data)

        df = pd.read_excel(pairwise_path)
        df = filter_pairwise_data(df)

        _dataset_cache = PairwiseDataset(emb_dict, df)
        print("Dataset cashed")
    else:
        print("using cashed dataset")
    input_dim = next(iter(_dataset_cache))[0].shape[0]

    # === Step 2: Load the scoring model ===
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, input_dim).to(device)

    # === Step 3: Few-shot finetuning ===
    adapted_model, test_set = few_shot_finetune(
        model, _dataset_cache,
        few_shot_len=few_shot_len,
        adaptation_steps=adaptation_steps,
        adaptation_lr=adaptation_lr,
        device=device
    )

    # === Step 4: Evaluation on the test set ===
    test_accuracy = evaluate_model(adapted_model, test_set, batch_size=16, finetune=False)
    print(f"[✓] Test accuracy after few-shot finetuning: {test_accuracy:.4f}")

    return test_accuracy, adapted_model


if __name__ == "__main__":
    # Example usage with replaceable parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", type=str, help="Path to the input JSONL file")
    parser.add_argument("--pairwise_path", type=str, help="Path to the pairwise XLSX file")
    parser.add_argument("--model_path", type=str, help="Path to the .pth model file")
    parser.add_argument("--adaptation_steps", type=int, default=10)
    parser.add_argument("--adaptation_lr", type=float, default=0.0001)
    parser.add_argument("--few_shot_len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.jsonl_path = "./middleware/traffic/blocks.jsonl"  # to be changed
    args.pairwise_path = "./middleware/traffic/pairwise_trend.xlsx"  # to be changed
    args.model_path = "./middleware/meta/rater_trend.pth"  # to be changed

    main(
        jsonl_path=args.jsonl_path,
        pairwise_path=args.pairwise_path,
        model_path=args.model_path,
        adaptation_steps=args.adaptation_steps,
        adaptation_lr=args.adaptation_lr,
    )
