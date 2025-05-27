import json
import torch
from train_rater import load_moment_model, load_jsonl_data, process_time_series
# from trainer import ScoreModel

import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
from meta_rater.trainer import ScoreModel
import time


def annotate_data(score_model_paths, input_jsonl_path, output_jsonl_path, input_dim=768, hidden_dim=256, num_layers=3):
    """
        Use multiple scoring models to score the JSONL data and save the results to a new JSONL file.

        Parameters:
            score_model_paths (dict): A dictionary of saved scoring model paths, containing paths for four models.
            input_jsonl_path (str): The path to the input JSONL data.
            output_jsonl_path (str): The path to the output JSONL data.
            input_dim (int): The input dimension for the scoring models.
    """
    models = {}
    for aspect, model_path in score_model_paths.items():
        # model = ScoreModel(input_dim=input_dim)
        model = ScoreModel(input_dim, hidden_dim, num_layers)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[aspect] = model
    print(f"Loaded models for aspects: {list(models.keys())}")

    data = load_jsonl_data(input_jsonl_path)
    print(f"Loaded {len(data)} records.")

    moment_model = load_moment_model()
    embeddings_dict = process_time_series(moment_model, data)

    start_time = time.time()

    results = []
    missing_embeddings = 0
    for record in data:
        record_id = record.get("index")
        if record_id is not None and record_id in embeddings_dict:
            embedding = embeddings_dict[record_id]
            embedding_tensor = embedding.clone().detach().unsqueeze(0)

            for aspect, model in models.items():
                score = model(embedding_tensor).item()
                record[f"{aspect}_score"] = score

            results.append(record)
        else:
            missing_embeddings += 1
            print(f"Warning: No embedding found for record ID {record_id}. Skipping.")

    print(f"Scoring completed. Missing embeddings for {missing_embeddings} records.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total annotation time: {elapsed_time:.2f} seconds")

    with open(output_jsonl_path, "w") as outfile:
        for result in results:
            outfile.write(f"{json.dumps(result)}\n")
    print(f"Annotation completed. Results saved to {output_jsonl_path}.")


if __name__ == "__main__":
    # Example usage with replaceable parameters
    score_model_paths = {  # to be changed
        "trend": "../middleware/traffic/rater_trend.pth",
        "frequency": "../middleware/traffic/rater_frequency.pth",
        "amplitude": "../middleware/traffic/rater_amplitude.pth",
        "pattern": "../middleware/traffic/rater_pattern.pth",
    }
    input_jsonl_path = "../middleware/traffic/blocks.jsonl"  # to be changed
    output_jsonl_path = "../middleware/traffic/annotation.jsonl"  # to be changed

    # 执行注释流程
    annotate_data(score_model_paths, input_jsonl_path, output_jsonl_path)
