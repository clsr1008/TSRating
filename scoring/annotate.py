import json
import torch
from train_rater import load_moment_model, load_jsonl_data, process_time_series
from trainer import ScoreModel
import time


def annotate_data(score_model_paths, input_jsonl_path, output_jsonl_path, input_dim=768):
    """
    使用多个评分模型对 JSONL 数据进行评分，并将结果保存到新的 JSONL 文件中。

    参数:
        score_model_paths (dict): 保存的评分模型路径字典，包含四个模型的路径。
        input_jsonl_path (str): 输入 JSONL 数据路径。
        output_jsonl_path (str): 输出 JSONL 数据路径。
        input_dim (int): 评分模型的输入维度。
    """
    # Step 1: 加载所有评分模型
    models = {}
    for aspect, model_path in score_model_paths.items():
        model = ScoreModel(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models[aspect] = model
    print(f"Loaded models for aspects: {list(models.keys())}")

    # Step 2: 加载 JSONL 数据
    data = load_jsonl_data(input_jsonl_path)
    print(f"Loaded {len(data)} records.")

    # Step 3: 加载 MOMENT 模型并生成特征
    moment_model = load_moment_model()
    embeddings_dict = process_time_series(moment_model, data)

    start_time = time.time()  # 记录开始时间

    # Step 4: 对每个数据块使用所有模型打分
    results = []
    missing_embeddings = 0
    for record in data:
        record_id = record.get("index")
        if record_id is not None and record_id in embeddings_dict:
            embedding = embeddings_dict[record_id]
            embedding_tensor = embedding.clone().detach().unsqueeze(0)  # 增加 batch 维度

            # 使用每个模型对该数据块打分
            for aspect, model in models.items():
                score = model(embedding_tensor).item()
                record[f"{aspect}_score"] = score

            results.append(record)
        else:
            missing_embeddings += 1
            print(f"Warning: No embedding found for record ID {record_id}. Skipping.")

    print(f"Scoring completed. Missing embeddings for {missing_embeddings} records.")

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算总耗时
    print(f"Total annotation time: {elapsed_time:.2f} seconds")  # 打印总耗时

    # Step 5: 保存打分结果
    with open(output_jsonl_path, "w") as outfile:
        for result in results:
            outfile.write(f"{json.dumps(result)}\n")
    print(f"Annotation completed. Results saved to {output_jsonl_path}.")


if __name__ == "__main__":
    # 配置模型路径
    score_model_paths = {
        "trend": "../middleware/traffic/rater_trend.pth",
        "frequency": "../middleware/traffic/rater_frequency.pth",
        "amplitude": "../middleware/traffic/rater_amplitude.pth",
        "pattern": "../middleware/traffic/rater_pattern.pth",
    }
    input_jsonl_path = "../middleware/traffic/blocks.jsonl"
    output_jsonl_path = "../middleware/traffic/annotation.jsonl"

    # 执行注释流程
    annotate_data(score_model_paths, input_jsonl_path, output_jsonl_path)
