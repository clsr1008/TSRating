import subprocess
from datasets import Dataset
import json

def run_score_pairwise(input_path, output_path):
    # Construct the command
    command = [
        "python", "score_pairwise.py", input_path, output_path,
        # "-n", "10", # Number of samples to select for comparison from the dataset
        "--num_examples_proportion", "1.0",
        "-k", "2",  # 2 indicates pairwise comparison
        "--model", "gpt-4o-mini",
        "--generations", "20",
        "--template_file", "templates/pairwise_trend.txt", # to be changed
        "--tokens_min", "512", # Each series must contain at least 512 tokens
        "--tokens_max", "1024", # Each series can contain at most 1024 tokens
        "--ratio", "1.0", # The valid pairwise count is how many times the number of blocks
        "--text_field", "input_str"  # The field name containing the target data in each data example
    ]

    if json_flag:  # Reading a dataset rather than a single file
        command.append("--json")

    if out_format:  # Set to make the output concise
        command.append("--flat_output_format")

    # 执行命令
    subprocess.run(command)

def convert_to_excel(dataset_path, excel_path, jsonl_path):
    """
        Add 'block_a' and 'block_b' columns to the Excel spreadsheet based on the index data in the JSONL file.

        Parameters:
            dataset_path (str): Path to the original dataset.
            excel_path (str): Path to save the generated Excel file.
            jsonl_path (str): Path to the JSONL file providing the index data.
    """
    dataset = Dataset.load_from_disk(dataset_path)

    df = dataset.to_pandas()
    with open(jsonl_path, "r") as f:
        indices = []
        for line in f:
            data = json.loads(line)
            indices.append(data['index'])
    df.to_excel(excel_path, index=False)
    print(f"Data has been saved in Excel file: {excel_path}")

# 示例用法
if __name__ == "__main__":
    # Example usage with replaceable parameters
    input_path = "../middleware/traffic/blocks.jsonl"  # to be changed
    output_path = "../middleware/traffic/pairwise_trend"  # to be changed

    json_flag = True
    out_format = True
    run_score_pairwise(input_path, output_path)
    # convert to Excel file
    excel_output_path = "../middleware/traffic/pairwise_trend.xlsx"  # to be changed
    convert_to_excel(output_path, excel_output_path, input_path)
