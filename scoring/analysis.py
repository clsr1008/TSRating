import json
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_blocks_from_jsonl(file_path, indices, output_dir):
    """
        Plot the corresponding data blocks from the JSONL file based on the index list and save them to the specified directory.

        Parameters:
            file_path (str): The path to the JSONL file.
            indices (list): A list of indices of the blocks to be plotted (e.g., [3, 7, 15]).
            output_dir (str): The directory path where the images will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    indices_set = set(indices)
    found_indices = set()

    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            block_index = data['index']

            if block_index in indices_set:
                input_arr = np.array(data['input_arr'])

                if input_arr.ndim == 1:
                    plt.figure()
                    plt.plot(input_arr)
                    plt.title(f"Block at index {block_index}")
                    plt.xlabel("Step")
                    plt.ylabel("Value")

                    output_path = os.path.join(output_dir, f"block_{block_index}.png")
                    plt.savefig(output_path)
                    plt.close()

                    print(f"Saved plot for block {block_index} at {output_path}")
                    found_indices.add(block_index)

                else:
                    print(f"Data at index {block_index} is not a 1D array for plotting.")

                if found_indices == indices_set:
                    break

    missing_indices = indices_set - found_indices
    if missing_indices:
        print(f"Warning: The following indices were not found in the file: {sorted(missing_indices)}")


def analyze_scores(input_jsonl_path, output_base_dir):
    """
        Read the JSONL file, retrieve the top 20 highest and bottom 20 lowest indices, and plot the corresponding graphs.

        Args:
            input_jsonl_path (str): The path to the input JSONL file.
            output_base_dir (str): The root directory for saving the output images.
    """
    print(f"Loading data from {input_jsonl_path}...")
    records = []

    with open(input_jsonl_path, "r") as infile:
        for line in infile:
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                print("Skipping invalid JSON line:", line.strip())

    print(f"Loaded {len(records)} records.")

    dimensions = ["trend_score", "frequency_score", "amplitude_score", "pattern_score"]

    for dimension in dimensions:
        print(f"\nAnalyzing {dimension}...")

        valid_records = [r for r in records if r.get(dimension) is not None]
        sorted_records = sorted(valid_records, key=lambda x: x[dimension], reverse=True)

        top_20 = [r["index"] for r in sorted_records[:20]]
        bottom_20 = [r["index"] for r in sorted_records[-20:]]

        print(f"  Top 20 {dimension} indices: {top_20}")
        print(f"  Bottom 20 {dimension} indices: {bottom_20}")

        top_20_dir = os.path.join(output_base_dir, dimension, "top_20")
        bottom_20_dir = os.path.join(output_base_dir, dimension, "bottom_20")
        os.makedirs(top_20_dir, exist_ok=True)
        os.makedirs(bottom_20_dir, exist_ok=True)

        # 画图并保存
        plot_blocks_from_jsonl(input_jsonl_path, top_20, top_20_dir)
        plot_blocks_from_jsonl(input_jsonl_path, bottom_20, bottom_20_dir)


if __name__ == "__main__":
    # Example usage with replaceable parameters
    input_jsonl_path = "../middleware/traffic/annotation.jsonl"  # to be changed
    output_base_dir = "../middleware/traffic/plots"  # to be changed

    analyze_scores(input_jsonl_path, output_base_dir)
