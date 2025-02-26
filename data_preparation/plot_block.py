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
            indices (list): A list of indices for the blocks to be plotted (e.g., [3, 7, 15]).
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


if __name__ == "__main__":
    # Example usage with replaceable parameters
    file_path = "../middleware/traffic/annotation.jsonl"  # to be changed
    indices = [2547, 9, 74, 1, 59]  # to be changed
    output_dir = "../middleware/traffic/plots"  # to be changed

    plot_blocks_from_jsonl(file_path, indices, output_dir)
