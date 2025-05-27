import random
import numpy as np
import json
import tiktoken
from time_moe.datasets.time_moe_dataset import TimeMoEDataset
from serialize import SerializerSettings, serialize_arr


def save_to_jsonl(input_arrs, input_strs, file_path, shuffle=False):
    if shuffle:
        indexed_data = [(index, arr, string) for index, (arr, string) in enumerate(zip(input_arrs, input_strs))]
        random.shuffle(indexed_data)
        indices, input_arrs, input_strs = zip(*indexed_data)
    else:
        indices = list(range(len(input_arrs)))

    with open(file_path, "w") as f:
        for index, (arr, string) in zip(indices, zip(input_arrs, input_strs)):
            data = {
                "index": index,
                "input_arr": arr.tolist(),
                "input_str": string
            }
            f.write(json.dumps(data) + "\n")

def count_tokens(input_strs, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)
    # Encode each string in the list and return the token IDs
    return [len(encoding.encode(text)) for text in input_strs]


def main(dataset_path, output_path, sample_size=10000, max_seq_len=128, serialize_prec=4, shuffle=False):
    ds = TimeMoEDataset(dataset_path)

    # randomly sample
    random_indices = random.sample(range(len(ds)), sample_size)

    # set serialization parameters
    settings = SerializerSettings(prec=serialize_prec)
    if isinstance(settings, dict):
        settings = SerializerSettings(**settings)

    input_arrs = []
    input_strs = []

    for idx in random_indices:
        seq = ds[idx]
        if len(seq) > max_seq_len:
            start = random.randint(0, len(seq) - max_seq_len)
            seq = seq[start:start + max_seq_len]
        input_arrs.append(seq)

        serialized_seq = serialize_arr(seq, settings)
        input_strs.append(str(serialized_seq))

    save_to_jsonl(input_arrs, input_strs, output_path, shuffle=shuffle)
    print(f"✅ saved {len(input_arrs)} samples to {output_path}")

    token_counts = count_tokens(input_strs, "gpt-4o-mini")
    print(f"max tokens: {max(token_counts)}")


if __name__ == "__main__":
    # Example usage with replaceable parameters
    dataset_path = "../Time-300B/web/wiki_daily_100k"   # to be changed
    output_path = "../middleware/meta/wiki_daily_100k/blocks.jsonl"   # to be changed
    sample_size = 10000          # to be changed
    max_seq_len = 128            # to be changed
    serialize_prec = 4           # to be changed
    shuffle = False
    random_seed = 42

    # === 固定随机种子 ===
    random.seed(random_seed)
    np.random.seed(random_seed)

    main(dataset_path, output_path, sample_size, max_seq_len, serialize_prec, shuffle)
