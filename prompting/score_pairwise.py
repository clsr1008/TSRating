import random
import tiktoken
import itertools
from tqdm import tqdm
from datasets import load_from_disk, load_dataset, Dataset
from transformers import AutoTokenizer
import argparse
import numpy as np
from openai_util import query_openai
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def encode_fn(text, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    # Encode the input text and return the token IDs
    token_ids = encoding.encode(text)
    return token_ids

def decode_fn(token_ids, model):
    # Get the tokenizer encoding for the specific model
    encoding = tiktoken.encoding_for_model(model)
    # Decode the token IDs back to the original string
    decoded_text = encoding.decode(token_ids)
    return decoded_text

class Comparator:
    @staticmethod
    def add_args(parser):
        parser.add_argument("-k", "--num_compare_all", type=int, default=2)
        parser.add_argument("-n", "--num_examples", type=int, default=100)
        parser.add_argument("--offset", type=int, default=0) # Number of samples to skip from the dataset, starting point offset
        parser.add_argument("--num_examples_proportion", type=float, default=None) # Sample selection proportion based on dataset size
        parser.add_argument("--num_examples_proportion_start", type=float, default=0.0) # Proportion of the starting point of the dataset

        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gemini-2.0-flash", "gpt-4o-mini", "claude-3-5-haiku-20241022", "deepseek-reasoner", "gpt-4o-ca", "deepseek-reasoner", "claude-2"])

        parser.add_argument("-g", "--generations", type=int, default=20) # Number of generated samples (number of votes)

        parser.add_argument("--template", type=str, default="")
        parser.add_argument("--template_file", type=str, default="annotate/templates/pairwise_default.txt") # Prompt template
        parser.add_argument("--labels", type=str, nargs=2, default=["A", "B"]) # Labels for the two candidates

        parser.add_argument("--text_field", type=str, default="text")
        parser.add_argument("--token_field", type=str, default="input_ids")
        parser.add_argument("--tokens_min", type=int, default=256)
        parser.add_argument("--tokens_max", type=int, default=512)
        parser.add_argument("--probability_tokens_max", type=float, default=1.0) # Probability of limiting the series length to the maximum value
        parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-2.7B")
        parser.add_argument("--ratio", type=float, default=1.0)
        parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")

        parser.add_argument("--flat_output_format", action="store_true")

    def __init__(self, args):
        self.args = args
        if self.args.template_file:
            with open(self.args.template_file) as f:
                self.args.template = f.read()

        # self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

        self.offset = 0
        self.high_confidence_count = 0  # Initialize the count of high-confidence comparisons
        self.total_samples = 0  # Total number of individual time blocks
        self.processing_complete = False

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    def extract_excerpt(self, text, index, num_tokens, token_ids=None):
        if token_ids is None:
            # heuristic for faster tokenization
            max_character_length = self.args.tokens_max * 40 # Assume each token has an average length of 40 characters
            if len(text) > max_character_length:
                np.random.seed(self.args.seed + index + self.offset + 1)
                start_pos = np.random.randint(0, len(text) - max_character_length + 1)
                text = text[start_pos:start_pos + max_character_length]

            # token_ids = self.tokenizer(text, truncation=False, padding=False, add_special_tokens=False).input_ids
            token_ids = encode_fn(text, self.args.model)

        if len(token_ids) <= self.args.tokens_max:
            return text

        np.random.seed(self.args.seed + index + self.offset)
        start_pos = np.random.randint(0, len(token_ids) - self.args.tokens_max + 1)
        token_ids = token_ids[start_pos:start_pos + num_tokens]
        # return self.tokenizer.decode(token_ids)
        return decode_fn(token_ids, self.args.model)

    def parse_generations(self, generations):
        for generation in generations:
            if generation == self.args.labels[0]:
                yield 0
            elif generation == self.args.labels[1]:
                yield 1

    def sample_num_tokens(self, indices):
        np.random.seed(self.args.seed + sum(indices) + self.offset)
        # use length tokens_max with probability probability_tokens_max, otherwise sample uniformly from [tokens_min, tokens_max]
        if self.args.probability_tokens_max > 0 and np.random.rand() < self.args.probability_tokens_max:
            return self.args.tokens_max
        else:
            return np.random.randint(self.args.tokens_min, self.args.tokens_max + 1)

    def __call__(self, examples, indices):
        num_tokens = self.sample_num_tokens(indices) # Generate the number of tokens for a sample based on the given indices and randomness settings

        if self.args.token_field in examples:
            texts = [self.extract_excerpt(text, index, num_tokens, token_ids) for text, token_ids, index in zip(examples[self.args.text_field], examples[self.args.token_field], indices)]
        else:
            texts = [self.extract_excerpt(text, index, num_tokens) for text, index in zip(examples[self.args.text_field], indices)]

        n = len(texts)
        votes_a = np.zeros((n, n), dtype=np.int32)
        votes_b = np.zeros((n, n), dtype=np.int32)
        predictions = np.full((n, n), -100, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                prompt = self.args.template.format(
                    text_a=texts[i],
                    text_b=texts[j],
                    label_a=self.args.labels[0],
                    label_b=self.args.labels[1])

                generations = query_openai(
                    prompt,
                    self.args.model,
                    system_prompt=self.args.system_prompt,
                    generations=self.args.generations,
                    labels=self.args.labels)

                for vote in self.parse_generations(generations):
                    if vote == 0:
                        votes_a[i, j] += 1
                    elif vote == 1:
                        votes_b[i, j] += 1
        # print(votes_a)
        # print(votes_b)
        np.divide(votes_b, votes_a + votes_b, out=predictions,
                  where=votes_a + votes_b != 0) #计算B获得的票数占总票数的比例（忽略对角）
        calibrated_predictions = np.where(
            (predictions != -100) & (predictions.T != -100), #对角依然是-100
            (predictions + (1 - predictions.T)) / 2, # 两次投票取平均
            -100)
        # print(calibrated_predictions)
        # The top-right of calibrated_predictions represents the vote proportion for text 1,
        # while the bottom-left represents the vote proportion for text 0. The sum of both is 1.
        indices_a, indices_b = np.where(np.triu(np.ones((n, n)), k=1))

        confidence_scores = calibrated_predictions[indices_a, indices_b][0]
        # print(confidence_scores)
        if (0 <= confidence_scores <= 1) and (confidence_scores <= 0.25 or confidence_scores >= 0.75):
            self.high_confidence_count += 1
            print(f"High confidence comparisons: {self.high_confidence_count}")

        if self.high_confidence_count >= self.total_samples * self.args.ratio:
            self.processing_complete = True

        if not self.args.flat_output_format:
            return {
                "indices": [indices],
                "examples": [examples],
                "texts": [texts],
                "votes_a": [votes_a.tolist()],
                "votes_b": [votes_b.tolist()],
                "average": [calibrated_predictions.tolist()],
            }
        else:
            return {
                "index_a": indices_a, # 0
                "index_b": indices_b,  # 1
                "block_a": [indices[0]],
                "block_b": [indices[1]],
                "texts_a": [texts[i] for i in indices_a], #text 0
                "texts_b": [texts[j] for j in indices_b], #text 1
                "comparisons_forward": predictions[indices_a, indices_b].tolist(), # The vote proportion for text 1 in the first vote
                "comparisons_backward": (1-predictions[indices_b, indices_a]).tolist(), # The vote proportion for text 1 in the second vote
                "comparisons_avg": calibrated_predictions[indices_a, indices_b].tolist(), # The average vote proportion for text 1 from both votes
            }

    def apply(self, dataset):
        total_samples = len(dataset)
        self.total_samples = total_samples
        print(f"Total samples in dataset: {total_samples}")

        # Generate all possible pairwise combinations
        all_combinations = list(itertools.combinations(range(total_samples), 2))
        print(f"Total combinations generated: {len(all_combinations)}")

        if self.args.num_examples_proportion is not None:
            offset = int(len(all_combinations) * self.args.num_examples_proportion_start)
            num_combinations = int(len(all_combinations) * self.args.num_examples_proportion)
        else:
            offset = self.args.offset
            num_combinations = self.args.num_examples

        offset = min(offset, len(all_combinations))
        num_combinations = min(num_combinations, len(all_combinations) - offset)

        selected_combinations = all_combinations[offset:offset + num_combinations]
        random.seed(42)
        random.shuffle(selected_combinations)

        flattened_indices = [idx for pair in selected_combinations for idx in pair]

        results = []
        for batch_start in tqdm(range(0, len(flattened_indices), self.args.num_compare_all), total=len(selected_combinations),
                                desc="Processing batches"):
            batch_flattened_indices = flattened_indices[batch_start:batch_start + self.args.num_compare_all]
            batch_dataset = dataset.select(batch_flattened_indices)

            #use process_with_original_indices to process batches
            def process_with_original_indices(batch, indices):
                # 获取原始索引
                original_indices = [batch_flattened_indices[idx] for idx in indices]
                return self.__call__(batch, indices=original_indices)

            batch_results = batch_dataset.map(
                process_with_original_indices,
                with_indices=True,
                batched=True,
                batch_size=self.args.num_compare_all,
                remove_columns=dataset.column_names,
            )
            results.append(batch_results)
            if self.processing_complete:
                print("High confidence threshold reached. Terminating.")
                break

        combined_results = Dataset.from_dict(
            {key: sum((result[key] for result in results), []) for key in results[0].column_names}
        )
        return combined_results

if __name__ == "__main__":
    # do not run here, run run_score_pairwise instead
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)

    parser.add_argument("--json", action="store_true")

    Comparator.add_args(parser)

    args = parser.parse_args()
    print(args)

    if args.json:
        dataset = load_dataset("json", data_files=[args.input], split="train", download_mode='force_redownload')
    else:
        dataset = load_from_disk(args.input)

    dataset = Comparator(args).apply(dataset)


    print(f"Saving to {args.output}")
    dataset.save_to_disk(args.output)
