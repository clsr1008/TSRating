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
# 全局禁用 progress bar
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
        parser.add_argument("--offset", type=int, default=0) #从数据集中跳过的样本数量，起始点偏移量
        parser.add_argument("--num_examples_proportion", type=float, default=None) #按照数据集大小的比例选择样本
        parser.add_argument("--num_examples_proportion_start", type=float, default=0.0) #数据集起始点的比例

        parser.add_argument("--seed", type=int, default=42) #随机种子
        parser.add_argument("--model", type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gemini-2.0-flash", "gpt-4o-mini", "claude-3-5-haiku-20241022", "deepseek-reasoner", "gpt-4o-ca", "deepseek-reasoner", "claude-2"])

        parser.add_argument("-g", "--generations", type=int, default=20) #生成的样本数量（投票的次数）

        parser.add_argument("--template", type=str, default="")
        parser.add_argument("--template_file", type=str, default="annotate/templates/pairwise_default.txt") #提示词模板
        parser.add_argument("--labels", type=str, nargs=2, default=["A", "B"]) #两个候选项的标签

        parser.add_argument("--text_field", type=str, default="text") #input数据集中用于比较的文本字段名称
        parser.add_argument("--token_field", type=str, default="input_ids") #数据集中标记为 token 的字段名称（本例中无此字段）
        parser.add_argument("--tokens_min", type=int, default=256) #文本最小长度（以token为单位）
        parser.add_argument("--tokens_max", type=int, default=512)  #文本最大长度（以token为单位）
        parser.add_argument("--probability_tokens_max", type=float, default=1.0)  #限制文本长度为最大值的概率。例如，0.5 表示有 50% 的概率将文本长度限制为 --tokens_max
        parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neo-2.7B") #用于处理文本的 tokenizer 名称，原先是llama
        parser.add_argument("--ratio", type=float, default=1.0) #有效pairwise数量是block数多少倍
        parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.") #系统提示，初始角色描述

        parser.add_argument("--flat_output_format", action="store_true") #如果设置此标志，输出将以扁平格式保存，该参数出现在命令行中，则将其值设置为 True

    def __init__(self, args):
        self.args = args
        if self.args.template_file:
            with open(self.args.template_file) as f:
                self.args.template = f.read()

        # self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

        self.offset = 0
        self.high_confidence_count = 0  # 初始化高置信度比较对计数
        self.total_samples = 0 #单个时间块的总数
        self.processing_complete = False

    def __getstate__(self):
        return self.args

    def __setstate__(self, state):
        self.__init__(state)

    def extract_excerpt(self, text, index, num_tokens, token_ids=None):
        if token_ids is None:
            # heuristic for faster tokenization 启发式截断
            max_character_length = self.args.tokens_max * 40 #假设每个 token 平均长度为 40 个字符
            if len(text) > max_character_length:
                np.random.seed(self.args.seed + index + self.offset + 1)
                start_pos = np.random.randint(0, len(text) - max_character_length + 1)
                text = text[start_pos:start_pos + max_character_length] #从文本中截取长度为 max_character_length 的子串，减少分词时间

            # token_ids = self.tokenizer(text, truncation=False, padding=False, add_special_tokens=False).input_ids #使用分词器将文本转化为 token 序列 token_ids
            token_ids = encode_fn(text, self.args.model)

        if len(token_ids) <= self.args.tokens_max: #如果 token 的数量小于等于 self.args.tokens_max，直接返回原始文本（或截断后的）
            return text

        np.random.seed(self.args.seed + index + self.offset)
        start_pos = np.random.randint(0, len(token_ids) - self.args.tokens_max + 1)
        token_ids = token_ids[start_pos:start_pos + num_tokens] #token_ids长度超过tokens_max，再次截取
        # return self.tokenizer.decode(token_ids)  #使用分词器将提取的 token 序列解码回可读文本
        return decode_fn(token_ids, self.args.model)

    def parse_generations(self, generations):
        for generation in generations:
            if generation == self.args.labels[0]:
                yield 0
            elif generation == self.args.labels[1]:
                yield 1

    def sample_num_tokens(self, indices):
        np.random.seed(self.args.seed + sum(indices) + self.offset) #设置随机种子
        # use length tokens_max with probability probability_tokens_max, otherwise sample uniformly from [tokens_min, tokens_max]
        if self.args.probability_tokens_max > 0 and np.random.rand() < self.args.probability_tokens_max: #以probability_tokens_max概率返回tokens_max
            return self.args.tokens_max
        else:
            return np.random.randint(self.args.tokens_min, self.args.tokens_max + 1) #在区间内随机选一个token数

    def __call__(self, examples, indices): #examples是原始输入数据集中样本，对应indices下标（一个批次）
        num_tokens = self.sample_num_tokens(indices) #根据给定的索引 indices 和随机性设置，生成一个样本的 token 数量

        if self.args.token_field in examples:
            texts = [self.extract_excerpt(text, index, num_tokens, token_ids) for text, token_ids, index in zip(examples[self.args.text_field], examples[self.args.token_field], indices)]
        else: #该数据集执行此分支，只提取example中的text字段，并根据num_tokens数截取文本
            texts = [self.extract_excerpt(text, index, num_tokens) for text, index in zip(examples[self.args.text_field], indices)]

        n = len(texts)  #等于num_compare_all
        votes_a = np.zeros((n, n), dtype=np.int32)
        votes_b = np.zeros((n, n), dtype=np.int32)
        predictions = np.full((n, n), -100, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j: #确保不会对同一个文本与自身进行比较 (i,j)只可能是（0,1）或（1,0），一共投票2次，以消除位置偏差
                    continue

                prompt = self.args.template.format( #只有下标反映文本位置，标签AB不反映
                    text_a=texts[i],
                    text_b=texts[j],
                    label_a=self.args.labels[0],
                    label_b=self.args.labels[1])

                generations = query_openai( #输出是长20的列表，元素是标签A或B
                    prompt,
                    self.args.model,
                    system_prompt=self.args.system_prompt,
                    generations=self.args.generations, #输入是20,每次投20票
                    labels=self.args.labels)

                for vote in self.parse_generations(generations):
                    if vote == 0: #表示投票给A
                        votes_a[i, j] += 1
                    elif vote == 1: #表示投票给B
                        votes_b[i, j] += 1
        # print(votes_a)
        # print(votes_b)
        # total_sum = sum(sum(row) for row in votes_a) + sum(sum(row) for row in votes_b)
        np.divide(votes_b, votes_a + votes_b, out=predictions,
                  where=votes_a + votes_b != 0) #计算B获得的票数占总票数的比例（忽略对角）
        # print(predictions)
        calibrated_predictions = np.where(
            (predictions != -100) & (predictions.T != -100), #对角依然是-100
            (predictions + (1 - predictions.T)) / 2, # 两次投票取平均
            -100)
        # print(calibrated_predictions)
        #calibrated_predictions的右上角代表text 1的得票比例，左下角代表text 0的得票比例，两者相加是1
        indices_a, indices_b = np.where(np.triu(np.ones((n, n)), k=1))

        confidence_scores = calibrated_predictions[indices_a, indices_b][0]
        # print(confidence_scores)
        if (0 <= confidence_scores <= 1) and (confidence_scores <= 0.25 or confidence_scores >= 0.75):
            self.high_confidence_count += 1
            print(f"High confidence comparisons: {self.high_confidence_count}")

        # 检查是否达到阈值
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
        else: #返回简洁的输出版本
            return {
                "index_a": indices_a, #为0
                "index_b": indices_b,  #为1
                "block_a": [indices[0]],
                "block_b": [indices[1]],
                "texts_a": [texts[i] for i in indices_a], #text 0
                "texts_b": [texts[j] for j in indices_b], #text 1
                "comparisons_forward": predictions[indices_a, indices_b].tolist(), #第一次投票text 1的得票比例
                "comparisons_backward": (1-predictions[indices_b, indices_a]).tolist(), #第二次投票text 1的得票比例
                "comparisons_avg": calibrated_predictions[indices_a, indices_b].tolist(), #两次投票text 1的得票比例
            }

    def apply(self, dataset):
        total_samples = len(dataset)
        self.total_samples = total_samples
        print(f"Total samples in dataset: {total_samples}")

        # 生成所有可能的两两组合
        all_combinations = list(itertools.combinations(range(total_samples), 2))
        print(f"Total combinations generated: {len(all_combinations)}")

        # 根据参数筛选组合
        if self.args.num_examples_proportion is not None:
            offset = int(len(all_combinations) * self.args.num_examples_proportion_start)
            num_combinations = int(len(all_combinations) * self.args.num_examples_proportion)
        else:
            offset = self.args.offset
            num_combinations = self.args.num_examples

        # 确保 offset 和组合数量不超出范围
        offset = min(offset, len(all_combinations))
        num_combinations = min(num_combinations, len(all_combinations) - offset)

        # 筛选符合条件的组合
        selected_combinations = all_combinations[offset:offset + num_combinations]
        # 固定随机种子
        random.seed(42)
        # 打乱组合
        random.shuffle(selected_combinations)

        # 转换为索引子集，生成新的“展平”数据集
        flattened_indices = [idx for pair in selected_combinations for idx in pair]

        results = []
        for batch_start in tqdm(range(0, len(flattened_indices), self.args.num_compare_all), total=len(selected_combinations),
                                desc="Processing batches"):
            # 当前批次的原始索引
            batch_flattened_indices = flattened_indices[batch_start:batch_start + self.args.num_compare_all]
            # 当前批次对应的子数据集
            batch_dataset = dataset.select(batch_flattened_indices)

            #使用 process_with_original_indices 处理批次
            def process_with_original_indices(batch, indices):
                # 获取原始索引
                original_indices = [batch_flattened_indices[idx] for idx in indices]
                return self.__call__(batch, indices=original_indices)

            # 映射当前批次
            batch_results = batch_dataset.map(
                process_with_original_indices,
                with_indices=True,
                batched=True,
                batch_size=self.args.num_compare_all,
                remove_columns=dataset.column_names,
            )
            # 累积结果
            results.append(batch_results)
            # 检查是否达到阈值
            if self.processing_complete:
                print("High confidence threshold reached. Terminating.")
                break

        # 合并所有结果为 Dataset
        combined_results = Dataset.from_dict(
            {key: sum((result[key] for result in results), []) for key in results[0].column_names}
        )
        return combined_results

if __name__ == "__main__":
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
