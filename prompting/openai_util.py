from typing import List
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import openai
from openai import OpenAI
import tiktoken
import os
import time
import json
from filelock import FileLock
import random


RANDOM = random.Random()


def query_openai(prompt: str,
                 model: str,
                 labels: List[str],
                 system_prompt: str = None,
                 generations: int = 1,
                 retries: int = 1,
                 log_file_path: int = "openai_api_cost.jsonl") -> List[str]:
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    client = OpenAI( # you can replace it with your own OPENAI proxy
        api_key= os.environ["OPENAI_API_KEY"],
        base_url="https://api.chatanywhere.tech/v1"
    )

    is_ok = False
    retry_count = 0

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    label_tokens = [enc.encode(label) for label in labels]
    logit_bias = {
        str(token): 100
        for token in set.union(*(set(tokens) for tokens in label_tokens))
    }
    max_tokens = max(len(tokens) for tokens in label_tokens)

    while not is_ok:
        try:
            # start_time = time.time()

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1.0,
                max_tokens=max_tokens,
                n=generations,
                logit_bias=logit_bias,
            )

            # end_time = time.time()
            # processing_time = end_time - start_time
            # print(f"Model processing time: {processing_time:.2f} seconds")

            is_ok = True
        except Exception as error:
            if "Please retry after" in str(error):
                timeout = int(str(error).split("Please retry after ")[1].split(" second")[0]) + 5*RANDOM.random()
                print(f"Wait {timeout}s before OpenAI API retry ({error})")
                time.sleep(timeout)
            elif retry_count < retries:
                print(f"OpenAI API retry for {retry_count} times ({error})")
                time.sleep(10)
                retry_count += 1
            else:
                print(f"OpenAI API failed for {retry_count} times ({error})")
                return []

    generations = [choice.message.content for choice in response.choices]
    generations = [text.strip() for text in generations]
    # print(generations)
    return generations


def query_anthropic(prompt: str,
                    model: str = "claude-2") -> List[str]:
    is_ok = False
    retry_count = 0

    while not is_ok:
        retry_count += 1
        try:
            anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            completion = anthropic.completions.create(
                model=model,
                max_tokens_to_sample=5,
                prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
            )
            return [completion.completion]
        except Exception as error:
             if retry_count <= 2:
                 print(f"OpenAI API retry for {retry_count} times ({error})")
                 time.sleep(2)
             else:
                 print(f"OpenAI API failed for {retry_count} times ({error})")
                 return []
