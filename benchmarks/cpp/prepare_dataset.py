#!/usr/bin/python

import argparse
import json

import random
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, LlamaTokenizer, T5Tokenizer, PreTrainedTokenizerBase

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        required=True,
                        help='Dataset path used for the test.')
    parser.add_argument('--max_input_len',
                        type=int,
                        required=True,
                        help='Specify max input length')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        required=True,
                        help='Specify tokenizer directory')
    parser.add_argument('--tokenizer_type',
                        type=str,
                        default='auto',
                        required=False,
                        choices=['auto', 't5', 'llama'],
                        help='Specify tokenizer type')
    parser.add_argument('--output',
                        type=str,
                        default='preprocessed_dataset.json',
                        help='Preprocessed dataset path.')
    parser.add_argument("--num-prompts",
                        type=int,
                        default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    FLAGS = parser.parse_args()

    if FLAGS.tokenizer_type == 't5':
        tokenizer = T5Tokenizer(vocab_file=FLAGS.tokenizer_dir,
                                padding_side='left')
    elif FLAGS.tokenizer_type == 'auto':
        tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                  padding_side='left')
    elif FLAGS.tokenizer_type == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained(FLAGS.tokenizer_dir,
                                                   legacy=False,
                                                   padding_side='left')
    else:
        raise AttributeError(
            f'Unexpected tokenizer type: {FLAGS.tokenizer_type}')
    tokenizer.pad_token = tokenizer.eos_token

    random.seed(0)
    requests = sample_requests(FLAGS.dataset, FLAGS.num_prompts, tokenizer, FLAGS.output_len)

    results = []
    for prompt, _, output_len in requests:
        line = tokenizer.encode(prompt)
        results.append({'input_ids': line, 'output_len': output_len})

    with open(FLAGS.output, 'w') as f:
        json.dump(results, f)
