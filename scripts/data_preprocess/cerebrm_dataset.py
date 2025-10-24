"""
Preprocess the CerebRM dataset to parquet format
"""

import os
import datasets
from transformers import AutoTokenizer

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data/cerebrm_dataset')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_start', type=int, default=0)
    parser.add_argument('--train_end', type=int, default=10000000)
    parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen2.5-Math-1.5B')

    args = parser.parse_args()

    data_source = 'CodeShield/CerebRM-Dataset'
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    train_dataset = dataset['train']
    test_dataset = dataset['test_weak_easy']
    LIST_REWARD_PROMPT = """You are an expert judge of coding problems. Given a coding problem and multiple candidate solutions, your task is to evaluate the correctness of each solution based on the problem description. Your evaluation should solely be based on the functional correctness of the codes. It is guaranteed that one and only one of the candidates is completely correct. Here is the coding question followed by the candidate solutions:

[QUESTION]
{question}
[/QUESTION]

{candidates}

You are to indicate your choice of candidate only by responding with one of the following options: {valid_options}. Enclose your final answer in the format \\boxed{{X}}, where X is your chosen option among the candidates. Do not provide any explanations or additional text. Your response should be exactly one of the options enclosed within \\boxed{{}}, without any extra characters or spaces. Anything else will be considered invalid.
"""
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
            candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
            candidate_str = "\n\n".join(candidates)
            reward_model = {
                "style": "rule",
                "ground_truth": example['chosen_answer']
            }

            data = {
                "data_source": 'cerebrm_dataset',
                "prompt": [
                    {
                        "role": "user",
                        "content": LIST_REWARD_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                    valid_options=", ".join(potential_answers),
                ).strip(),
                    }
                ],
                "ability": "mcq",
                "reward_model": reward_model,
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'num_candidates': example['num_candidates'],
                }
            }
            return data

        return process_fn
        
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True, remove_columns=test_dataset.column_names)
    print(train_dataset[0])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
                                              
