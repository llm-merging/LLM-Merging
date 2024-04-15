import argparse
from tqdm import tqdm

import torch

from llm_merging.merging import *
from llm_merging.eval import * 

DATASETS = [
    "boolq",
    "triviaqa",
    "gsm8k",
    "humaneval"
]

METHODS = {
    "llama_avg": LlamaAvg,
    "flan_t5_avg": FlanT5Avg
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str)
    parser.add_argument("-d", "--datasets", type=str, nargs='+')
    parser.add_argument("-b", "--eval_batch_size", type=int, default=2)
    args = parser.parse_args()

    # Check datasets are valid 
    for dataset in args.datasets:
        assert dataset in DATASETS

    # Load correct merging method 
    merge_method = METHODS[args.merging_method]()

    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()

    # Evaluate method 
    evaluate_model(
        merge_method,
        args.datasets,
        args.eval_batch_size
    )