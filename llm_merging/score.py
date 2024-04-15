import argparse
import os

import torch

from llm_merging.merging import *
from llm_merging.eval import * 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_file", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.output_file)

    filename = os.path.basename(args.output_file)
    # filename has structure {dataset}_predictions.json
    dataset = str(filename[:filename.index("_")])

    dataset_predictions = []
    with open(args.output_file, "r") as f:
        for line in f.readlines():
            dataset_predictions.append(json.loads(line))


    if dataset == "accuracy":
        score, dataset_predictions = accuracy(dataset_predictions)
    elif dataset == "exact_match_multiple_references":
        score, dataset_predictions  = exact_match_multiple_references(dataset_predictions)
    elif dataset == "gsm8k":
        score, dataset_predictions = numerical_accuracy(dataset_predictions)
    elif dataset == "humaneval":
        score, dataset_predictions = humaneval_preprocess(dataset_predictions)
    else:
        raise NotImplementedError(f"Invalid dataset {dataset}")

    with open("temp_predictions.json", "w+") as f:
        for example in dataset_predictions:
            f.write(json.dumps(example) + "\n")