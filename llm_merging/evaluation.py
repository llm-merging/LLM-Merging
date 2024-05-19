import logging
import json 
import os 
import re
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from torch.utils import data
from string import punctuation

from llm_merging.data import *


def convert_dict_of_lists_to_list_of_dicts(dict_of_lists: Dict[Any, List]) -> List[Dict]:
    """
    Args:
        dict_of_lists:

    Returns:
        list_ofDict
    """
    list_of_dicts = []
    for datapoint_values in zip(*dict_of_lists.values()):
        list_of_dicts.append(dict(zip(dict_of_lists, datapoint_values)))
    return list_of_dicts

def collate_fn(batch_of_datapoints: List[Dict]) -> Dict[Any, List]:
    """
    Convert a batch of datapoints into a datapoint that is batched. This is meant to override the default collate function in pytorch and specifically can handle when the value is a list 

    Args:
        batch_ofDatapoints:

    Returns:

    """
    datapoint_batched = {}
    for datapoint in batch_of_datapoints:
        # Gather together all the values per key
        for key, value in datapoint.items():
            if key in datapoint_batched:
                datapoint_batched[key].append(value)
            else:
                datapoint_batched[key] = [value]
    return datapoint_batched

def extract_predicted_number(predicted_txt):
    """
    The number is extracted according to the following order:
    1. Use 
    2. Getting the number right after '='

    Afterwards,
    - Commas are removed from the number
    """

    # Only keep the suffix after the last equal sign
    split_equalSignPrefix = predicted_txt.split("=")
    equalSign_suffix = split_equalSignPrefix[-1]
    numbers_afterEqualSign = re.findall( r"\d+(?:[,.]\d+)?", equalSign_suffix)

    if len(split_equalSignPrefix) > 1 and len(numbers_afterEqualSign) > 0:
        remaining_answer = split_equalSignPrefix[-1]
        number_idx = 0

    else:
        remaining_answer = predicted_txt
        # Use the last number in the text
        number_idx = -1

    # Remove comma in numbers large numbers
    # Find all numbers (including decimals) and return the last number
    numerical_answer = re.findall(r"\d+(?:[,.]\d+)?", remaining_answer)
    if len(numerical_answer) > 0:
        return numerical_answer[number_idx].replace(
                punctuation, ""
            ).replace(",", "")
    else:
        return None


def accuracy(all_batches):
    num_correct = 0
    total = 0

    for example in all_batches:
        if example["predicted_choice"] == example["label"]:
            num_correct += 1
    
    total = len(all_batches)

    return {
        "accuracy": float(num_correct) / total
    }, all_batches



def numerical_accuracy(all_batches):
    num_correct = 0

    for example in all_batches:

        gold_number = example["target"]
        predicted_number = extract_predicted_number(example["predicted_text"])
        example["gold_number"] = gold_number
        example["predicted_number"] = predicted_number

        if predicted_number is not None:
            if "/" in gold_number:
                numerator, denominator = gold_number.split("/")
                gold_number =  float(numerator) / float(denominator)
            
            is_correct = float(predicted_number) == float(gold_number)
            if is_correct:
                num_correct += 1

    total = len(all_batches)

    return {
        "accuracy": float(num_correct) / total
    }, all_batches

def evaluate_dataset(
    merge_method,
    dataset: str,
    eval_batch_size: int,
    dataset_filepath=None
) -> (Dict, List):
    
    # Different datasets have different evaluation functions to call and different metrics 
    if dataset == "boolq":
        dataset = BoolQDataset(split="validation", max_examples_per_dataset=100, round_robin_template=True)
        eval_type = "multiple_choice"
        metric = "accuracy"
    elif dataset == "mawps":
        dataset = MAWPSDataset(split="validation", max_examples_per_dataset=100, round_robin_template=True)
        eval_type = "generation"
        metric = "numerical_accuracy"
    elif dataset == "text_file":
        dataset = TextFileDataset(dataset_filepath)
        eval_type = "multiple_choice"
        metric = "accuracy"
    else:
        raise NotImplementedError

    data_loader = data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn
    )

    all_batches = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            # There are two types of evaluation models:
            # 1) multiple choice where the model scores each choice and predicts the choice with the highest score 
            # 2) generation where the model generate some output give some input 
            if eval_type == "multiple_choice":
                (
                    predicted_choice,
                    answer_choice_scores,
                ) = merge_method.predict_multiple_choice(batch)

                batch["predicted_choice"] = predicted_choice.cpu().numpy().tolist()
                batch["answer_choices_scores"] = answer_choice_scores.cpu().numpy().tolist()
                
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))
            
            else:
                assert eval_type == "generation"
                (
                    generated_ids, generated_txt
                ) = merge_method.generate(batch
                )
                batch["predicted_ids"] = generated_ids
                batch["predicted_text"] = generated_txt 
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))

    if metric == "accuracy":
        score, all_batches = accuracy(all_batches)
    elif metric == "numerical_accuracy":
        score, all_batches = numerical_accuracy(all_batches)
    else:
        raise NotImplementedError(f"Invalid metric {metric}")
    return score, all_batches

def evaluate_model(
    merge_method,
    list_datasets: List[str],
    list_dataset_filepaths: List[str],
    eval_batch_size: int
) -> Dict:
    logging.info(f"Evaluating model")
    assert list_datasets is None or list_dataset_filepaths is None, f"Assume either list of datasets or list of dataset filepaths are passed in"
    output_dir = os.path.join("output", merge_method.get_name())
    os.makedirs(output_dir, exist_ok=True)

    all_scores = {}

    # Evaluate all list of datasets 
    if list_datasets is not None:
        for dataset in list_datasets:
            score, dataset_predictions = evaluate_dataset(merge_method, dataset, eval_batch_size, dataset_filepath=None)
            all_scores[dataset] = score

            # Save the predictions 
            with open(os.path.join(output_dir, f"{dataset}_predictions.jsonl"), "w+") as f:
                for example in dataset_predictions:
                    f.write(json.dumps(example) + "\n")
    # Evaluate all list of dataset filepaths 
    else:
        assert list_dataset_filepaths is not None

        for dataset_filepath in list_dataset_filepaths:
            score, dataset_predictions = evaluate_dataset(merge_method, "text_file", eval_batch_size, dataset_filepath)

            dataset_name = dataset_filepath.split("/")[-1].replace(".json", "")
            all_scores[dataset_name] = score
            # Save the predictions 
            with open(os.path.join(output_dir, f"{dataset_name}_predictions.jsonl"), "w+") as f:
                for example in dataset_predictions:
                    f.write(json.dumps(example) + "\n")        

    with open(os.path.join(output_dir, f"scores.jsonl"), "a+") as f:
        f.write(json.dumps(all_scores) + "\n")
