import logging
import json 
import os 
from typing import List, Tuple, Dict, Callable, Any

import torch
import torch.distributed as dist
from tqdm import tqdm
from torch.utils import data

from llm_merging.data import *
from llm_merging.eval.metrics import accuracy, exact_match_multiple_references, numerical_accuracy, humaneval_preprocess

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


def evaluate_dataset(
    merge_method,
    dataset: str,
    eval_batch_size: int
) -> (Dict, List):
    
    # Different datasets have different evaluation functions to call and different metrics 
    if dataset == "boolq":
        dataset = BoolQDataset(split="validation", max_examples_per_dataset=100, round_robin_template=True)
        eval_type = "multiple_choice"
        metric = "accuracy"
    elif dataset == "triviaqa":
        dataset = TriviaQADataset(split="validation", max_examples_per_dataset=100, round_robin_template=True)
        eval_type = "generation"
        metric = "exact_match_multiple_references"
    elif dataset == "gsm8k":
        dataset = GSM8kDataset(split="test", max_examples_per_dataset=None, round_robin_template=True)
        eval_type = "generation"
        metric = "numerical_accuracy"
    elif dataset == "humaneval":
        dataset = HumanEvalDataset(split="test", max_examples_per_dataset=None, round_robin_template=True)
        eval_type = "generation"
        metric = "humaneval" 
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
    elif metric == "exact_match_multiple_references":
        score, all_batches  = exact_match_multiple_references(all_batches)
    elif metric == "numerical_accuracy":
        score, all_batches = numerical_accuracy(all_batches)
    # Note that the human eval score is not calculated since it requires calling another library. Instead, the predictions are just saved  
    elif metric is "humaneval":
        score, all_batches = humaneval_preprocess(all_batches)
    else:
        raise NotImplementedError(f"Invalid metric {metric}")
    return score, all_batches

def evaluate_model(
    merge_method,
    list_datasets: List[str],
    eval_batch_size: int
) -> Dict:
    logging.info(f"Evaluating model")

    output_dir = os.path.join("output", merge_method.get_name())
    os.makedirs(output_dir, exist_ok=True)

    all_scores = {}

    # Loop through and evaluate the merge method on all the datasets
    for dataset in list_datasets:
        score, dataset_predictions = evaluate_dataset(merge_method, dataset, eval_batch_size)
        all_scores[dataset] = score

        # Save the predictions 
        with open(os.path.join(output_dir, f"{dataset}_predictions.jsonl"), "w+") as f:
            for example in dataset_predictions:
                f.write(json.dumps(example) + "\n")

    with open(os.path.join(output_dir, f"scores.jsonl"), "a+") as f:
        f.write(json.dumps(all_scores) + "\n")
