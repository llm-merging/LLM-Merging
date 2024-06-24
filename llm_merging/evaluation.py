import json 
import os 
import evaluate 

from typing import List, Dict, Any

import torch
from tqdm import tqdm
from torch.utils import data

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


def evaluate_dataset(
    merge_method,
    dataset_filepath: str,
    eval_type: str,
    metric: str,
) -> (Dict, List):

    data_loader = data.DataLoader(
        Dataset(dataset_filepath),
        batch_size=1,
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
        accuracy = evaluate.load('accuracy')
        score = accuracy.compute(references=[example["label"] for example in all_batches], predictions=[example["predicted_choice"] for example in all_batches])
    elif metric == "rouge":
        rouge = evaluate.load('rouge')
        score = rouge.compute(
            predictions=[example["predicted_text"] for example in all_batches],
            references=[example["target"] for example in all_batches])
    elif metric == "none":
        score = {}
    else:
        raise NotImplementedError(f"Invalid metric {metric}")

    for metric, value in score.items():
        score[metric] = round(value, 3)

    return score, all_batches

def evaluate_model(
    merge_method,
    all_dataset_filepaths: List[str],
    all_eval_types: List[str],
    all_metrics: List[str],
) -> Dict:   
    output_dir = os.path.join("output", merge_method.get_name())
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    # Save merged model 
    merge_method.save_model(output_dir)

    all_scores = {}

    for dataset_filepath, eval_type, metric in zip(all_dataset_filepaths, all_eval_types, all_metrics):
        score, dataset_predictions = evaluate_dataset(merge_method, dataset_filepath, eval_type, metric)

        # Get dataset_name from filepath assuming the filepath is in the format of "data/{dataset_name}.json"
        dataset_name = dataset_filepath.split("/")[-1].replace(".json", "")
        all_scores[dataset_name] = score

        # Save the predictions 
        with open(os.path.join(prediction_dir, f"{dataset_name}.jsonl"), "w+") as f:
            for example in dataset_predictions:
                f.write(json.dumps(example) + "\n")        

    with open(os.path.join(output_dir, f"scores.jsonl"), "a+") as f:
        f.write(json.dumps(all_scores) + "\n")
