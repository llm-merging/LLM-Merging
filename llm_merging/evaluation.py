import evaluate 
import json 
import os 
import pandas as pd

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
            eval_type = batch["eval_type"][0]
            if eval_type == "multiple_choice":
                (
                    predicted_choice,
                    answer_choice_scores,
                ) = merge_method.predict_multiple_choice(batch)

                batch["prediction"] = str(predicted_choice.cpu().numpy().tolist()[0])
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))
            
            else:
                assert eval_type == "generation"
                (
                    generated_ids, generated_txt
                ) = merge_method.generate(batch
                )
                batch["prediction"] = generated_txt 
                all_batches.extend(convert_dict_of_lists_to_list_of_dicts(batch))

    return all_batches


def evaluate_model(
    merge_method,
    all_dataset_filepaths: List[str],
    output_folder: str,
) -> Dict:   
    output_dir = os.path.join("output", merge_method.get_name())
    prediction_dir = os.path.join(output_dir, "predictions")
    os.makedirs(prediction_dir, exist_ok=True)
    # Save merged model 
    merge_method.save_model(output_dir)

    all_scores = {}

    for dataset_filepath in all_dataset_filepaths:
        dataset_predictions = evaluate_dataset(merge_method, dataset_filepath)
        dp_df = pd.DataFrame(dataset_predictions)
        dp_df["dummy_field"] = 0
        fn = os.path.basename(dataset_filepath)
        dp_df.to_csv(f"{output_folder}/{fn}", columns=["id", "prediction", "dummy_field"], index=False)