import argparse
from tqdm import tqdm

import torch

from importlib.metadata import entry_points

from llm_merging.evaluation import * 
from llm_merging.data import * 

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    discovered_merges = entry_points(group="llm_merging.merging.Merges")
    loaded_merges = {ep.name: ep.load() for ep in discovered_merges}
    return loaded_merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str, required=True)
    parser.add_argument("-b", "--eval_batch_size", type=int, default=2)
    args = parser.parse_args()

    # Load correct merging method 
    loaded_merges = all_merge_handlers()
    merge_method = loaded_merges[args.merging_method]()

    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()

    # Evaluate method 
    evaluate_model(
        merge_method,
        ["boolq", "mawps"],
        args.eval_batch_size
    )

