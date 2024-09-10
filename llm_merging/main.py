import argparse
import sys

from importlib.metadata import entry_points

from llm_merging.evaluation import * 
from llm_merging.data import * 
from llm_merging.merging import *

def all_merge_handlers():
    """Enumerate and Load (import) all merge methods."""
    loaded_merges = {
        "llama_avg": LlamaAvg,
        "tiny_llama_avg": TinyLlamaAvg,
        "flant5_avg": FlanT5Avg,
        ## TODO Add more merge methods here
    }
    
    return loaded_merges

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--merging_method", type=str, required=True)
    parser.add_argument(
        "--dataset_filepaths", 
        type=str, 
        default=["data/validation/cosmos_qa.csv", "data/validation/xsum.csv"], 
        nargs='+'
    )
    parser.add_argument("--output_folder", type=str, default="../output")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    loaded_merges = all_merge_handlers()
    merge_method = loaded_merges[args.merging_method](args.merging_method)

    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()
    
    # Evaluate method on datsets passed in (used for testing)
    evaluate_model(
        merge_method,
        args.dataset_filepaths,
        args.output_folder,
    )


