import argparse


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
    parser.add_argument("--dataset_filepaths", type=str, default=None, nargs='+')
    parser.add_argument("--eval_types", type=str, default=None, nargs='+')
    args = parser.parse_args()

    # Load correct merging method 
    loaded_merges = all_merge_handlers()
    merge_method = loaded_merges[args.merging_method](args.merging_method)

    # Call the merge function. The merged model is stored under merging_method object 
    merge_method.merge()

    if args.dataset_filepaths is not None:
        assert args.eval_types is not None, "If dataset_filepaths is passed, eval_types"

        assert len(args.dataset_filepaths) == len(args.eval_types), "All lists should be of the same length"

        # Evaluate method on datsets passed in (used for testing)
        evaluate_model(
            merge_method,
            args.dataset_filepaths,
            args.eval_types,
            ["none"] * len(args.dataset_filepaths),
        )
    else:
        assert args.eval_types is None, "If dataset_filepaths is not passed, eval_types should not be passed in"
        # Evaluate method on fixed datasets (used for developing method)
        evaluate_model(
            merge_method,
            ["data/cosmos_qa.json", "data/xsum.json"],
            ["multiple_choice", "generation"],
            ["accuracy", "rouge"],
        )

