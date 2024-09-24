<div align="center">


<h1>LLM-Merging: Building LLMs Efficiently through Merging </h1>

[![](https://img.shields.io/badge/Documentation-online-green)](https://llm-merging.readthedocs.io)
[![](https://img.shields.io/badge/Website-online-green)](https://llm-merging.github.io)
[![](https://img.shields.io/badge/License-MIT-blue)](#License)
</div>

This repository contains the starter code for the LLM-Merging competition.

## Important Tips
1.  Please do not specify any device_id in the code because the device_id might not hold in our setup. If you need to specify a device_id in your setup, one solution is to use environment variables like
```bash
export CUDA_VISIBLE_DEVICES=0  
```
2. Please do not specify any filepaths because they may not be the same in our setup. If you need to specify the HuggingFace cache, one solution is to use environment variables like
```bash
export HUGGINGFACE_HUB_CACHE=/tmp/
```
and then access this path in Python via 
```python
path=os.environ["HUGGINGFACE_HUB_CACHE"]
```
3. When running `tar` on this repo `LLM-Merging` to submit it, please ensure this directory is called `LLM-Merging` and not renamed to any directories. This can cause issues when evaluating your submissions.   

## Setup Environment

The library was tested on CUDA 10.1 on an A6000.

```bash
conda env create -f environment.yml --name llm-merging
conda activate llm-merging
export PYTHONPATH=`pwd`
```

Authentication tokens are required for certain models like Llama2, which require users to agree to specific terms. You can find the authentication token [here](https://huggingface.co/settings/tokens).

```bash
export HF_AUTH_TOKEN=""
```

## Developing New Merging Methods

You may make some modifications to the starter kit code, constrained by the terms listed under the Submissions section.

1. To add a new merging method, create a new file in `llm_merging/merging`.

    This file should extend `llm_merging/merging/Merges` and implement `__init__()` and `merge()` functions.
    See `llm_merging/merging/FlanT5Avg.py`, `llm_merging/merging/LlamaAvg.py`, and `llm_merging/merging/TinyLlamaAvg.py` for examples.

2. Add the new merging method to the dictionary returned by `all_merge_handlers()` in `llm_merging/main.py`

3. Add the new module to `llm_merging/merging/__init__.py`

4. Add any additional required libraries to `setup.py`.

## Test Method

```bash
python llm_merging/setup.py install
python llm_merging/main.py -m {merging_method}
```

The validation dataset (consisting of CosmosQA and XSum) is mainly included to ensure the merging method (with evaluation on those datasets) runs in under the 1-hour time limit. Our results on `llama_avg` are `{"cosmos_qa": {"accuracy": 0.234}, "xsum": {"rouge1": 0.123, "rouge2": 0.023, "rougeL": 0.093, "rougeLsum": 0.102}}`, which run in about 25 minutes on our A6000.

## Submissions

Most modifications to the starter kit are allowed. In general, any change that honors the spirit of the competition, to understand how best to merge models, will be allowed. For example, modifying the generation code to allow for dynamic selection of a prompt is allowed, whether that's your own code or imported. Changes which are not allowed might include optimizations focused only on making the forward pass of a model faster. This is because squeezing out such speedups does not contribute to drawing any conclusions about the merging method.

You may use any publicly available library/module, as long as we will also be able to install it easily from a requirements or dependency list. You may finetune the model, but keep in mind that the compute limits apply to the finetuning stage as well, and you will eventually need to run your code, exactly as it is, on a held out test set that you will not have access to until after you finalize and submit your code.

### How to submit

You must submit the output file on Kaggle, and the model files via the instructions below.

First, generate the output file, using the input dataset file found in `data/test.csv`. Name your output file `submission.csv`.
To submit to Kaggle, go to our [Kaggle competition site](https://www.kaggle.com/competitions/llm-merging-competition/overview) and click `Submit Prediction`, uploading your `submission.csv`.

Next, tar this repo for submission:

```bash
tar -cvf {merging_method}.tar LLM-Merging
```

Submit the tar file using this [form](https://docs.google.com/forms/d/17TPg7N02o8qvw1czx55Zbh_5Kp7-YStUIOhQDJYc23g/)

## Leaderboard

The leaderboard being used is on our [Kaggle competition site](https://www.kaggle.com/competitions/llm-merging-competition/overview).
The leaderboard's standings are *not* final.The final results of the competition will be calculated after the conclusion of the competition. At that point, we will release the inputs for our final held out evaluation, and you will have a week to run your model code on this input. The input will be in the same format as the `test.csv` file in this competition. You will then be responsible for submitting this final output file to us. For all top placers, we will be verifying that the code you submitted via the form before the closing of the competition does indeed yield your final submission csv. 

The old leaderboard of the submitted solutions can be found [here](https://huggingface.co/spaces/margsli/merging_competition). 
