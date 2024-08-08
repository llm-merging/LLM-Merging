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

Do not modify any files other than the new file you create and `setup.py`. Doing so can result in the grounds for invalidating your submission. If you need to change code in other files, feel free to open a pull request.

1. To add a new merging method, create a new file in `llm_merging/merging`.

    This file should implement `__init__.py` and `merge.py` functions and extend `llm_merging/merging/Merges`.
    See `llm_merging/merging/FlanT5Avg.py` or `llm_merging/merging/LlamaAvg.py` for examples.

2. Modify `setup.py` and add an entry with the merging method in `llm_merging.merging.Merges`.

    For example, the entry `llama_avg = llm_merging.merging.LlamaAvg:LlamaAvg` indicates the method is called `llama_avg` and the file is at `llm_merging/merging/LlamaAvg`.

    Any additional required libraries can be specified in `setup.py`.

## Test Method

```bash
python llm_merging/setup.py install
python llm_merging/main.py -m {merging_method}
```

The datasets (CosmosQA and XSum) are mainly included to ensure the merging method (with evaluation on those datasets) runs in under the 1-hour time limit. Our results on `llama_avg` are `{"cosmos_qa": {"accuracy": 0.234}, "xsum": {"rouge1": 0.123, "rouge2": 0.023, "rougeL": 0.093, "rougeLsum": 0.102}}`, which run in about 25 minutes on our A6000.

## Submissions

After modifying the file, tar the file into a tarball using the command:

```bash
tar -cvf {merging_method}.tar LLM-Merging
```

Submit the tar file using this [form](https://docs.google.com/forms/d/17TPg7N02o8qvw1czx55Zbh_5Kp7-YStUIOhQDJYc23g/)

## Leaderboard

The leaderboard of the submitted solutions can be found [here](https://huggingface.co/spaces/margsli/merging_competition). Please note that your submission might not appear on the leaderboard immediately, as it is updated every few days. If you encounter any issues, please contact us.

Note: This submission method is only temporary and another automatic submission method should be comming soon.
