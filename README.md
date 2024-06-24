<div align="center">


<h1>LLM-Merging: Building LLMs Efficiently through Merging </h1>

[![](https://img.shields.io/badge/Documentation-online-green)](https://llm-merging.readthedocs.io)
[![](https://img.shields.io/badge/Website-online-green)](https://llm-merging.github.io)
[![](https://img.shields.io/badge/License-MIT-blue)](#License)
</div>

This includes the starter code for the LLM-Merging competition. 

## Setup Environment 

The library was tested on CUDA 10.1 on an A6000. 

```
conda env create -f environment.yml --name llm-merging
conda activate llm-merging 
export PYTHONPATH=`pwd`
```

Authentication tokens are requied for certain models like Llama2 which require user agreeing to some terms. 
The authentication token can be found [here](https://huggingface.co/settings/tokens)

```
export HF_AUTH_TOKEN=""
```

## Developing New Merging Methods 

Do not modify any files other than the new file created and `setup.py`. Doing so can result in the grounds for invalidating your submission. If there is any code in the other files you need to change, feel free to open a pull request to change it. 

1. To add a new merging method, create a new file in `llm_merging/merging`. 

    This file should implement `__init__.py` and `merge.py` functions and extend `llm_merging/merging/Merges`. 
    See `llm_merging/merging/FlanT5Avg.py` or `llm_merging/merging/LlamaAvg.py` for examples.  

2. Modify `setup.py` and add an entry with the merging method in`llm_merging.merging.Merges`. 
   
   For example, the entry `llama_avg = llm_merging.merging.LlamaAvg:LlamaAvg` indicates the method is called `llama_avg` and the file is at `llm_merging/merging/LlamaAvg` 

    Any other libraries required can be specified in `setup.py`

## Test Method 

```
python llm_merging/setup.py install 
python llm_merging/main.py -m {merging_method}
```

The datasets (CosmosQA and BoolQ) are mainly included for ensuring the merging method (with eval on those datasets) run in under the 1 hour time limit. 
Our results on `llama_avg` are `{"cosmos_qa": {"accuracy": 0.234}, "xsum": {"rouge1": 0.123, "rouge2": 0.023, "rougeL": 0.093, "rougeLsum": 0.102}}` , which run in about `25` minutes on our A6000. 


## Submissions


After modifying the file, tar the file in a tarball using the command 

```
tar -cvf llm-merging.tar LLM-Merging
``` 

Attach the tar file in an email and send to `llm.merging@gmail.com` with the subject as `submission: {method_name}` where `{method_name}` is the same method_name used when calling `llm_merging/main.py`.

Note this submission method is only temporary and another automatic submission method should be comming soon. 