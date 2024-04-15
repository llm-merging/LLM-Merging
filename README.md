<div align="center">


<h1>LLM-Merging: Building LLMs Efficiently through Merging </h1>

[![](https://img.shields.io/badge/Documentation-online-green)](https://llm-merging.readthedocs.io)
[![](https://img.shields.io/badge/Website-online-green)](https://llm-merging.github.io)
[![](https://img.shields.io/badge/License-MIT-blue)](#License)
</div>

This includes the starter code for the LLM-Merging competition. 

## Setup Environment 

```
conda env create -f environment.yml --name llm-merging
conda activate llm-merging 
```

## Developing New Merging Methods 

To add a new merging method, create a new file in `llm_merging/merging`. 

This file should the implement `__init__.py` and `merge.py` functions and extend `llm_merging/merging/BaseMerging`. 

See `llm_merging/merging/FlanT5Avg.py` or `llm_merging/merging/LlamaAvg.py` for examples.  

