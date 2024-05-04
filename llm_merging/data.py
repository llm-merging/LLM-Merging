import torch 

from typing import List, Tuple, Dict, Callable, Any

from jinja2 import Template
from promptsource.templates import DatasetTemplates
from datasets import load_dataset as load_huggingface_dataset

# Code from https://github.com/r-three/phatgoose

def find_label(target: str, answer_choices: List[str]) -> int:
    """

    Args:
        target: 
        answer_choices: 

    Returns:
        
    """
    for idx, choice in enumerate(answer_choices):
        if choice == target:
            return idx
    
    return -1


class Dataset(object):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        
        assert split in ["train", "validation", "test"]

        self.split = split
        self.max_examples_per_dataset = max_examples_per_dataset
        self.round_robin_template = round_robin_template

        self._get_data()
        self._get_templates()
        self.process_examples()

    def process_examples(self):
        if self.max_examples_per_dataset is not None:
            self._examples = []
            for idx, example in enumerate(self._data):
                if idx < self.max_examples_per_dataset: 
                    self._examples.append(example)
                else:
                    break
            assert len(self._examples) == self.max_examples_per_dataset
        else:
            self._examples = [example for example in self._data]

    def _get_data(self):
        raise NotImplementedError

    def _get_templates(self):
        raise NotImplementedError
    
    def __len__(self):
        if self.round_robin_template:
            return len(self._examples)
        else:
            return len(self._examples) * len(self._templates)

    def __getitem__(self, idx):
        if self.round_robin_template:
            example_idx = idx
        else:
            example_idx = idx // len(self._templates)
        
        template_idx = idx % len(self._templates)

        return self._preprocess_example(example_idx, template_idx)


class P3Dataset(Dataset):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        super().__init__(split, max_examples_per_dataset, round_robin_template)

    def _get_data(self):
        self._data = load_huggingface_dataset(
                *self.dataset_name, split=self.split
        )

    def _get_templates(self):
        # Only use the original templates 
        all_templates = DatasetTemplates(*self.dataset_name)
        self._templates = [
            all_templates[template_name]
            for template_name in all_templates.all_template_names
            if all_templates[template_name].metadata.original_task
        ]

    def _preprocess_example(self, example_idx, template_idx):

        example = self._data[example_idx]
        template = self._templates[template_idx]

        try:
            inputs_and_targets = template.apply(example)
        except:
            raise ValueError(f"Error in applying template {template_idx} to example {example_idx}")


        if len(inputs_and_targets) == 2:
            input, target = template.apply(example)
            if target == "":
                raise ValueError("target is empty")
        else:
            raise ValueError(f"Length of input and target is {len(inputs_and_targets)}")
        answer_choices = template.get_answer_choices_list(example)

        if answer_choices is None:
            answer_choices = []

        label = find_label(target, answer_choices)
        
        tokenized_example = {
            "example_idx": example_idx,
            "template_idx": template_idx,
            "input": input,
            "target": target,
            "answer_choices": answer_choices,
            "label": label,
        }
        tokenized_example = {
            k: v for k, v in tokenized_example.items() if v is not None
        }
        return tokenized_example
    
class BoolQDataset(P3Dataset):
    def __init__(self,
                 split: str,
                max_examples_per_dataset: int,
                round_robin_template=False,):
        # Dataset attributes
        self.dataset_name = ("super_glue", "boolq")
        super().__init__(split, max_examples_per_dataset, round_robin_template)

class MAWPSDataset(Dataset):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        self.dataset = ("MU-NLPC/Calc-mawps", )
        super().__init__(split, max_examples_per_dataset, round_robin_template)

    def _get_data(self):
        self._data = load_huggingface_dataset(
                *self.dataset, split=self.split
        )

    def _get_templates(self):
         self._templates = [
            ("{question}", "{result}")
        ]
    
    def _preprocess_example(self, example_idx, template_idx):

        example = self._examples[example_idx]
        template = self._templates[template_idx]

        input = template[0].format(question=example["question"])
        answer = template[1].format(result=example["result"])
        
        preprocessed_example = {
            "example_idx": example_idx,
            "template_idx": template_idx,
            "input": input,
            "answer": answer,
        }
        preprocessed_example = {
            k: v for k, v in preprocessed_example.items() if v is not None
        }
        return preprocessed_example
    