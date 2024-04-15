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
        self.data = load_huggingface_dataset(
                *self.dataset, split=self.split
        )

    def _get_templates(self):
        # Only use the original templates 
        all_templates = DatasetTemplates(*self.dataset)
        self._templates = [
            all_templates[template_name]
            for template_name in all_templates.all_template_names
            if all_templates[template_name].metadata.original_task
        ]

    def _preprocess_example(self, example_idx, template_idx):

        example = self.data[example_idx]
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
        self.dataset = ("super_glue", "boolq")
        super().__init__(split, max_examples_per_dataset, round_robin_template)


class TriviaQADataset(Dataset):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        self.dataset = ("mandarjoshi/trivia_qa", "rc.nocontext")
        super().__init__(split, max_examples_per_dataset, round_robin_template)

    def _get_data(self):
        self._data = load_huggingface_dataset(
                *self.dataset, split=self.split
        )

    def _get_templates(self):
        # From FLAN paper 
        self._templates = [
            ("Answer this questions: Q: {question}? A:", "{answer}"),
        ]
        #  self._templates = [
        #     ("Please answer this question: {question}", "{answer}"),
        #     ("{question}", "{answer}"),
        #     ("Write the answer: {question}", "{answer}"),
        #     ("What is the answer: {question}", "{answer}"),
        #     ("Answer this question.\n\n{question}", "{answer}"),
        #     ("Answer the following question. {question}", "{answer}"),
        #     ("Question: {question}\nAnswer:", "{answer}"),
        #     ("{question}???", "{answer}"),
        #     ("Trivia question: {question}\nAnd the answer is?", "{answer}"),
        #     ("{question}\nWhat is the answer?", "{answer}"),
        # ]
    
    def _preprocess_example(self, example_idx, template_idx):

        example = self._examples[example_idx]
        template = self._templates[template_idx]
        
        input = template[0].format(question=example["question"])

        answers = [template[1].format(answer=possible_answer) for possible_answer in example["answer"]["aliases"]]
        
        preprocessed_example = {
            "example_idx": example_idx,
            "question_id": example["question_id"],
            "template_idx": template_idx,
            "input": input,
            "answers": answers,
        }
        preprocessed_example = {
            k: v for k, v in preprocessed_example.items() if v is not None
        }
        return preprocessed_example


# GSM8K_ORIGINAL_COT_TEMPLATE = "Answer the following question step by step"
GSM8K_ORIGINAL_COT_TEMPLATE = """
Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 5 = 6. The answer is 6.\n
Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.\n
Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.\n
Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.\n
Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: Shawn started with 5 toys. If he got two toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.\n
Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 + 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.\n
Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.\n
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n\n"""


class GSM8kDataset(Dataset):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        self.dataset = ("gsm8k", "main")
        super().__init__(split, max_examples_per_dataset, round_robin_template)

    def _get_data(self):
        self._data = load_huggingface_dataset(
                *self.dataset, split=self.split
        )

    def _get_templates(self):
         self._templates = [
            ("{prompt} Q: {question} A: ", "{answer}")
        ]
    
    def _preprocess_example(self, example_idx, template_idx):

        example = self._examples[example_idx]
        template = self._templates[template_idx]
        
        input = template[0].format(prompt=GSM8K_ORIGINAL_COT_TEMPLATE, question=example["question"])
        answer = template[1].format(answer=example["answer"])
        
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



class HumanEvalDataset(Dataset):
    def __init__(
        self,
        split: str,
        max_examples_per_dataset: int,
        round_robin_template=False,
    ):
        self.dataset = ("openai_humaneval", )
        super().__init__(split, max_examples_per_dataset, round_robin_template)

    def _get_data(self):
        self._data = load_huggingface_dataset(
                *self.dataset, split=self.split
        )

    def _get_templates(self):
         self._templates = [
            ("{prompt}", "{canonical_solution}")
        ]
    
    def _preprocess_example(self, example_idx, template_idx):

        example = self._examples[example_idx]
        template = self._templates[template_idx]
        
        input = template[0].format(prompt=example["prompt"])
        answer = template[1].format(canonical_solution=example["canonical_solution"])
        
        preprocessed_example = {
            "example_idx": example_idx,
            "template_idx": template_idx,
            "task_id": example["task_id"],
            "input": input,
            "answer": answer,
        }
        preprocessed_example = {
            k: v for k, v in preprocessed_example.items() if v is not None
        }
        return preprocessed_example
    