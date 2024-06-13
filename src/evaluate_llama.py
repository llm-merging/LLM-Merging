import os
import re
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from utils import seed_all
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

seed_all(42)


def parallel_collate_fn(batch):
    # Just concates to form a big tensor
    input_ids = torch.cat([item[0] for item in batch], dim=0)
    attention_mask = torch.cat([item[1] for item in batch], dim=0)
    labels = [item[2] for item in batch]
    return input_ids, attention_mask, labels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_type", choices=["qa", "math"], required=True)
    parser.add_argument("--split_name", type=str, required=True)
    parser.add_argument("--load_in_8_bit", action="store_true", default=False)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    return args


class CoTdataset(Dataset):

    def __init__(self, data_path, tokenizer: AutoTokenizer, max_len=1024):
        self.data = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.preprocess()

    def preprocess(self):

        for i in tqdm(range(len(self.data)), desc="Tokenizing..."):
            instance = self.data[i]["input"]
            tokenized = self.tokenizer(instance,
                                       return_tensors="pt",
                                       padding="max_length",
                                       max_length=self.max_len,
                                       truncation=True)
            self.data[i]["input_ids"] = tokenized["input_ids"]
            self.data[i]["attention_mask"] = tokenized["attention_mask"]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx][
            "attention_mask"], self.data[idx]["label"]


def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model_args = {
        "pretrained_model_name_or_path":
        f"saved/base_models/{args.model_name}",
        "low_cpu_mem_usage": True,
        "load_in_8bit": args.load_in_8_bit,
    }
    if not args.load_in_8_bit:
        model_args["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(**model_args)
    if not args.load_in_8_bit:
        model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        f"saved/base_models/{args.model_name}", padding_side="left")

    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    gen_kwargs = json.load(open("data/baseline_gen_config.json"))
    # Needed because math explanations are long
    if args.data_type == "math":
        gen_kwargs["max_new_tokens"] = 200
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    gen_kwargs["eos_token_id"] = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    dataset = CoTdataset(
        f"data/{args.dataset_name}/{args.split_name}_prompts.json", tokenizer,
        args.max_len)

    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            collate_fn=parallel_collate_fn)

    new_df = {"decoded": [], "answer": [], "label": []}
    if args.data_type == "qa":
        pattern = r"\((.)\)"
    elif args.dataset_name == "aqua":
        # Aqua has a different format, and the LLM sometimes doesn't generate "(" or ")"
        pattern = r"The answer is \(?(.)\)?"
    elif args.data_type == "math":
        # This is for the math dataset
        pattern = r"The answer is (-?\d+)."

    err_inds = []

    for (ids, attn_mask, labels) in tqdm(dataloader):

        generated = model.generate(input_ids=ids.to(device),
                                   attention_mask=attn_mask.to(device),
                                   **gen_kwargs)
        decoded = tokenizer.batch_decode(generated[:, args.max_len:],
                                         skip_special_tokens=True)

        answers = []
        for i in range(len(decoded)):
            # No need to split for math because the answers themselves cointain "\n", for aqua the case is different
            if args.data_type == "qa" or args.dataset_name == "aqua":
                answers.append(decoded[i].split("\n")[0])
            else:
                answers.append(decoded[i])
            match = re.search(pattern, answers[i])
            if match:
                answers[i] = match.group(1).lower()
                if args.dataset_name == "aqua":
                    # Aqua's pattern also matches "(" and ")" in the answer, need to remove them
                    answers[i] = answers[i].replace("(", "")
                    answers[i] = answers[i].replace(")", "")
                try:
                    # Convert to ints if the data type is math
                    if args.data_type == "math" and not args.dataset_name == "aqua":
                        answers[i] = int(answers[i].replace(",", ""))
                except:
                    err_inds.append(i)
                    answers[i] = -1

        new_df["decoded"].extend(decoded)
        new_df["answer"].extend(answers)
        if args.data_type == "qa":
            new_df["label"].extend(labels)
        elif args.dataset_name == "aqua":
            new_df["label"].extend([x.lower() for x in labels])
        else:
            new_df["label"].extend([int(x) for x in labels])

    new_df = pd.DataFrame(new_df)
    if not os.path.exists(f"out/{args.dataset_name}"):
        os.makedirs(f"out/{args.dataset_name}")
    if not os.path.exists(f"out/{args.dataset_name}/{args.model_name}"):
        os.makedirs(f"out/{args.dataset_name}/{args.model_name}")
    new_df.to_csv(
        f"out/{args.dataset_name}/{args.model_name}/8_bit_{args.load_in_8_bit}_{args.split_name}_predictions.csv",
        index=False)
    acc = (new_df["answer"] == new_df["label"]).mean()
    print(f"Accuracy: {acc}")
    acc = {"Accuracy": acc, "Error Indices": err_inds}
    json.dump(
        acc,
        open(
            f"out/{args.dataset_name}/{args.model_name}/8_bit_{args.load_in_8_bit}_{args.split_name}_metrics.json",
            "w"))


if __name__ == '__main__':

    main()
