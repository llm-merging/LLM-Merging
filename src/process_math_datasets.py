import argparse
from datasets import load_dataset

from tqdm import tqdm
from utils import *
import re


def process_gsm8k(row):
    pattern = r"#### (-?\d+)"
    match = re.search(pattern, row["answer"])
    int_answer = None
    if match:
        int_answer = match.group(1)
    else:
        print("Exception")
    return {
        "question": row["question"],
        "label": int_answer,
        "label_text": row["answer"]
    }

def process_boolq(row):
    return {
        'question': row['question'],
        'label': row['answer'],
        'label_text': row["passage"],
    }

def process_mawps(row):
    return {
        'id': row['id'],
        'question': row['question'],
        'chain': row["chain"],
        'result': row["result"],
        'label': row["result_float"],
        'label_text': row["equation"],
        'expression': row["expression"],
    }


def process_svamp(row):
    return {
        'id': row['ID'],
        'question': row['Body'] + row['Question'],
        'label': row['Answer'],
        'label_text': row["Type"] + " ," + row["Equation"],
    }


def process_aqua(row):
    full_question = row['question'] + ' ' + ' '.join(row['options'])
    return {
        'question': full_question,
        'label': row['correct'],
        'label_text': row['rationale'],
    }


def main(args):
    if args.dataset == 'gsm8k':
        raw_dataset = load_dataset('gsm8k', 'main')
        demonstration_file = 'prompts/math/cot_prompt.txt'
        splits = ['train', 'test']
        process_fn = process_gsm8k

    elif args.dataset == 'svamp':
        raw_dataset = load_dataset('ChilleD/SVAMP')
        demonstration_file = 'prompts/math/cot_prompt.txt'
        splits = ['train', 'test']
        process_fn = process_svamp

    elif args.dataset == 'aqua':
        raw_dataset = load_dataset('aqua_rat')
        demonstration_file = 'prompts/aqua/cot_prompt.txt'
        splits = ['train', 'validation', 'test']
        process_fn = process_aqua

    elif args.dataset == 'boolq':
        raw_dataset = load_dataset('google/boolq')
        demonstration_file = 'prompts/boolq/cot_prompt.txt'
        splits = ['train', 'validation']
        process_fn = process_boolq

    elif args.dataset == 'mawps':
        raw_dataset = load_dataset('MU-NLPC/Calc-mawps', "original-splits")
        demonstration_file = 'prompts/mawps/cot_prompt.txt'
        splits = ['train', 'validation', 'test']
        process_fn = process_mawps

    # read the demonstration
    # with open(demonstration_file) as f:
    #     demonstration = f.read()

    if args.combine:
        combined = []
    # process the dataset by prepending the demonstration to create prompt
    for split in splits:
        prompt_data = []
        for row in tqdm(raw_dataset[split],
                        desc=f'Processing {args.dataset} {split}...'):
            item = process_fn(row)
            # prompt = demonstration + f"\nQ: {item['question']} \nA: "
            prompt = f"\nQ: {item['question']} \nA: "
            if args.dataset == 'mawps':
                prompt_data.append({
                'id': item['id'],
                'input': prompt,
                'dataset': args.dataset,
                'label': item['label'],
                'label_text': item['label_text'],
                'expression': item['expression'],
                'chain': item['chain'],
                'result': item['result'],
                
                })
                if args.combine:
                    combined.append({
                        'id': item['id'],
                        'input': prompt,
                        'dataset': args.dataset,
                        'label': item['label'],
                        'label_text': item['label_text'],
                        'expression': item['expression'],
                        'chain': item['chain'],
                        'result': item['result'],
                    })
            else:
                prompt_data.append({
                    'input': prompt,
                    'dataset': args.dataset,
                    'label': item['label'],
                    'label_text': item['label_text'],
                })
                if args.combine:
                    combined.append({
                        'input': prompt,
                        'dataset': args.dataset,
                        'label': item['label'],
                        'label_text': item['label_text'],
                    })




        write_json(
            f'./data/{args.dataset}/{split}_prompts.json',
            prompt_data,
            mode='w',
            make_dir=True)
    if args.combine:
        write_json(
            f'./data/{args.dataset}/combined_prompts.json',
            combined,
            mode='w',
            make_dir=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MATH datasets')
    parser.add_argument('--dataset', choices=['gsm8k', 'svamp', 'aqua','boolq','mawps'])
    parser.add_argument('--combine', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
