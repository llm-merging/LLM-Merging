# -*- coding: utf-8 -*-
""" Official evaluation script for v1.0 of the TriviaQA dataset.
Extended from the evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import sys
import argparse
from string import punctuation


# From https://github.com/mandarjoshi90/triviaqa/blob/master/evaluation/triviaqa_evaluation.py 
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s))))).strip()


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def get_ground_truths(answer):
    return [normalize_answer(ans) for ans in answer]
    # return answer['NormalizedAliases'] + [normalize_answer(ans) for ans in answer.get('HumanAnswers', [])]


def evaluate_triviaqa(ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = 'Missed question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = 'Irrelavant question {} will receive score 0.'.format(qid)
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = get_ground_truths(ground_truth[qid])
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        # if em_for_this_question == 0 and not mute:
        #     print("em=0:", prediction, ground_truths)
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match}

def extract_predicted_trivia_answer(predicted_answer):
    substrings_to_stop = [":", "."]
    for substring in substrings_to_stop:        
        predicted_answer = predicted_answer[:predicted_answer.find(substring)]
    
    return predicted_answer


def exact_match_multiple_references(all_batches):

    answers = {}
    predictions = {}
    for example in all_batches:
        answers[example["example_idx"]] = example["answers"]
        predicted_answer = extract_predicted_trivia_answer(example["predicted_text"])
        predictions[example["example_idx"]] = predicted_answer 
        example["predicted_answer"] = predicted_answer

    return evaluate_triviaqa(answers, predictions), all_batches

def accuracy(all_batches):
    num_correct = 0
    total = 0

    for example in all_batches:

        if example["predicted_choice"] == example["label"]:
            num_correct += 1
    
    total = len(all_batches)

    return {
        "accuracy": num_correct / total
    }, all_batches


 # From https://github.com/archiki/ReCEval/blob/825f840cc4645da7501d0e5a7b7b3b1f868f8dbb/run_flan.py#L76
def extract_gold_number(answer):
    return answer.split("####")[-1].lstrip().replace(
                punctuation, ""
            ).replace(",", "")


# From https://github.com/archiki/ReCEval/blob/825f840cc4645da7501d0e5a7b7b3b1f868f8dbb/run_flan.py#L52
def extract_cot_predicted_number(predicted_txt):
    """
    The number is extracted according to the following order:
    1. Getting the number right after the prefix 'The answer is `
    2. Getting the number right before '\n\n'
    3. Getting the number right after '='

    Afterwards,
    - Commas are removed from the number
    """
    ANSWER_PREFIX = "The answer is "
    split_answerPrefix = predicted_txt.split(ANSWER_PREFIX)
    # Only keep the suffix after the answer_prefix and the suffix is not empty
    if len(split_answerPrefix) > 1 and len(split_answerPrefix[1]) > 0:
        # Keep the suffix after 'The answer is'
        remaining_answer = split_answerPrefix[1]

        # Use the prefix before the .
        split_period = remaining_answer.split(".\n")
        if len(split_period) > 1:
            remaining_answer = split_period[0]

        # Use the last number before the period (in case there is an equal sign)
        number_idx = -1
    else:
        # Only keep the suffix after the last equal sign
        EQUAL_SIGN_PREFIX = "="
        split_equalSignPrefix = predicted_txt.split(EQUAL_SIGN_PREFIX)

        equalSign_suffix = split_equalSignPrefix[-1]
        numbers_afterEqualSign = re.findall( r"\d+(?:[,.]\d+)?", equalSign_suffix)

        if len(split_equalSignPrefix) > 1 and len(numbers_afterEqualSign) > 0:
            remaining_answer = split_equalSignPrefix[-1]

            # Use the first number after the suffix
            number_idx = 0

        else:
            remaining_answer = predicted_txt

            # Use the last number in the text
            number_idx = -1

    # Remove comma in numbers large numbers
    # Find all numbers (including decimals) and return the last number
    numerical_answer = re.findall(r"\d+(?:[,.]\d+)?", remaining_answer)

    if len(numerical_answer) > 0:
        return numerical_answer[number_idx].replace(
                punctuation, ""
            ).replace(",", "")
    else:
        return None

def extract_zero_shot_predicted_number(predicted_txt):
    """
    Extract the first number 
    """
    # For zero-shot predictions, ignore any predictions before \n
    predicted_txt = predicted_txt[:predicted_txt.find("\n")]

    # Find all numbers (including decimals) and return the last number
    numerical_answer = list(filter(lambda x: len(x) > 0, re.findall(r"[\d.,]*(?<![.,])", predicted_txt)))

    print(predicted_txt, numerical_answer)
    if len(numerical_answer) > 0:
        return numerical_answer[-1].replace(
                punctuation, ""
            ).replace(",", "")
    else:
        return None


def numerical_accuracy(all_batches):
    num_correct = 0

    for example in all_batches:

        gold_number = extract_gold_number(example["answer"])
        predicted_number = extract_cot_predicted_number(example["predicted_text"])
        # predicted_number = extract_zero_shot_predicted_number(example["predicted_text"])

        example["gold_number"] = gold_number
        example["predicted_number"] = predicted_number

        if predicted_number is not None:
            is_correct = float(predicted_number) == float(gold_number)
            if is_correct:
                num_correct += 1

    total = len(all_batches)

    return {
        "accuracy": num_correct / total
    }, all_batches

def humaneval_preprocess(all_batches):
    for example in all_batches:
        example["solution"] = example["predicted_text"]
    return {}, all_batches

