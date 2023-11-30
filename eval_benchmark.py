import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


@torch.no_grad()
def eval(args, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    logits = model(input_ids=input_ids).logits[:, -1, :].flatten()

    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer("A").input_ids[0]],
                    logits[tokenizer("B").input_ids[0]],
                    logits[tokenizer("C").input_ids[0]],
                    logits[tokenizer("D").input_ids[0]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    return pred


def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with open("wiktionary_indian_english_lexicon_quiz.json", "r") as json_file:
        mcqs = [json.loads(jline) for jline in json_file.readlines()]
    corr = 0
    for mcq in tqdm(mcqs):
        pred = eval(args, model, tokenizer, mcq["prompt"])
        if pred == mcq["correct_answer"]:
            corr += 1

    print(corr / float(len(mcqs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
    )
    args = parser.parse_args()
    main(args)
