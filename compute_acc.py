import argparse
import os
import torch
import numpy as np
import json
from tqdm import tqdm
import time


def main(args):
    with open(args.input, "r") as json_file:
        mcqs = [json.loads(jline) for jline in json_file.readlines()]
    corr = 0
    for mcq in tqdm(mcqs):
        if mcq["model_prediction"] == mcq["correct_answer"]:
            corr += 1

    print(corr / float(len(mcqs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./predictions/meta-llama-Llama-2-7b-hf_predictions.json",
    )
    args = parser.parse_args()
    main(args)
