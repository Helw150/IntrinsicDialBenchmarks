import numpy as np
import json, os, re
import argparse
from tqdm import tqdm

MODELS = [
    "gemini-pro",
    "anthropic.claude-v2:1",
    "anthropic.claude-instant-v1",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1B",
    "EleutherAI/pythia-1.4B",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9B",
    "EleutherAI/pythia-12B",
    "huggyllama/llama-7b",
    "huggyllama/llama-13b",
    "huggyllama/llama-30b",
    "huggyllama/llama-65b",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "tiiuae/falcon-7B",
    "tiiuae/falcon-40b",
    "togethercomputer/RedPajama-INCITE-7B-Base",
    "togethercomputer/RedPajama-INCITE-Base-3B-v1",
    "mistralai/Mistral-7B-v0.1",
    "mosaicml/mpt-7b",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-72B",
    "01-ai/Yi-6B",
    "01-ai/Yi-34B",
]


# Bootstrap
# Repeat R times: randomly create new samples from the data with repetitions, calculate delta(A,B).
# let r be the number of times that delta(A,B)<2*orig_delta(A,B). significance level: r/R
# This implementation follows the description in Berg-Kirkpatrick et al. (2012),
# "An Empirical Investigation of Statistical Significance in NLP".
def Bootstrap(data_A, data_B, R=10000, alpha=0.05):
    n = len(data_A)
    R = max(R, int(len(data_A) * (1 / float(alpha))))
    delta_orig = float(sum([x - y for x, y in zip(data_A, data_B)])) / n
    r = 0
    for x in range(0, R):
        temp_A = []
        temp_B = []
        samples = np.random.randint(
            0, n, n
        )  # which samples to add to the subsample with repetitions
        for samp in samples:
            temp_A.append(data_A[samp])
            temp_B.append(data_B[samp])
        delta = float(sum([x - y for x, y in zip(temp_A, temp_B)])) / n
        if delta > 2 * delta_orig:
            r = r + 1
    pval = float(r) / (R)
    return pval


def split_corr(pred_1, gold_1, pred_2, gold_2):
    model_1_pred = np.array(pred_1)
    model_1_gold = np.array(gold_1)
    corr_1 = (model_1_pred == model_1_gold).astype(int)

    model_2_pred = np.array(pred_2)
    model_2_gold = np.array(gold_2)
    corr_2 = (model_2_pred == model_2_gold).astype(int)

    print(
        "Model 1 and Model 2 mean accuracy. Model 1: {} Model 2 {}".format(
            corr_1.mean(), corr_2.mean()
        )
    )
    return corr_1, corr_2


def main(args):
    with open(
        f"./predictions/{args.model.replace('/', '-')}_predictions.json", "r"
    ) as json_file:
        dial_mcqs = [json.loads(jline) for jline in json_file.readlines()]
    dial_pred = [mcq["model_prediction"] for mcq in dial_mcqs]
    dial_gold = [mcq["correct_answer"] for mcq in dial_mcqs]

    with open(
        f"./baseline_predictions/{args.model.replace('/', '-')}_predictions.json", "r"
    ) as json_file:
        gen_mcqs = [json.loads(jline) for jline in json_file.readlines()]
    gen_pred = [mcq["model_prediction"] for mcq in gen_mcqs]
    gen_gold = [mcq["correct_answer"] for mcq in gen_mcqs]
    print(args.model)
    corrs_1, corrs_2 = split_corr(dial_pred, dial_gold, gen_pred, gen_gold)

    try:
        sig = Bootstrap(corrs_2, corrs_1)

        if args.verbose:
            print(
                "The p-value for General English knowledge being better than Indian English Knowledge in {} is {}".format(
                    args.model, sig
                )
            )
            print("--------------")
        return {"Significance": sig}
    except:
        return {"Significance": None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="eval_significance")
    parser.add_argument(
        "--model", "-m", type=str, default="all", choices=["all"] + MODELS
    )

    args = parser.parse_args()
    args.verbose = True
    results = {}
    if args.model == "all":
        for model in MODELS:
            args.model = model
            r = main(args)
            results[args.model] = r
    else:
        r = main(args)
        results[args.model] = r
    with open("sig.json", "w") as outfile:
        json.dump(results, outfile)
