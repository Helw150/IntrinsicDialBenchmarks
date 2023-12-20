import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
from torch.cuda.amp import autocast

import anthropic_bedrock
from anthropic_bedrock import AnthropicBedrock

import vertexai
from vertexai.preview.generative_models import (
    GenerativeModel,
    ChatSession,
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)


def gemini_eval(args, model, prompt):
    time.sleep(1)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    # Set parameters to reduce variability in responses
    generation_config = GenerationConfig(
        temperature=0,
        top_k=1,
        max_output_tokens=1,
    )
    responses = model.generate_content(
        contents=[prompt],
        generation_config=generation_config,
        safety_settings=safety_settings,
    )
    if len(responses.candidates) == 0:
        return "None"
    if not responses.candidates[0].text.strip() in ["A", "B", "C", "D"]:
        print("EXCEPT")
    return responses.candidates[0].text.strip()


def claude_eval(args, client, prompt):
    completion = client.completions.create(
        model="anthropic.claude-v2:1",
        max_tokens_to_sample=1,
        temperature=0,
        prompt=f"{anthropic_bedrock.HUMAN_PROMPT}"
        + prompt.replace("Answer:", f"{anthropic_bedrock.AI_PROMPT} Answer:"),
    )

    print(completion.completion.strip())
    if not completion.completion.strip() in ["A", "B", "C", "D"]:
        print("EXCEPT")
    return completion.completion.strip()


@torch.no_grad()
def eval(args, model, tokenizer, prompt):
    input_ids = tokenizer(prompt[:-1], return_tensors="pt").input_ids.cuda()
    # print(tokenizer.decode(model.generate(input_ids, max_new_tokens=20)[0, -20:]))
    logits = model(input_ids=input_ids).logits[:, -1, :].flatten()

    probs = (
        torch.nn.functional.softmax(
            torch.tensor(
                [
                    logits[tokenizer(" A", add_special_tokens=False).input_ids[-1]],
                    logits[tokenizer(" B", add_special_tokens=False).input_ids[-1]],
                    logits[tokenizer(" C", add_special_tokens=False).input_ids[-1]],
                    logits[tokenizer(" D", add_special_tokens=False).input_ids[-1]],
                ]
            ),
            dim=0,
        )
        .detach()
        .cpu()
        .type(torch.float16)
        .numpy()
    )
    pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]

    return pred


def main(args):
    # torch.set_default_device('cuda')
    # model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model = None
    if "gemini" in args.model:
        project_id = os.environ["GCLOUD_PROJ"]
        location = "us-central1"
        vertexai.init(project=project_id, location=location)
        client = GenerativeModel("gemini-pro")
    elif "claude" in args.model:
        client = AnthropicBedrock(
            aws_access_key=os.environ["AWS_KEY"],
            aws_secret_key=os.environ["AWS_SECRET"],
            aws_region="us-west-2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open("wiktionary_indian_english_lexicon_quiz.json", "r") as json_file:
        mcqs = [json.loads(jline) for jline in json_file.readlines()]
    corr = 0
    for mcq in tqdm(mcqs):
        if "gemini" in args.model:
            pred = gemini_eval(args, client, mcq["prompt"])
        elif "claude" in args.model:
            pred = claude_eval(args, client, mcq["prompt"])
        else:
            pred = eval(args, model, tokenizer, mcq["prompt"])
        if pred == mcq["correct_answer"]:
            corr += 1
        mcq["model_prediction"] = pred

    with open(
        f"predictions/{args.model.replace('/', '-')}_predictions.json", "w"
    ) as json_file:
        for mcq in mcqs:
            json_file.write(json.dumps(mcq) + "\n")
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
