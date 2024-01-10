import argparse
import json
import os
import time

import string
import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def openai_eval(args, client, tokenizer, prompt):
    time.sleep(1)
    api_query = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        logit_bias={tokenizer.encode(let)[0]: 20 for let in ["A", "B"]},
        temperature=0,
        max_tokens=1,
        user="RESEARCH-DATASET-DialSynt",
    )
    response = api_query.choices[0].message.content
    print(response.strip())
    return response.strip()


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
    if not responses.candidates[0].text.strip() in ["A", "B"]:
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
    if not completion.completion.strip() in ["A", "B"]:
        print("EXCEPT")
    return completion.completion.strip()


@torch.no_grad()
def eval(args, model, tokenizer, prompts):
    if prompts[-1] not in string.punctuation:
        prompts = prompts + "."
    prompts = "The following is an example of acceptable Indian English: '" + prompts + "'"
    input_ids = tokenizer(prompts, padding=False, return_tensors="pt").input_ids.cuda()
    logits = model(input_ids=input_ids).logits
    
    loss = torch.nn.CrossEntropyLoss(reduction="sum")
    nll = loss(logits[0, :-1, :], input_ids[0, 1:])
    return nll.cpu().numpy().item()


def main(args):
    model = None
    if "gpt" in args.model:
        import tiktoken
        from openai import OpenAI

        client = OpenAI()
        tokenizer = tiktoken.encoding_for_model(args.model)
    elif "gemini" in args.model:
        import vertexai
        from vertexai.preview.generative_models import (
            ChatSession,
            GenerationConfig,
            GenerativeModel,
            HarmBlockThreshold,
            HarmCategory,
        )

        project_id = os.environ["GCLOUD_PROJ"]
        location = "us-central1"
        vertexai.init(project=project_id, location=location)
        client = GenerativeModel("gemini-pro")
    elif "claude" in args.model:
        import anthropic_bedrock
        from anthropic_bedrock import AnthropicBedrock

        client = AnthropicBedrock(
            aws_access_key=os.environ["AWS_KEY"],
            aws_secret_key=os.environ["AWS_SECRET"],
            aws_region="us-west-2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", trust_remote_code=True
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    with open("inde_blimp.json", "r") as json_file:
        triplets = [{**json.loads(jline), **{"type": "Gen"}} for jline in json_file.readlines()]
    corrs = {"Gen": 0, "Ind": 0}
    for i, triplet in tqdm(enumerate(triplets)):
        if "gpt" in args.model:
            pred = openai_eval(args, client, tokenizer, mcq["prompt"])
        elif "gemini" in args.model:
            pred = gemini_eval(args, client, mcq["prompt"])
        elif "claude" in args.model:
            pred = claude_eval(args, client, mcq["prompt"])
        else:
            prompts = [triplet[key] for key in triplet]
            triplet["gen_nll"] = eval(args, model, tokenizer, prompts[0])
            triplet["ind_nll"] = eval(args, model, tokenizer, prompts[1])
            triplet["inv_nll"] = eval(args, model, tokenizer, prompts[2])
            triplet["gen_corr"] = triplet["gen_nll"] < triplet["inv_nll"]
            triplet["ind_corr"] = triplet["ind_nll"] < triplet["inv_nll"]
            if triplet["gen_corr"]:
                corrs["Gen"] += 1
            if triplet["ind_corr"]:
                corrs["Ind"] += 1

    for key in corrs.keys():
        print(key)
        print(corrs[key] / len(triplets))
    with open(
        f"predictions/{args.model.replace('/', '-')}_predictions.json", "w"
    ) as json_file:
        for triplet in triplets:
            json_file.write(json.dumps(triplet) + "\n")


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
