import json
import random

import pandas as pd
from datasets import load_dataset
from src.Dialects import DialectFromFeatureList, DialectFromVector, IndianDialect
from tqdm import tqdm

random.seed(42)

ie = IndianDialect()
rev_ie = DialectFromVector(vector=(ie.vector + 1) * (ie.vector == 0))
features = [key for key, value in rev_ie.morphosyntax_transforms.items()]

transformations = [
    DialectFromFeatureList(feature_list=[feature])
    for feature in features
    if feature
    not in [
        "a_ing",
        "indefinite_for_zero",
        "adj_postfix",
        "existential_it",
        "you_ye",
        "do_tense_marker",
    ]
]

df = load_dataset("WillHeld/demszky_pairs")["train"].to_pandas()
df = pd.merge(
    df[df["feature_present"] == 0],
    df[df["feature_present"] == 1],
    left_on="phrase_ID",
    right_on="phrase_ID",
)
df = df.groupby("phrase_ID").sample(1)
gen_e = df["sentence_x"].array.tolist()
ind_e = df["sentence_y"].array.tolist()
invents = [
    [
        (transform.transform(ex), transform.morphosyntax_transforms)
        for transform in transformations
    ]
    for ex in tqdm(gen_e)
]

invents = [
    [(ex, name) for ex, name in invent if ex != gen_e[i]]
    for i, invent in enumerate(invents)
]

invents = [random.sample(invent, 1)[0][0] for invent in invents]

gen_e_jsons = []
for ex in gen_e:
    letter = random.sample(["A", "B"], 1)[0]
    prompt = f"Sentence: '{ex}'\nWhich of the following best describes the acceptability of the grammar of this sentence in the Indian English dialect?\n\n"
    if letter == "A":
        prompt += "A: Acceptable\nB: Unacceptable"
    elif letter == "B":
        prompt += "A: Unacceptable\nB: Acceptable"
        prompt += "\n\n Answer: "
        gen_e_jsons.append({"prompt": prompt, "correct_answer": letter})

ind_e_jsons = []
for ex in ind_e:
    letter = random.sample(["A", "B"], 1)[0]
    prompt = f"Sentence: '{ex}'\nWhich of the following best describes the acceptability of the grammar of this sentence in the Indian English dialect?\n\n"
    if letter == "A":
        prompt += "A: Acceptable\nB: Unacceptable"
    elif letter == "B":
        prompt += "A: Unacceptable\nB: Acceptable"
        prompt += "\n\n Answer: "
        ind_e_jsons.append({"prompt": prompt, "correct_answer": letter})

invent_jsons = []
for ex in invents:
    letter = random.sample(["A", "B"], 1)[0]
    prompt = f"Sentence: '{ex}'\nWhich of the following best describes the acceptability of the grammar of this sentence in the Indian English dialect?\n\n"
    if letter == "A":
        prompt += "A: Acceptable\nB: Unacceptable"
    elif letter == "B":
        prompt += "A: Unacceptable\nB: Acceptable"
        prompt += "\n\n Answer: "
        invent_jsons.append({"prompt": prompt, "correct_answer": letter})

with open("demszky_indian_english_syntax_quiz.json", "w") as json_file:
    for entry in ind_e_jsons:
        json_file.write(json.dumps(entry) + "\n")

with open("demszky_general_english_syntax_quiz.json", "w") as json_file:
    for entry in gen_e_jsons:
        json_file.write(json.dumps(entry) + "\n")

with open("demszky_invented_english_syntax_quiz.json", "w") as json_file:
    for entry in invent_jsons:
        json_file.write(json.dumps(entry) + "\n")
