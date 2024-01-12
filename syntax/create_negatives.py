import json
import random

import string
import pandas as pd
from datasets import load_dataset
from src.Dialects import DialectFromFeatureList, DialectFromVector, IndianDialect
from tqdm import tqdm

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
df = df[df["sentence_x"] != df["sentence_y"]]
df = df.groupby("phrase_ID").sample(1, random_state=42)
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

random.seed(42)

<<<<<<< HEAD
triplets = []
for i, ex in enumerate(gen_e):
    ind_ex = ind_e[i]
    inv_ex = invents[i]
    triplets.append({"base": ex, "demszky": ind_ex, "invent": inv_ex})

with open("inde_blimp.json", "w") as json_file:
    for entry in triplets:
        json_file.write(json.dumps(entry) + "\n")
