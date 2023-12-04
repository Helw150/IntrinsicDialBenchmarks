import json
import random

random.seed(42)


def validate_dial(sense):
    if "glosses" in sense:
        for gloss in sense["glosses"]:
            if "plural" in gloss:
                return False

    if "categories" not in sense:
        return True
    return False


def assign_letters(options):
    options[0] = "A: " + options[0]
    options[1] = "B: " + options[1]
    options[2] = "C: " + options[2]
    options[3] = "D: " + options[3]
    return options


with open("kaikki.org-dictionary-English.json", "r") as json_file:
    raw_dict = [json.loads(jline) for jline in json_file.readlines()]
    raw_dict = random.sample(raw_dict, 1041)

processed_dict = []
answers = {"noun": [], "verb": [], "adjective": []}
for entry in raw_dict:
    word = entry["word"]
    pos = entry["pos"]
    pos = pos if pos != "adj" else "adjective"
    raw_glosses = [
        gloss
        for sense in entry["senses"]
        if "glosses" in sense
        for gloss in sense["glosses"]
        if validate_dial(sense)
    ]
    if raw_glosses and pos in ["noun", "verb", "adjective"]:
        processed_dict.append(
            {
                "pos": pos,
                "term": word,
                "correct_definition": raw_glosses[0],
            }
        )
        answers[pos].append(raw_glosses[0])

for entry in processed_dict:
    alternates = random.sample(answers[entry["pos"]], 4)
    true_alternates = [
        alternate
        for alternate in alternates
        if entry["correct_definition"] != alternate
    ][:3]
    options = [entry["correct_definition"]] + true_alternates
    random.shuffle(options)
    letter_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct_index = letter_map[options.index(entry["correct_definition"])]
    letters = assign_letters(options)
    entry["correct_answer"] = correct_index
    entry["options"] = letters
    string_options = "\n".join(letters)
    entry[
        "prompt"
    ] = f"Which of the following could \"{entry['term']}\" mean in Indian English when used as a {entry['pos']}?\n\n{string_options}\n\nAnswer: "

with open("wiktionary_unmarked_english_lexicon_quiz.json", "w") as json_file:
    for entry in processed_dict:
        json_file.write(json.dumps(entry) + "\n")
