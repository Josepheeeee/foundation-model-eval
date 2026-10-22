import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
responses = []

def add(sample, text):
    rows.append(sample)
    responses.append({"sample_id": sample["sample_id"], "text": text})

gsm = [
    ("gsm8k-001", "Natalia sold clips to 48 friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "In April Natalia sold 48 clips.\nIn May she sold half: 24 clips.\nTotal: 48 + 24 = 72.\n#### 72", "#### 72"),
    ("gsm8k-002", "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
     "50 minutes is 50/60 of an hour.\n12 * 50 / 60 = 10.\n#### 10", "#### 10"),
    ("gsm8k-003", "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
     "Betty has 50. Parents give 15, grandparents give 30.\nTotal 95, so she needs 5.\n#### 5", "#### 5"),
    ("gsm8k-004", "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
     "She uses 3 + 4 = 7 eggs.\nRemainder 16 - 7 = 9 eggs.\n9 * 2 = 18.\n#### 18", "#### 18"),
    ("gsm8k-005", "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
     "White is half of 2, which is 1.\nTotal 2 + 1 = 3.\n#### 3", "#### 3"),
    ("gsm8k-006", "Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?",
     "Charleston has 4 * 20 = 80.\nToulouse has 2 * 80 = 160.\nTotal 20 + 80 + 160 = 260.\n#### 260", "#### 260"),
    ("gsm8k-007", "Leah has 32 plums and 42 peaches. She gives one fourth of the plums and one third of the peaches to her brother. How many pieces of fruit does she have left?",
     "Plums left: 32 * 3/4 = 24.\nPeaches left: 42 * 2/3 = 28.\nTotal 24 + 28 = 52.\n#### 52", "#### 52"),
    ("gsm8k-008", "A factory produces 1200 widgets in 8 hours. At the same rate, how many widgets does it produce in 5 hours?",
     "Rate 1200 / 8 = 150 per hour.\n150 * 5 = 750.\n#### 750", "#### 750"),
]

for sample_id, question, answer, final in gsm:
    add({"sample_id": sample_id, "task": "gsm8k", "split": "train", "prompt": question, "answer": answer,
         "metadata": {"source": "hand-written demo fixture"}},
        answer)

# A few non-math fixtures are scored by the same code path in tests.
add({"sample_id": "mmlu-001", "task": "mmlu_pro", "split": "test",
     "prompt": "Which planet is known as the Red Planet?\nA. Venus\nB. Mars\nC. Jupiter\nD. Saturn",
     "answer": "B", "metadata": {"options": 4}}, "B")
add({"sample_id": "mmlu-002", "task": "mmlu_pro", "split": "test",
     "prompt": "What is 7 * 8?\nA. 54\nB. 56\nC. 58\nD. 64", "answer": "B",
     "metadata": {"options": 4}}, "A")
add({"sample_id": "mmlu-003", "task": "mmlu_pro", "split": "test",
     "prompt": "Which gas do plants absorb?\nA. Oxygen\nB. Nitrogen\nC. Carbon dioxide\nD. Helium",
     "answer": "C", "metadata": {"options": 4}}, "C")

add({"sample_id": "bbh-001", "task": "bbh", "split": "test",
     "prompt": "If all cats are animals and some animals are black, can we conclude some cats are black? Answer yes or no.",
     "answer": "no"}, "ANSWER: no")
add({"sample_id": "bbh-002", "task": "bbh", "split": "test",
     "prompt": "Choose the option that is a synonym of 'rapid'.\nA. slow\nB. quick\nC. heavy\nD. quiet",
     "answer": "B"}, "ANSWER: B")
add({"sample_id": "bbh-003", "task": "bbh", "split": "test",
     "prompt": "What is the next number: 2, 4, 8, 16, ?", "answer": "32"}, "ANSWER: 31")

add({"sample_id": "ifeval-001", "task": "ifeval", "split": "test",
     "prompt": "Write exactly two sentences about the sea. Use the word 'blue' at least once.",
     "answer": "pass", "metadata": {"instruction_id_list": ["length_constraints:number_sentences", "keywords:existence"],
                                    "kwargs": [{"relation": "exactly", "num_sentences": 2}, {"keywords": ["blue"]}]}},
    "The sea is blue. It moves with the wind.")
add({"sample_id": "ifeval-002", "task": "ifeval", "split": "test",
     "prompt": "Reply with a valid JSON object containing the key 'ok'.",
     "answer": "pass", "metadata": {"instruction_id_list": ["detectable_format:json_format"], "kwargs": [{}]}},
    '{"ok": true}')
add({"sample_id": "ifeval-003", "task": "ifeval", "split": "test",
     "prompt": "Write more than ten words about rain and do not use any comma.",
     "answer": "pass", "metadata": {"instruction_id_list": ["length_constraints:number_words", "punctuation:no_comma"],
                                    "kwargs": [{"relation": "more than", "num_words": 10}, {}]}},
    "Rain falls softly on the roof and runs down the glass while the street shines.")
add({"sample_id": "ifeval-004", "task": "ifeval", "split": "test",
     "prompt": "Use an unsupported synthetic instruction.",
     "answer": "pass", "metadata": {"instruction_id_list": ["synthetic:unknown_rule"], "kwargs": [{}]}},
    "anything")

add({"sample_id": "humaneval-001", "task": "humaneval_plus", "split": "test",
     "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n", "answer": "code",
     "metadata": {"entry_point": "add_one", "test_code": "def check(candidate):\n    assert candidate(1) == 2\n    assert candidate(-1) == 0\n"}},
    "```python\ndef add_one(x):\n    return x + 1\n```")
add({"sample_id": "humaneval-002", "task": "humaneval_plus", "split": "test",
     "prompt": "def add_one(x):\n    \"\"\"Return x + 1.\"\"\"\n", "answer": "code",
     "metadata": {"entry_point": "add_one", "test_code": "def check(candidate):\n    assert candidate(1) == 2\n"}},
    "```python\ndef add_one(x):\n    return x - 1\n```")

add({"sample_id": "ruler-001", "task": "ruler", "split": "diagnostic",
     "prompt": "The magic number is 7391.\nQuestion: What is the magic number?", "answer": "7391",
     "metadata": {"target_length": 4096, "task": "niah_single_1"}}, "7391")
add({"sample_id": "ruler-002", "task": "ruler", "split": "diagnostic",
     "prompt": "The magic number is 7391.\nQuestion: What is the magic number?", "answer": "7391",
     "metadata": {"target_length": 4096, "task": "niah_single_1"}}, "The answer is 7000.")

root.joinpath("data/fixtures/demo_samples.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
root.joinpath("data/fixtures/responses.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in responses), encoding="utf-8")
print(len(rows), len(responses))
