#!/usr/bin/env python3
"""Download and normalize the public evaluation tasks on the Linux data host.

The script is intentionally explicit about revisions: it writes the resolved
dataset fingerprint next to every file, and refuses to pretend that `main` is a
fixed revision. It does not run any model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"


def fingerprint(dataset) -> str:
    value = getattr(dataset, "_fingerprint", None) or str(dataset.info)
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_manifest(path: Path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def gsm8k():
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")
    write_jsonl(RAW / "gsm8k_train.jsonl", [{"id": str(i), "question": row["question"], "answer": row["answer"]}
                                            for i, row in enumerate(train)])
    write_jsonl(RAW / "gsm8k_test.jsonl", [{"id": str(i), "question": row["question"], "answer": row["answer"]}
                                           for i, row in enumerate(test)])
    write_manifest(RAW / "gsm8k.json", train=len(train), test=len(test), train_fingerprint=fingerprint(train),
                   test_fingerprint=fingerprint(test))


def svamp():
    ds = load_dataset("ChilleD/SVAMP", split="test")
    rows = []
    for i, row in enumerate(ds):
        answer = row.get("Answer") or row.get("answer") or row.get("final_answer")
        question = row.get("Question") or row.get("question") or row.get("Body")
        rows.append({"sample_id": f"svamp-{i:04d}", "task": "svamp", "split": "test",
                     "prompt": str(question), "answer": str(answer)})
    write_jsonl(PROCESSED / "svamp/test.jsonl", rows)
    write_manifest(PROCESSED / "svamp.json", n=len(rows), fingerprint=fingerprint(ds))


def mmlu_pro():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    for i, row in enumerate(ds):
        options = row.get("options") or []
        letters = "ABCDEFGHIJ"
        prompt = row["question"] + "\n" + "\n".join(f"{letters[j]}. {opt}" for j, opt in enumerate(options))
        answer = row.get("answer") or letters[int(row.get("answer_index", 0))]
        rows.append({"sample_id": f"mmlu-pro-{i:05d}", "task": "mmlu_pro", "split": "test",
                     "prompt": prompt, "answer": str(answer)[0].upper(),
                     "metadata": {"options": len(options), "category": row.get("category")}})
    write_jsonl(PROCESSED / "mmlu_pro/test.jsonl", rows)
    write_manifest(PROCESSED / "mmlu_pro.json", n=len(rows), fingerprint=fingerprint(ds))


def bbh():
    rows = []
    for config in get_dataset_config_names("lukaemon/bbh"):
        try:
            ds = load_dataset("lukaemon/bbh", config, split="test")
        except Exception as exc:  # record a partial download instead of hiding it
            print(f"skip bbh/{config}: {type(exc).__name__}")
            continue
        for i, row in enumerate(ds):
            rows.append({"sample_id": f"bbh-{config}-{i:04d}", "task": "bbh", "split": "test",
                         "prompt": row["input"], "answer": str(row["target"]).strip(),
                         "metadata": {"subtask": config}})
    write_jsonl(PROCESSED / "bbh/test.jsonl", rows)
    write_manifest(PROCESSED / "bbh.json", n=len(rows), subtasks=sorted({r['metadata']['subtask'] for r in rows}))


def ifeval():
    ds = load_dataset("google/IFEval", split="train")
    rows = []
    for i, row in enumerate(ds):
        rows.append({"sample_id": f"ifeval-{i:04d}", "task": "ifeval", "split": "test",
                     "prompt": row["prompt"], "answer": "pass",
                     "metadata": {"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"]}})
    write_jsonl(PROCESSED / "ifeval/test.jsonl", rows)
    write_manifest(PROCESSED / "ifeval.json", n=len(rows), fingerprint=fingerprint(ds))


def humaneval_plus():
    ds = load_dataset("evalplus/humanevalplus", split="test")
    rows = []
    for i, row in enumerate(ds):
        rows.append({"sample_id": f"humaneval-plus-{i:04d}", "task": "humaneval_plus", "split": "test",
                     "prompt": row["prompt"], "answer": "code",
                     "metadata": {"entry_point": row["entry_point"], "test_code": row["test"],
                                  "task_id": row.get("task_id")}})
    write_jsonl(PROCESSED / "humaneval_plus/test.jsonl", rows)
    write_manifest(PROCESSED / "humaneval_plus.json", n=len(rows), fingerprint=fingerprint(ds))


def main():
    tasks = {"gsm8k": gsm8k, "svamp": svamp, "mmlu_pro": mmlu_pro, "bbh": bbh,
             "ifeval": ifeval, "humaneval_plus": humaneval_plus}
    for name, function in tasks.items():
        print(f"== {name} ==", flush=True)
        function()
    print("Data prepared. Next: foundation-eval prepare --input data/raw/gsm8k_train.jsonl --test data/raw/gsm8k_test.jsonl")


if __name__ == "__main__":
    main()
