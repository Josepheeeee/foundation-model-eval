#!/usr/bin/env python3
"""Compare our GSM8K extraction with the pinned upstream lm-eval task regex.

This reads the upstream YAML (the same file the harness uses to configure its
strict-match filter) and applies its `regex_pattern` to every local gold answer
and response. It is a per-sample parity check, not a full lm-eval run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from foundation_eval.data import extract_final_number

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "external/lm-evaluation-harness"


def upstream_regex() -> str:
    candidates = list(UPSTREAM.glob("lm_eval/tasks/gsm8k/*.yaml"))
    if not candidates:
        raise SystemExit("fetch the pinned harness first: python scripts/fetch_references.py lm-evaluation-harness")
    for path in candidates:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in data.get("filter_list", []):
            for item in entry.get("filter", []):
                pattern = item.get("regex_pattern")
                if pattern:
                    return pattern
    raise SystemExit("no regex_pattern found in the upstream gsm8k task")


def upstream_extract(pattern: str, text: str) -> str | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return str(value).replace(",", "").replace("$", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=ROOT / "data/raw/gsm8k_test.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    pattern = upstream_regex()
    rows = [json.loads(line) for line in args.answers.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    mismatches = []
    for index, row in enumerate(rows):
        gold = row.get("answer", "")
        ours = extract_final_number(gold)
        theirs = upstream_extract(pattern, gold)
        if ours != theirs:
            mismatches.append({"index": index, "ours": ours, "upstream": theirs, "answer": gold[-120:]})
    print(json.dumps({"n": len(rows), "regex": pattern, "mismatches": len(mismatches),
                      "examples": mismatches[:10]}, ensure_ascii=False, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
