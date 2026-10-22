#!/usr/bin/env python3
"""Validate recorded runs and assemble the matrix/comparison artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from foundation_eval.report import eval_matrix, paired_comparison
from foundation_eval.schema import write_json

ROOT = Path(__file__).resolve().parents[1]


def load_manifest(run: Path):
    path = run / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", help="only validate this model label")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    runs, issues = [], []
    (ROOT / "runs").mkdir(exist_ok=True)
    for folder in sorted((ROOT / "runs").iterdir()):
        if not folder.is_dir():
            continue
        manifest = load_manifest(folder)
        if not manifest or manifest.get("kind") != "collect":
            continue
        if args.label and manifest.get("model_label") != args.label:
            continue
        runs.append(folder)
        if manifest.get("status") != "complete":
            issues.append(f"{folder.name}: status={manifest.get('status')}")
        if manifest.get("actual_records") != manifest.get("expected_records"):
            issues.append(f"{folder.name}: records {manifest.get('actual_records')}/{manifest.get('expected_records')}")
        if not manifest.get("protocol_signature"):
            issues.append(f"{folder.name}: missing protocol signature")
    if not runs:
        issues.append("no collect runs found")
    output = ROOT / "runs/validation"
    rows = eval_matrix(runs, output) if runs else []
    gsm = {"B1": {}, "B2": {}}
    for folder in runs:
        manifest = load_manifest(folder)
        if (manifest.get("task") == "gsm8k" and manifest.get("protocol") == "completion"
                and manifest.get("split") == "test" and manifest.get("training_seed") in {42, 43, 44}
                and manifest.get("model_label") in gsm and manifest.get("status") == "complete"):
            gsm[manifest["model_label"]][manifest["training_seed"]] = folder
    paired = None
    if all(set(gsm[arm]) == {42, 43, 44} for arm in gsm):
        a = [gsm["B1"][seed] for seed in (42, 43, 44)]
        b = [gsm["B2"][seed] for seed in (42, 43, 44)]
        paired = paired_comparison(a, b)
        write_json(output / "gsm8k_b2_minus_b1.json", paired)
        print(json.dumps({"difference": paired["difference"], "ci95": paired["ci95"]}, ensure_ascii=False))
    else:
        print("main comparison pending: need B1 and B2 GSM8K test runs for seeds 42/43/44")
    print(json.dumps({"runs": len(runs), "issues": issues, "matrix_rows": len(rows)}, ensure_ascii=False, indent=2))
    if issues and not args.allow_incomplete:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
