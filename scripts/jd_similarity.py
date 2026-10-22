#!/usr/bin/env python3
"""Re-derive the JD-coverage estimate documented in docs/JD_SIMILARITY.md.

The per-capability depth weights are an explicit, auditable judgement — not a
measured benchmark score. Keeping them in code makes the claim reproducible and
open to challenge, instead of hiding a hand-written percentage in prose.

Input: data/manifests/jd_review.json (70 frozen JD records, each with a
capability tag list). Output: per-group coverage, sensitivity band, loss
decomposition, improvement scenarios and the strongest/weakest JDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]

# How much of each JD capability this project actually carries.
# 1.0 = implemented and exercised; 0.6 = thin/subset implementation;
# 0.2 = background only; 0.0 = explicit non-goal.
DEPTH = {
    "benchmark": 0.60,      # wires up public benchmarks; does not build new ones
    "metric": 0.85,         # native metrics, extraction, Wilson/bootstrap
    "judge": 0.25,          # auxiliary error notes only, never the metric
    "agent": 0.05,          # trajectory work lives in the sibling retail project
    "calibration": 0.75,    # paired bootstrap, per-seed deltas, protocol checks
    "contamination": 0.70,  # exact / near-duplicate / group checks on visible data
    "safety": 0.00,         # non-goal
    "multimodal": 0.00,     # non-goal
    "longctx": 0.60,        # RULER-style 4K/8K/16K local diagnostic
    "code": 0.45,           # HumanEval+ pass@1 only; no Code Agent
    "data": 0.75,           # fixed split, token-stratified selection, export
    "training": 0.45,       # LoRA SFT only; no pretraining/mid-training
    "platform": 0.35,       # thin harness, no scheduler/web product
    "diagnosis": 0.70,      # pool-level right/wrong diagnosis feeding selection
}

# Capabilities the project targets in its own lane (platform is deliberately
# excluded: the docs claim a thin harness, not a platform product). Used for the
# "main lane" coverage number and the capability-set Jaccard.
TARGET = {"benchmark", "metric", "calibration", "contamination", "longctx",
          "code", "data", "training", "diagnosis"}

NON_GOAL = {"multimodal", "code", "longctx"}

GROUPS = {
    "all70": lambda r: True,
    "core51": lambda r: r.get("category") == "core",
    "adjacent18": lambda r: r.get("category") == "adjacent",
    "intern_campus42": lambda r: r.get("level_kind") in {"intern", "campus"},
    "intern_core28": lambda r: r.get("category") == "core" and r.get("level_kind") in {"intern", "campus"},
    "social28": lambda r: r.get("level_kind") == "social",
}

SCENARIOS = {
    "platform=0.8": {"platform": 0.80},
    "agent=0.5": {"agent": 0.50},
    "safety=0.7": {"safety": 0.70},
    "training=0.9": {"training": 0.90},
    "judge=0.8": {"judge": 0.80},
    "five_up": {"platform": 0.80, "agent": 0.50, "safety": 0.70, "training": 0.90, "judge": 0.80},
    "five_up_plus_multimodal_code": {"platform": 0.80, "agent": 0.50, "safety": 0.70,
                                     "training": 0.90, "judge": 0.80, "multimodal": 0.50,
                                     "code": 0.80},
}


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data["records"]
    missing = sorted({cap for r in records for cap in r["capabilities"]} - set(DEPTH))
    if missing:
        raise SystemExit(f"capabilities without a registered depth: {missing}")
    return records


def coverage(record: dict, depth: dict | None = None) -> float:
    caps = record.get("capabilities") or []
    depth = depth or DEPTH
    return mean(depth[cap] for cap in caps) if caps else 0.0


def jaccard(record: dict) -> float:
    caps = set(record.get("capabilities") or [])
    return len(caps & TARGET) / len(caps | TARGET) if caps else 0.0


def main_lane(record: dict) -> float | None:
    caps = [cap for cap in record.get("capabilities") or [] if cap in TARGET]
    return mean(DEPTH[cap] for cap in caps) if caps else None


def coverage_without_non_goals(record: dict) -> float | None:
    caps = [cap for cap in record.get("capabilities") or [] if cap not in NON_GOAL]
    return mean(DEPTH[cap] for cap in caps) if caps else None


def shifted(delta: float) -> dict:
    return {cap: min(1.0, max(0.0, depth + delta)) for cap, depth in DEPTH.items()}


def loss_table(records: list[dict]) -> list[dict]:
    freq = {cap: 0 for cap in DEPTH}
    for record in records:
        for cap in set(record["capabilities"]):
            freq[cap] += 1
    total = sum(freq.values())
    rows = [{"capability": cap, "mentions": count, "depth": DEPTH[cap],
             "loss": round(count * (1 - DEPTH[cap]), 2)}
            for cap, count in freq.items()]
    rows.sort(key=lambda row: -row["loss"])
    return [{"total_mentions": total, "total_loss": round(sum(row["loss"] for row in rows), 2)}] + rows


def distribution(records: list[dict]) -> list[dict]:
    buckets = [(0.50, 1.01), (0.40, 0.50), (0.35, 0.40), (0.30, 0.35), (0.00, 0.30)]
    values = [coverage(record) for record in records]
    return [{"low": low, "high": high, "n": sum(1 for value in values if low <= value < high)}
            for low, high in buckets]


def summarise(records: list[dict]) -> dict:
    groups = {}
    for name, select in GROUPS.items():
        rows = [r for r in records if select(r)]
        groups[name] = {
            "n": len(rows),
            "coverage": round(mean(coverage(r) for r in rows), 3),
            "jaccard": round(mean(jaccard(r) for r in rows), 3),
        }
    lanes = [main_lane(r) for r in records if main_lane(r) is not None]
    without = [coverage_without_non_goals(r) for r in records if coverage_without_non_goals(r) is not None]
    pairs = [(cap, DEPTH[cap]) for r in records for cap in r["capabilities"]]
    base = mean(coverage(r) for r in records)
    scenarios = {"base": round(base, 3)}
    for name, updates in SCENARIOS.items():
        depth = dict(DEPTH)
        depth.update(updates)
        scenarios[name] = round(mean(coverage(r, depth) for r in records), 3)
    return {
        "records": len(records),
        "capability_mentions": len(pairs),
        "groups": groups,
        "main_lane": {"n": len(lanes), "coverage": round(mean(lanes), 3)},
        "without_non_goals": round(mean(without), 3),
        "sensitivity": {"pessimistic": round(mean(coverage(r, shifted(-0.15)) for r in records), 3),
                        "optimistic": round(mean(coverage(r, shifted(0.15)) for r in records), 3)},
        "binary_coverage": round(sum(1 for _, depth in pairs if depth >= 0.6) / len(pairs), 3),
        "out_of_scope_share": round(sum(1 for _, depth in pairs if depth < 0.3) / len(pairs), 3),
        "distribution": distribution(records),
        "scenarios": scenarios,
        "top": [
            {"jd_id": r["jd_id"], "company": r["company"], "coverage": round(coverage(r), 3)}
            for r in sorted(records, key=coverage, reverse=True)[:8]
        ],
        "bottom": [
            {"jd_id": r["jd_id"], "company": r["company"], "coverage": round(coverage(r), 3)}
            for r in sorted(records, key=coverage)[:5]
        ],
    }


def capability_table(records: list[dict]) -> list[dict]:
    freq = {cap: 0 for cap in DEPTH}
    for record in records:
        for cap in set(record["capabilities"]):
            freq[cap] += 1
    total = sum(freq.values())
    return [
        {"capability": cap, "mentions": count, "depth": DEPTH[cap],
         "weighted_share": round(count * DEPTH[cap] / total, 3)}
        for cap, count in sorted(freq.items(), key=lambda item: -item[1])
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, default=ROOT / "data/manifests/jd_review.json")
    parser.add_argument("--json", action="store_true", help="emit the raw summary as JSON")
    args = parser.parse_args()

    records = load_records(args.review)
    summary = summarise(records)
    if args.json:
        print(json.dumps({"summary": summary, "capabilities": capability_table(records),
                          "loss": loss_table(records)}, ensure_ascii=False, indent=2))
        return

    print(f"{summary['records']} JDs, {summary['capability_mentions']} capability mentions")
    for name, row in summary["groups"].items():
        print(f"  {name:16s} n={row['n']:2d}  coverage={row['coverage']:.3f}  jaccard={row['jaccard']:.3f}")
    print(f"  {'main_lane':16s} n={summary['main_lane']['n']:2d}  coverage={summary['main_lane']['coverage']:.3f}")
    print(f"  without_non_goals={summary['without_non_goals']:.3f}  "
          f"pessimistic={summary['sensitivity']['pessimistic']:.3f}  "
          f"optimistic={summary['sensitivity']['optimistic']:.3f}")
    print(f"  binary={summary['binary_coverage']:.3f}  out-of-scope={summary['out_of_scope_share']:.3f}")
    print("\ndistribution:")
    for row in summary["distribution"]:
        print(f"  [{row['low']:.2f},{row['high']:.2f}): {row['n']}")
    print("\ncapability       mentions  depth  weighted_share")
    for row in capability_table(records):
        print(f"  {row['capability']:14s} {row['mentions']:4d}   {row['depth']:.2f}   {row['weighted_share']:.3f}")
    print("\nloss:")
    for row in loss_table(records)[1:]:
        print(f"  {row['capability']:14s} freq={row['mentions']:3d} loss={row['loss']:6.2f}")
    print("\nscenarios:")
    for name, value in summary["scenarios"].items():
        print(f"  {name:32s} {value:.3f}")
    print("\ntop:")
    for row in summary["top"]:
        print(f"  {row['jd_id']} {row['company']} {row['coverage']:.3f}")
    print("bottom:")
    for row in summary["bottom"]:
        print(f"  {row['jd_id']} {row['company']} {row['coverage']:.3f}")


if __name__ == "__main__":
    main()
