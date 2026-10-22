"""Native-metric reports, paired comparisons and the evaluation matrix.

No composite score is produced. Each task keeps its own denominator, statuses
and interval; completion and chat protocols are reported separately.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .schema import Score, digest, read_jsonl, wilson_interval, write_json

STATUSES = ("ok", "extract_error", "unsupported", "execution_error", "judge_error", "error")


def summarize(scores: list[Score], task: str):
    by_status = Counter(score.status for score in scores)
    decided = [s for s in scores if s.status == "ok" and s.correct is not None]
    correct = sum(bool(s.correct) for s in decided)
    return {
        "task": task,
        "n": len(scores),
        "decided": len(decided),
        "correct": correct,
        "accuracy": correct / len(decided) if decided else None,
        "wilson95": wilson_interval(correct, len(decided)) if decided else [None, None],
        "coverage": len(decided) / len(scores) if scores else None,
        "status": {status: by_status.get(status, 0) for status in STATUSES},
    }


def read_run(run_dir: Path):
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    scores = [Score.model_validate(row) for row in read_jsonl(run_dir / "scores.jsonl")]
    return manifest, scores


def summarize_run(run_dir: Path):
    manifest, scores = read_run(run_dir)
    summary = summarize(scores, manifest.get("task", "unknown"))
    return {"run": str(run_dir), "model": manifest.get("model_label"), "protocol": manifest.get("protocol"),
            "training_seed": manifest.get("training_seed"), "manifest_status": manifest.get("status"),
            "protocol_signature": manifest.get("protocol_signature"), **summary}


def paired_comparison(a_runs: list[Path], b_runs: list[Path], samples: int = 10000, seed: int = 2026):
    """B minus A on paired test samples, preserving training-seed replicates."""
    if len(a_runs) != len(b_runs) or not a_runs:
        raise ValueError("comparison needs one B run per A run")
    seeds, per_seed, pairs_for_bootstrap = [], {}, []
    protocols = set()
    for a_dir, b_dir in zip(a_runs, b_runs):
        a_manifest, a_scores = read_run(a_dir)
        b_manifest, b_scores = read_run(b_dir)
        if a_manifest.get("status") != "complete" or b_manifest.get("status") != "complete":
            raise ValueError("comparison requires complete runs")
        if not a_manifest.get("benchmark_complete") or not b_manifest.get("benchmark_complete"):
            raise ValueError("comparison requires benchmark-complete runs")
        if a_manifest.get("actual_records") != a_manifest.get("expected_records") or b_manifest.get("actual_records") != b_manifest.get("expected_records"):
            raise ValueError("comparison requires every expected record")
        if a_manifest.get("task") != b_manifest.get("task") or a_manifest.get("protocol") != b_manifest.get("protocol"):
            raise ValueError("task/protocol mismatch")
        if a_manifest.get("split") != "test" or b_manifest.get("split") != "test":
            raise ValueError("the main comparison uses the frozen test split only")
        if a_manifest.get("protocol_signature") != b_manifest.get("protocol_signature"):
            raise ValueError("protocol signature changed between runs")
        protocols.add(a_manifest["protocol_signature"])
        a_map = {s.sample_id: s for s in a_scores if s.status == "ok"}
        b_map = {s.sample_id: s for s in b_scores if s.status == "ok"}
        if set(a_map) != set(b_map):
            raise ValueError("paired comparison needs identical decided samples")
        ids = sorted(a_map)
        diffs = np.array([int(bool(b_map[i].correct)) - int(bool(a_map[i].correct)) for i in ids], dtype=float)
        training_seed = a_manifest.get("training_seed")
        if training_seed != b_manifest.get("training_seed"):
            raise ValueError("training seeds do not match within a pair")
        seeds.append(training_seed)
        per_seed[str(training_seed)] = float(diffs.mean()) if len(diffs) else None
        pairs_for_bootstrap.append(diffs)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        seed_draws = [float(diff[rng.integers(0, len(diff), size=len(diff))].mean()) for diff in pairs_for_bootstrap if len(diff)]
        draws.append(float(np.mean(seed_draws)))
    per_seed_values = [v for v in per_seed.values() if v is not None]
    return {"estimand": "B minus A native accuracy on paired test samples; equal seed weight",
            "difference": float(np.mean(per_seed_values)) if per_seed_values else None,
            "ci95": np.quantile(draws, [.025, .975]).tolist() if draws else [None, None],
            "training_seeds": seeds, "per_seed_difference": per_seed,
            "seed_spread": [min(per_seed_values), max(per_seed_values)] if per_seed_values else None,
            "bootstrap_samples": samples, "bootstrap_seed": seed,
            "protocol_signatures": sorted(protocols),
            "warning": "one training seed cannot estimate training variability" if len(seeds) == 1 else None}


def make_report(run_dir: Path, output: Path | None = None):
    manifest, scores = read_run(run_dir)
    output = output or (run_dir / "report")
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(scores, manifest.get("task", "unknown"))
    responses = read_jsonl(run_dir / "responses.jsonl") if (run_dir / "responses.jsonl").exists() else []
    costs = {"calls": len(responses), "known_tokens": 0, "usage_records": 0,
             "mean_seconds": float(np.mean([r.get("elapsed_seconds", 0) for r in responses])) if responses else None}
    for row in responses:
        usage = row.get("usage") or {}
        tokens = usage.get("total_tokens")
        if tokens is None and usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
            tokens = usage["prompt_tokens"] + usage["completion_tokens"]
        if tokens is not None:
            costs["known_tokens"] += tokens
            costs["usage_records"] += 1
    payload = {"run": str(run_dir), "model": manifest.get("model_label"), "protocol": manifest.get("protocol"),
               "training_seed": manifest.get("training_seed"), "status": manifest.get("status"),
               "not_a_benchmark": manifest.get("not_a_benchmark", False), "native_metric": summary,
               "cost": costs, "contamination": manifest.get("contamination"),
               "protocol_signature": manifest.get("protocol_signature"),
               "data_hashes": manifest.get("data_hashes", {}), "endpoint": manifest.get("endpoint", {})}
    write_json(output / "summary.json", payload)
    with (output / "per_sample.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "status", "correct", "metric", "detail"])
        writer.writeheader()
        for score in scores:
            writer.writerow({"sample_id": score.sample_id, "status": score.status, "correct": score.correct,
                             "metric": score.metric, "detail": json.dumps(score.detail, ensure_ascii=False)})
    title = "SCRIPTED DEMO — not model benchmark results" if manifest.get("not_a_benchmark") else "Foundation-model evaluation run"
    fmt = lambda value: "N/A" if value is None else f"{value:.3f}"
    lines = [f"# {title}", "",
             "Native metrics are reported per task. No composite score is computed.", "",
             "| Task | Protocol | Model | Seed | N | Decided | Accuracy | 95% Wilson | Coverage |",
             "|---|---|---|---|---:|---:|---:|---|---:|",
             f"| {summary['task']} | {manifest.get('protocol')} | {manifest.get('model_label')} | "
             f"{manifest.get('training_seed')} | {summary['n']} | {summary['decided']} | "
             f"{fmt(summary['accuracy'])} | {fmt(summary['wilson95'][0])}–{fmt(summary['wilson95'][1])} | "
             f"{fmt(summary['coverage'])} |", "",
             "## Status accounting", "", json.dumps(summary["status"], ensure_ascii=False), "",
             "## Cost", "", json.dumps(costs, ensure_ascii=False), "",
             "## Contamination check", "",
             json.dumps(manifest.get("contamination", {"status": "not recorded"}), ensure_ascii=False), "",
             "## Failure examples", ""]
    failures = [s for s in scores if s.status != "ok" or s.correct is False][:10]
    for score in failures:
        lines.append(f"- `{score.sample_id}`: status={score.status}, correct={score.correct}, "
                     f"detail={json.dumps(score.detail, ensure_ascii=False)[:240]}")
    lines += ["", "## What this does not show", "",
              "- It does not prove that the base model or its pretraining data is contamination-free.",
              "- It does not turn IFEval rule checks or LLM error notes into a general capability score.",
              "- A single run is not evidence of a training effect; use the B0/B1/B2 matrix and paired CI.",
              ""]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_summary(payload, output)
    return payload


def plot_summary(payload: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    statuses = payload["native_metric"]["status"]
    axes[0].bar(list(statuses), list(statuses.values()), color="#356c9b")
    axes[0].set(title="Score status accounting", ylabel="Samples")
    axes[1].bar(["decided", "undecided"],
                [payload["native_metric"]["decided"],
                 payload["native_metric"]["n"] - payload["native_metric"]["decided"]], color="#c6763b")
    axes[1].set(title="Coverage (unsupported is not pass)", ylabel="Samples")
    fig.suptitle("SCRIPTED DEMO — no benchmark claims" if payload["not_a_benchmark"] else "Native metric diagnostics")
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)


def comparison_report(result: dict, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    fmt = lambda value: "N/A" if value is None else f"{value:+.3f}"
    lines = ["# Paired B−A comparison", "",
             "| Estimand | Difference | 95% CI | Seeds |", "|---|---:|---|---|",
             f"| {result['estimand']} | {fmt(result['difference'])} | "
             f"{fmt(result['ci95'][0])} to {fmt(result['ci95'][1])} | {result['training_seeds']} |", "",
             "## Per training seed", ""]
    for seed, value in result["per_seed_difference"].items():
        lines.append(f"- seed {seed}: {fmt(value)}")
    if result.get("warning"):
        lines += ["", f"**Warning**: {result['warning']}"]
    lines += ["", "A CI crossing zero does not support a stable improvement claim. "
              "The result is conditional on this model, data revision and protocol."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def eval_matrix(runs: list[Path], output: Path):
    rows = []
    for run in runs:
        if not (run / "manifest.json").exists():
            continue
        manifest, scores = read_run(run)
        summary = summarize(scores, manifest.get("task", "unknown"))
        rows.append({"run": run.name, "model": manifest.get("model_label"), "task": manifest.get("task"),
                     "protocol": manifest.get("protocol"), "seed": manifest.get("training_seed"),
                     "status": manifest.get("status"), "n": summary["n"], "accuracy": summary["accuracy"],
                     "decided": summary["decided"], "protocol_signature": manifest.get("protocol_signature")})
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "matrix.json", rows)
    with (output / "matrix.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Evaluation matrix", "", "| Model | Task | Protocol | Seed | Status | N | Accuracy |",
             "|---|---|---|---:|---|---:|---:|"]
    for row in rows:
        acc = "N/A" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
        lines.append(f"| {row['model']} | {row['task']} | {row['protocol']} | {row['seed']} | "
                     f"{row['status']} | {row['n']} | {acc} |")
    (output / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
