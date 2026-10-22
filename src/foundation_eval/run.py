"""One CLI for reproducible foundation-model measurement and LightSFT export."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml
from dotenv import load_dotenv

from .agent import ModelAdapter, ModelEndpoint, ScriptedAdapter
from .data import (assert_no_contamination, build_ruler_samples, contamination_report, export_sft,
                   load_samples, prepare_gsm8k_split, save_samples, write_llamafactory_config)
from .judge import score_sample
from .report import comparison_report, eval_matrix, make_report, paired_comparison, summarize_run
from .schema import Sample, Score, WeightRecord, digest, file_digest, read_jsonl, write_json, write_jsonl
from .selection import make_selections, register_weight, training_export

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "foundation-eval-protocol-v1"


def load_config(path: Path | None = None):
    config = yaml.safe_load((path or ROOT / "configs/experiment.yaml").read_text(encoding="utf-8"))
    tasks = yaml.safe_load((ROOT / config["tasks_file"]).read_text(encoding="utf-8"))["tasks"]
    models = yaml.safe_load((ROOT / config["models_file"]).read_text(encoding="utf-8"))["models"]
    if config["selection_count"] <= 0 or config["dev_count"] <= 0:
        raise ValueError("dev_count and selection_count must be positive")
    if config["token_tolerance"] > 0.01:
        raise ValueError("token budget tolerance cannot exceed 1%")
    if set(config["b2"]["wrong_weight"] for _ in [0]) != {3}:
        raise ValueError("B2 wrong-answer weight is preregistered at 3")
    return config, tasks, models


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def processed(task: str, split: str) -> Path:
    return ROOT / "data/processed" / task / f"{split}.jsonl"


def prompt_path(task: str) -> Path:
    specific = ROOT / f"prompts/{task}.md"
    return specific if specific.exists() else ROOT / "prompts/system.md"


def fewshot_for(task: str):
    path = ROOT / "data/fixtures/fewshot.jsonl"
    if not path.exists():
        return []
    return [s for s in load_samples(path) if s.task == task]


def code_hashes():
    return {str(path.relative_to(ROOT)).replace("\\", "/"): file_digest(path)
            for path in sorted((ROOT / "src").rglob("*.py"))}


def task_signature(task_spec: dict, generation: dict, data_hash: str, prompt_hash: str, fewshot_hash: str):
    return digest({"protocol_version": PROTOCOL_VERSION, "task": task_spec, "generation": generation,
                   "data_sha256": data_hash, "prompt_sha256": prompt_hash, "fewshot_sha256": fewshot_hash,
                   "code": code_hashes()})


def load_tokenizer(name: str | None):
    if not name:
        return None, "chars/4"
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ValueError("install transformers to use an exact tokenizer, or omit --tokenizer for chars/4") from None
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    return (lambda text: tokenizer.encode(text, add_special_tokens=False)), f"hf:{name}"


def endpoint_public():
    endpoint = ModelEndpoint.from_env("TARGET")
    return endpoint, endpoint.public()


def assert_budget(config: dict, calls: int):
    budget = config.get("budget", {})
    limit = int(budget.get("max_calls", 0))
    if limit and calls > limit:
        raise ValueError(f"registered API budget exhausted: {calls} calls > {limit}")
    host = (os.getenv("TARGET_BASE_URL") or "").lower()
    remote = host and not any(local in host for local in ("127.0.0.1", "localhost", "0.0.0.0"))
    if remote and not limit:
        raise ValueError("remote endpoint requires configs/experiment.yaml budget.max_calls before any call")
    if remote and (budget.get("price_input_per_million") is None or budget.get("price_output_per_million") is None):
        raise ValueError("remote endpoint requires input/output prices (0 is allowed) before any call")


def new_run(kind: str, config: dict, task: str, split: str, protocol: str, model_label: str,
            training_seed: int | None, endpoint: dict, data_hashes: dict, signature: str,
            notes: str = "", not_a_benchmark=False):
    path = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + kind + "-" + uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    for folder in ("raw", "records"):
        (path / folder).mkdir()
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = "uncommitted"
    contamination_path = ROOT / "data/manifests/contamination.json"
    contamination = (json.loads(contamination_path.read_text(encoding="utf-8"))
                     if contamination_path.exists() else {"status": "not run"})
    manifest = {"schema_version": 1, "kind": kind, "status": "running",
                "created_utc": datetime.now(timezone.utc).isoformat(), "git_head": git_head,
                "task": task, "split": split, "protocol": protocol, "model_label": model_label,
                "training_seed": training_seed, "endpoint": endpoint, "protocol_signature": signature,
                "config": {"generation": config.get("generation", {})}, "source_hashes": code_hashes(),
                "data_hashes": data_hashes, "contamination": contamination,
                "not_a_benchmark": not_a_benchmark, "notes": notes,
                "python": platform.python_version(),
                "packages": {name: importlib.metadata.version(name) for name in ("pydantic", "httpx", "numpy", "PyYAML")
                             if importlib.util.find_spec(name) is not None}}
    write_json(path / "manifest.json", manifest)
    return path, manifest


def finish(path: Path, manifest: dict, **updates):
    manifest.update(updates)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(path / "manifest.json", manifest)


def cli_parser():
    parser = argparse.ArgumentParser(description="Foundation-model measurement and LightSFT selection")
    parser.add_argument("--config", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("demo")
    q = sub.add_parser("check")
    q.add_argument("--offline", action="store_true", help="validate code/manifests without downloaded datasets")
    q = sub.add_parser("prepare")
    q.add_argument("--input", type=Path, help="local GSM8K train JSONL; use scripts/linux/download_data.sh first")
    q.add_argument("--test", type=Path, help="local GSM8K test JSONL")
    q = sub.add_parser("ruler-build")
    q.add_argument("--lengths", type=int, nargs="+", default=None)
    q = sub.add_parser("collect")
    q.add_argument("--task", required=True)
    q.add_argument("--split", required=True, choices=["train", "dev", "test", "diagnostic"])
    q.add_argument("--protocol", choices=["completion", "chat"], default=None)
    q.add_argument("--model-label", required=True)
    q.add_argument("--training-seed", type=int)
    q.add_argument("--limit", type=int)
    q.add_argument("--samples", type=Path)
    q.add_argument("--diagnose-only", action="store_true")
    q = sub.add_parser("select")
    q.add_argument("--pool", type=Path, required=True)
    q.add_argument("--diagnosis", type=Path, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "data/generated/selection")
    q.add_argument("--tokenizer", default=None, help="HF tokenizer id for exact Token accounting; default is chars/4")
    q = sub.add_parser("train-export")
    q.add_argument("--selection", type=Path, default=ROOT / "data/generated/selection")
    q.add_argument("--output", type=Path, default=ROOT / "data/generated/training")
    q = sub.add_parser("train-config")
    q.add_argument("--data", type=Path, required=True)
    q.add_argument("--base-model", required=True)
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "configs/generated")
    q = sub.add_parser("weight-register")
    q.add_argument("--label", required=True)
    q.add_argument("--base-model", required=True)
    q.add_argument("--adapter", type=Path, required=True)
    q.add_argument("--data-sha256", required=True)
    q.add_argument("--config-sha256", required=True)
    q.add_argument("--training-seed", type=int, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "data/manifests/weights.jsonl")
    q = sub.add_parser("freeze")
    q.add_argument("--task", required=True)
    q.add_argument("--protocol", choices=["completion", "chat"], default=None)
    q = sub.add_parser("report")
    q.add_argument("--run", type=Path, required=True)
    q.add_argument("--output", type=Path)
    q = sub.add_parser("compare")
    q.add_argument("--a", nargs="+", type=Path, required=True)
    q.add_argument("--b", nargs="+", type=Path, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "runs/comparison")
    q = sub.add_parser("eval-matrix")
    q.add_argument("--runs", nargs="+", type=Path, required=True)
    q.add_argument("--output", type=Path, default=ROOT / "runs/matrix")
    return parser


def main(argv=None):
    args = cli_parser().parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    try:
        config, tasks, models = load_config(args.config)
        execute(args, config, tasks, models)
    except (ValueError, RuntimeError, FileNotFoundError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


def execute(args, config, tasks, models):
    cmd = args.command
    if cmd == "doctor":
        endpoint = "not configured"
        try:
            endpoint = ModelEndpoint.from_env("TARGET").public()
        except ValueError:
            pass
        print(json.dumps({"python": platform.python_version(), "root": str(ROOT), "endpoint": endpoint,
                          "tasks": {k: v["role"] for k, v in tasks.items()},
                          "data_processed": (ROOT / "data/processed").exists(),
                          "code_hash": digest(code_hashes())}, ensure_ascii=False, indent=2))
        return
    if cmd == "prepare":
        if not args.input:
            raise ValueError("prepare needs --input (local GSM8K train JSONL); use the Linux download script")
        rows = read_jsonl(args.input)
        splits = prepare_gsm8k_split(rows, config["dev_count"], config["dev_salt"])
        for split, samples in splits.items():
            save_samples(processed("gsm8k", split), samples)
        if args.test:
            from .data import make_sample_id
            test_rows = read_jsonl(args.test)
            test = []
            for row in test_rows:
                test.append(Sample.model_validate({
                    "sample_id": make_sample_id("gsm8k", row.get("id"), row.get("question", "")),
                    "task": "gsm8k", "split": "test", "prompt": row.get("question", ""),
                    "answer": row.get("answer", "")}))
            save_samples(processed("gsm8k", "test"), test)
        test_samples = load_samples(processed("gsm8k", "test")) if processed("gsm8k", "test").exists() else []
        write_json(ROOT / "data/manifests/gsm8k.json",
                   {"dev": len(splits["dev"]), "train": len(splits["train"]), "test": len(test_samples),
                    "dev_salt": config["dev_salt"], "input_sha256": file_digest(args.input),
                    "dataset_revision": "record-at-prepare; see run manifest"})
        write_json(ROOT / "data/manifests/splits.json",
                   {"schema_version": 1, "status": "prepared", "salt": config["dev_salt"],
                    "dev": [s.sample_id for s in splits["dev"]],
                    "train": [s.sample_id for s in splits["train"]],
                    "test": [s.sample_id for s in test_samples],
                    "dev_count": len(splits["dev"]), "train_count": len(splits["train"]),
                    "test_count": len(test_samples),
                    "note": "Official test is frozen for measurement only; it never enters diagnosis or selection."})
        print(json.dumps({"dev": len(splits["dev"]), "train": len(splits["train"])}))
        return
    if cmd == "ruler-build":
        lengths = args.lengths or config["ruler"]["lengths"]
        for length in lengths:
            for task_name in config["ruler"]["tasks"]:
                samples = build_ruler_samples(length, task_name, config["ruler"]["samples_per_cell"], config["ruler"]["seed"])
                save_samples(processed("ruler", f"{length}_{task_name}"), samples)
        print(json.dumps({"lengths": lengths, "tasks": config["ruler"]["tasks"],
                          "per_cell": config["ruler"]["samples_per_cell"]}))
        return
    if cmd == "check":
        issues = check_workspace(config, tasks, offline=args.offline)
        print(json.dumps(issues, ensure_ascii=False, indent=2))
        if any(issues.values()):
            raise ValueError("workspace checks failed; see the report above")
        return
    if cmd == "demo":
        run_demo(config, tasks, models)
        return
    if cmd == "freeze":
        task = args.task
        protocol = args.protocol or config["primary_protocol"]
        sig, data_hashes = freeze_signature(task, protocol, config, tasks)
        target = ROOT / "data/manifests/freeze" / f"{task}-{protocol}.json"
        if target.exists():
            raise ValueError("freeze already exists; keep the first frozen protocol")
        write_json(target, {"frozen_utc": datetime.now(timezone.utc).isoformat(), "task": task,
                            "protocol": protocol, "protocol_signature": sig, "data_hashes": data_hashes,
                            "task_spec": tasks[task], "generation": config["generation"]})
        print(target)
        return
    if cmd == "collect":
        collect(args, config, tasks, models)
        return
    if cmd == "select":
        tokenizer, counter_name = load_tokenizer(args.tokenizer)
        manifest = make_selections(args.pool, args.diagnosis, config, args.output, tokenizer, counter_name)
        print(json.dumps({"seeds": [row["seed"] for row in manifest["seeds"]], "count": manifest["count"],
                          "token_counter": counter_name}))
        return
    if cmd == "train-export":
        plan = training_export(args.selection, config, args.output)
        print(f"{len(plan)} training jobs -> {args.output}")
        return
    if cmd == "train-config":
        training = write_llamafactory_config(args.seed, args.data, args.output, args.base_model, config)
        print(json.dumps(training, ensure_ascii=False))
        return
    if cmd == "weight-register":
        record = WeightRecord(model_label=args.label, base_model=args.base_model, adapter_path=str(args.adapter),
                              adapter_sha256="pending", training_data_sha256=args.data_sha256,
                              training_config_sha256=args.config_sha256, training_seed=args.training_seed,
                              git_head="working-tree", created_utc=datetime.now(timezone.utc).isoformat())
        register_weight(args.output, record)
        print(args.output)
        return
    if cmd == "report":
        payload = make_report(args.run, args.output)
        print(json.dumps(payload["native_metric"], ensure_ascii=False))
        return
    if cmd == "compare":
        result = paired_comparison(args.a, args.b, config["bootstrap_samples"], config["bootstrap_seed"])
        comparison_report(result, args.output)
        print(json.dumps({"difference": result["difference"], "ci95": result["ci95"]}, ensure_ascii=False))
        return
    if cmd == "eval-matrix":
        rows = eval_matrix(args.runs, args.output)
        print(f"{len(rows)} runs -> {args.output}")
        return
    raise ValueError("unhandled command")


def freeze_signature(task: str, protocol: str, config: dict, tasks: dict):
    spec = tasks[task]
    path = processed(task, spec.get("test_split", "test") if task != "ruler" else "diagnostic")
    if not path.exists():
        raise ValueError("prepare the test/diagnostic split before freezing")
    samples = load_samples(path, task=task)
    data_hash = digest([s.model_dump() for s in samples])
    prompt_hash = file_digest(prompt_path(task))
    fewshot_hash = digest([s.model_dump() for s in fewshot_for(task)])
    return task_signature(spec, config["generation"], data_hash, prompt_hash, fewshot_hash), {"test": data_hash}


def check_workspace(config, tasks, offline=False):
    issues = {}
    for task in tasks:
        path = processed(task, "test")
        if task == "ruler":
            continue
        if path.exists():
            issues[f"test_data:{task}"] = []
        else:
            issues[f"test_data:{task}"] = [] if offline else ["missing; prepare before freezing"]
    gsm_splits = {split: processed("gsm8k", split) for split in ("train", "dev", "test")}
    if all(path.exists() for path in gsm_splits.values()):
        split_samples = {split: load_samples(path, task="gsm8k") for split, path in gsm_splits.items()}
        report = contamination_report(split_samples, fewshot_for("gsm8k"))
        write_json(ROOT / "data/manifests/contamination.json", report)
        issues["contamination"] = ([f"{key}={value}" for key, value in report["counts"].items() if value]
                                   if any(report["counts"].values()) else [])
    else:
        issues["contamination"] = [] if offline else ["prepare GSM8K train/dev/test before checking"]
    selection = ROOT / "data/generated/selection/selection.json"
    if selection.exists():
        record = read(selection)
        problems = []
        if record.get("token_counter") == "chars/4":
            problems.append("selection used chars/4 token approximation; rerun with --tokenizer before training")
        for seed in record["seeds"]:
            gap = seed["budget"]["relative_gap"]
            if gap > config["token_tolerance"]:
                problems.append(f"seed {seed['seed']} token gap {gap:.4f}")
            if seed["b1"]["count"] != config["selection_count"] or seed["b2"]["count"] != config["selection_count"]:
                problems.append(f"seed {seed['seed']} count mismatch")
        issues["selection"] = problems
    else:
        issues["selection"] = [] if offline else ["not built yet (expected before training)"]
    return issues


def collect(args, config, tasks, models):
    if args.task not in tasks:
        raise ValueError("unknown task")
    spec = dict(tasks[args.task])
    protocol = args.protocol or spec["primary_protocol"]
    if protocol not in {"completion", "chat"}:
        raise ValueError("unknown protocol")
    path = args.samples or processed(args.task, args.split)
    if not path.exists():
        raise ValueError(f"missing samples: {path}")
    samples = load_samples(path, task=args.task, split=args.split)
    if args.limit:
        if args.split == "test":
            raise ValueError("formal test cannot be truncated")
        samples = samples[:args.limit]
    if not samples:
        raise ValueError("no samples selected")
    fewshot = fewshot_for(args.task)
    system_prompt = prompt_path(args.task).read_text(encoding="utf-8")
    data_hash = digest([s.model_dump() for s in samples])
    prompt_hash = digest(system_prompt)
    fewshot_hash = digest([s.model_dump() for s in fewshot])
    signature = task_signature(spec, config["generation"], data_hash, prompt_hash, fewshot_hash)
    if args.split == "test":
        frozen = ROOT / "data/manifests/freeze" / f"{args.task}-{protocol}.json"
        if not frozen.exists():
            raise ValueError("formal test requires a frozen protocol")
        if read(frozen)["protocol_signature"] != signature:
            raise ValueError("frozen protocol/data/prompt changed; refreeze before test")
    endpoint, public = endpoint_public()
    assert_budget(config, 0)
    run, manifest = new_run("collect", config, args.task, args.split, protocol, args.model_label,
                            args.training_seed, public, {"samples": data_hash}, signature)
    write_jsonl(run / "samples.jsonl", samples)
    adapter = ModelAdapter(endpoint, {**config["generation"], "max_new_tokens": spec["max_new_tokens"]})
    scores = []
    try:
        for index, sample in enumerate(samples, 1):
            assert_budget(config, index)
            response = adapter.generate(sample, protocol, fewshot, system_prompt, args.model_label)
            score = score_sample(sample, response, spec["native_metric"])
            write_json(run / "raw" / (sample.sample_id + ".json"), {"response": response.model_dump(), "score": score.model_dump()})
            with (run / "responses.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(response.model_dump_json() + "\n")
            with (run / "scores.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(score.model_dump_json() + "\n")
            scores.append(score)
            if index % 10 == 0:
                print(f"{index}/{len(samples)}", flush=True)
        assert_budget(config, len(samples))
        ok = sum(score.status == "ok" for score in scores)
        finish(run, manifest, status="complete", expected_records=len(samples), actual_records=len(scores),
               complete_records=ok, benchmark_complete=ok == len(samples))
        make_report(run)
        if args.diagnose_only:
            write_jsonl(run / "diagnosis.jsonl", [{"sample_id": score.sample_id, "status": score.status,
                                                   "correct": score.correct, "task": score.task} for score in scores])
        print(run)
    except BaseException:
        finish(run, manifest, status="interrupted", expected_records=len(samples), actual_records=len(scores),
               complete_records=sum(score.status == "ok" for score in scores), benchmark_complete=False)
        raise


def run_demo(config, tasks, models):
    """Offline end-to-end demo over a handful of self-made fixtures."""
    fixtures = ROOT / "data/fixtures/demo_samples.jsonl"
    if not fixtures.exists():
        raise ValueError("demo fixtures are missing")
    all_samples = load_samples(fixtures)
    gsm = [s for s in all_samples if s.task == "gsm8k" and s.split == "train"]
    dev_count = min(3, max(1, len(gsm) // 3))
    splits = prepare_gsm8k_split([{"id": s.sample_id.removeprefix("gsm8k-"), "question": s.prompt, "answer": s.answer}
                                 for s in gsm], dev_count, "demo-salt")
    pool = splits["train"]
    # Demo diagnosis: a simple keyword-based stand-in, explicitly marked scripted.
    diagnosis = {}
    for sample in pool:
        diagnosis[sample.sample_id] = len(sample.answer) % 2 == 0
    from .data import build_b1_b2, length_strata
    strata = length_strata(pool, min(3, len(pool)), None)
    selected = build_b1_b2(pool, diagnosis, min(2, len(pool)), [42], strata, None, 0.5)
    output = ROOT / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-demo-" + uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection.json", {"seeds": selected, "count": len(selected[0]["b1"]["ids"]),
                                           "demo": True, "note": "scripted diagnosis; not research data"})
    responses = {row["sample_id"]: row["text"] for row in read_jsonl(ROOT / "data/fixtures/responses.jsonl")}
    adapter = ScriptedAdapter(responses, "scripted-demo")
    system_prompt = prompt_path("gsm8k").read_text(encoding="utf-8")
    for split_name, samples in (("train", splits["train"]), ("dev", splits["dev"])):
        path = output / split_name
        path.mkdir()
        scores = []
        for sample in samples:
            if sample.sample_id not in responses:
                continue
            response = adapter.generate(sample, "completion", [], system_prompt)
            score = score_sample(sample, response, "exact_match_final_answer")
            with (path / "responses.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(response.model_dump_json() + "\n")
            with (path / "scores.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(score.model_dump_json() + "\n")
            scores.append(score)
        manifest = {"schema_version": 1, "status": "complete", "kind": "demo", "task": "gsm8k",
                    "split": split_name, "protocol": "completion", "model_label": "scripted-demo",
                    "training_seed": 42, "not_a_benchmark": True, "protocol_signature": PROTOCOL_VERSION + "-demo",
                    "endpoint": {"model": "scripted"}, "data_hashes": {}, "contamination": {"status": "demo"},
                    "created_utc": datetime.now(timezone.utc).isoformat()}
        write_json(path / "manifest.json", manifest)
        make_report(path)
    print(output)


if __name__ == "__main__":
    raise SystemExit(main())
