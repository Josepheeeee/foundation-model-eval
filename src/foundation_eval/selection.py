"""LightSFT selection, LlamaFactory export and weight registration.

B1 and B2 are a controlled pair: same pool, same count, same length strata,
same token budget within 1%, different sampling weights. B2 is impossible
without a complete base-model diagnosis of the training pool.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .data import (build_b1_b2, export_sft, length_strata, load_diagnosis, load_samples,
                   save_samples, write_llamafactory_config)
from .schema import WeightRecord, digest, file_digest, write_json, write_jsonl

SYSTEM_PROMPT = ("Solve the math problem carefully. Show the reasoning. "
                 "Put the final numeric answer after a line that starts with ####.")


def make_selections(pool_path: Path, diagnosis_path: Path, config: dict, output: Path, tokenizer=None,
                    tokenizer_name: str = "chars/4"):
    if output.exists() and any(output.iterdir()):
        raise ValueError("selection output directory already has artifacts")
    output.mkdir(parents=True, exist_ok=True)
    pool = load_samples(pool_path, task="gsm8k")
    if any(s.split != "train" for s in pool):
        raise ValueError("selection pool must contain only the GSM8K train split")
    diagnosis = load_diagnosis(diagnosis_path)
    strata = length_strata(pool, config["length_bins"], tokenizer)
    results = build_b1_b2(pool, diagnosis, config["selection_count"], config["seeds"], strata, tokenizer,
                          tolerance=config["token_tolerance"],
                          wrong_weight=config["b2"]["wrong_weight"],
                          correct_weight=config["b2"]["correct_weight"])
    by_id = {s.sample_id: s for s in pool}
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "pool": str(pool_path), "pool_sha256": file_digest(pool_path),
                "diagnosis": str(diagnosis_path), "diagnosis_sha256": file_digest(diagnosis_path),
                "count": config["selection_count"], "length_bins": config["length_bins"],
                "token_tolerance": config["token_tolerance"], "token_counter": tokenizer_name,
                "seeds": results}
    write_json(output / "selection.json", manifest)
    for row in results:
        for arm in ("b1", "b2"):
            ids = row[arm]["ids"]
            save_samples(output / f"{arm}_seed{row['seed']}.jsonl", [by_id[i] for i in ids])
            write_json(output / f"{arm}_seed{row['seed']}.record.json", row[arm])
    return manifest


def training_export(selection_dir: Path, config: dict, output: Path):
    if output.exists() and any(output.iterdir()):
        raise ValueError("training export directory already has artifacts")
    output.mkdir(parents=True, exist_ok=True)
    selected = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    plan = []
    for seed_row in selected["seeds"]:
        seed = seed_row["seed"]
        for arm, base_key in (("b1", "B1"), ("b2", "B2")):
            samples_path = selection_dir / f"{arm}_seed{seed}.jsonl"
            samples = load_samples(samples_path, task="gsm8k")
            sft_path = output / f"{arm}_seed{seed}.jsonl"
            rows = export_sft(samples, sft_path, SYSTEM_PROMPT)
            for family, base_model in (("8b", config["sft"]["base_model"]),
                                       ("4b", config["sft"]["replication_model"])):
                config_dir = output / f"{arm}_seed{seed}_{family}"
                training = write_llamafactory_config(seed, sft_path, config_dir, base_model, config)
                plan.append({"arm": arm, "seed": seed, "family": family, "base_model": base_model,
                             "rows": rows, "data_sha256": digest([s.sample_id for s in samples]),
                             "config": training, "config_dir": str(config_dir),
                             "status": "pending"})
    write_json(output / "training_matrix.json", {"schema_version": 1, "plan": plan})
    return plan


def _artifact_digest(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return file_digest(path)
    if path.is_dir():
        return digest({str(file.relative_to(path)).replace("\\", "/"): file_digest(file)
                       for file in sorted(path.rglob("*")) if file.is_file()})
    raise ValueError("adapter path does not exist")


def register_weight(path: Path, record: WeightRecord):
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(row.get("model_label") == record.model_label for row in existing):
        raise ValueError("weight label already registered; keep the existing artifact")
    record = record.model_copy(update={"adapter_sha256": _artifact_digest(Path(record.adapter_path))})
    existing.append(record.model_dump())
    write_jsonl(path, existing)
    return record
