"""Task data, deterministic splits, contamination checks and LightSFT selection.

The central claim of this project is not "more data helps". It is that a
base-model error diagnosis can allocate a fixed sample/token budget better than
a length-matched random draw. This module therefore makes every allocation
decision explicit and reproducible, and refuses to select without a recorded
diagnosis for B2.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .schema import Sample, digest, read_jsonl, write_json, write_jsonl

WORD_RE = re.compile(r"[A-Za-z0-9_]+")
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
CHOICE_RE = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?", re.I)
FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return WORD_RE.findall(normalize_text(text))


def token_length(text: str, tokenizer: Callable[[str], list[int]] | None = None) -> int:
    if tokenizer is not None:
        return len(tokenizer(text))
    # Conservative offline approximation; every manifest records the counter.
    return max(1, len(text) // 4)


def extract_final_number(text: str) -> str | None:
    """Prefer GSM8K's explicit #### marker; otherwise take the last number."""
    if not text:
        return None
    marked = re.findall(r"####\s*([^\n]+)", text)
    if marked:
        value = marked[-1].strip()
        number = NUMBER_RE.search(value)
        return _canonical_number(number.group(0)) if number else value
    matches = NUMBER_RE.findall(text.replace(",", ""))
    return _canonical_number(matches[-1]) if matches else None


def _canonical_number(value: str) -> str:
    value = value.replace(",", "").replace("$", "").strip()
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:.10g}"


def extract_choice(text: str, options: int = 10) -> str | None:
    if not text:
        return None
    matches = CHOICE_RE.findall(text)
    if matches:
        return matches[-1].upper()
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    loose = re.findall(r"\b([A-J])\b", tail.upper())
    return loose[-1] if loose else None


def extract_code(text: str) -> str:
    match = FENCE_RE.search(text or "")
    return (match.group(1) if match else (text or "")).strip() + "\n"


def make_sample_id(task: str, raw_id, prompt: str) -> str:
    if raw_id is not None:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw_id)).strip("-")
        if cleaned:
            return f"{task}-{cleaned}"
    return f"{task}-{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"


def prepare_gsm8k_split(rows: Iterable[dict], dev_count: int = 500, salt: str = "gsm8k-dev-v1",
                        tokenizer=None) -> dict[str, list[Sample]]:
    """Fixed-hash dev holdout from the official GSM8K train split only."""
    train = []
    for row in rows:
        prompt = row.get("question") or row.get("prompt") or ""
        answer = row.get("answer") or row.get("solution") or ""
        if not prompt or not answer:
            raise ValueError("GSM8K rows need question and answer")
        sample = Sample(sample_id=make_sample_id("gsm8k", row.get("id"), prompt), task="gsm8k",
                        split="train", prompt=prompt, answer=answer, group_id=row.get("group_id"))
        train.append(sample)
    if len({s.sample_id for s in train}) != len(train):
        raise ValueError("duplicate GSM8K train identities")
    if not 0 < dev_count < len(train):
        raise ValueError("dev_count must be smaller than the GSM8K train pool")
    ordered = sorted(train, key=lambda s: hashlib.sha256(f"{salt}:{s.sample_id}".encode()).hexdigest())
    dev_ids = {s.sample_id for s in ordered[:dev_count]}
    dev = [s.model_copy(update={"split": "dev"}) for s in ordered[:dev_count]]
    pool = [s for s in ordered if s.sample_id not in dev_ids]
    if {s.sample_id for s in dev} & {s.sample_id for s in pool}:
        raise ValueError("GSM8K dev/train overlap")
    return {"dev": dev, "train": pool}


def load_samples(path: Path, task: str | None = None, split: str | None = None) -> list[Sample]:
    rows = read_jsonl(path)
    samples = []
    for row in rows:
        if task and row.get("task") not in {None, task}:
            continue
        if split and row.get("split") not in {None, split}:
            continue
        samples.append(Sample.model_validate(row))
    return samples


def save_samples(path: Path, samples: list[Sample]):
    write_jsonl(path, samples)


def length_strata(samples: list[Sample], bins: int = 10, tokenizer=None) -> dict[str, int]:
    """Quantile strata over the training pool (never over test)."""
    if not samples:
        raise ValueError("cannot stratify an empty pool")
    lengths = sorted(token_length(s.prompt + "\n" + s.answer, tokenizer) for s in samples)
    cuts = []
    for index in range(1, bins):
        cuts.append(lengths[min(len(lengths) - 1, int(len(lengths) * index / bins))])
    result = {}
    for sample in samples:
        length = token_length(sample.prompt + "\n" + sample.answer, tokenizer)
        stratum = sum(length > cut for cut in cuts)
        result[sample.sample_id] = min(stratum, bins - 1)
    return result


def _allocate(groups: dict[int, list[Sample]], count: int) -> dict[int, int]:
    total = sum(len(v) for v in groups.values())
    if count > total:
        raise ValueError("requested sample count exceeds the training pool")
    raw = {key: count * len(group) / total for key, group in groups.items()}
    alloc = {key: min(int(value), len(groups[key])) for key, value in raw.items()}
    # Redistribute any remainder to strata that still have capacity.
    while sum(alloc.values()) < count:
        available = [key for key in groups if alloc[key] < len(groups[key])]
        if not available:
            raise ValueError("cannot allocate the requested count across strata")
        key = max(available, key=lambda k: (raw[k] - alloc[k], -k))
        alloc[key] += 1
    return alloc


def weighted_sample(candidates: list[Sample], count: int, weights: dict[str, float], rng: random.Random) -> list[Sample]:
    """Efraimidis-Spirakis weighted sampling without replacement, one draw."""
    if count > len(candidates):
        raise ValueError("cannot draw more samples than the stratum contains")
    keyed = []
    for sample in candidates:
        weight = max(1e-9, float(weights.get(sample.sample_id, 1.0)))
        key = math.log(max(rng.random(), 1e-12)) / weight
        keyed.append((key, sample.sample_id, sample))
    keyed.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in keyed[:count]]


def select_subset(pool: list[Sample], count: int, seed: int, strata: dict[str, int],
                  weights: dict[str, float] | None = None) -> list[Sample]:
    if len({s.sample_id for s in pool}) != len(pool):
        raise ValueError("duplicate samples in pool")
    weights = weights or {}
    rng = random.Random(seed)
    groups: dict[int, list[Sample]] = defaultdict(list)
    for sample in pool:
        if sample.sample_id not in strata:
            raise ValueError("every pool sample needs a length stratum")
        groups[strata[sample.sample_id]].append(sample)
    alloc = _allocate(groups, count)
    selected = []
    for stratum in sorted(groups):
        selected.extend(weighted_sample(groups[stratum], alloc[stratum], weights, rng))
    if len(selected) != count or len({s.sample_id for s in selected}) != count:
        raise ValueError("selection did not produce distinct samples")
    return selected


def _tokens(samples: list[Sample], tokenizer=None) -> int:
    return sum(token_length(s.prompt + "\n" + s.answer, tokenizer) for s in samples)


def match_token_budget(a: list[Sample], b: list[Sample], pool: list[Sample], strata: dict[str, int],
                       tokenizer=None, tolerance: float = 0.01, max_swaps: int = 2000):
    """Swap same-stratum items so B's total tokens stay within 1% of A's.

    The two arms keep exactly the registered count. Swapping never changes the
    length stratum, so matching does not undo the length-stratified design.
    """
    a = list(a)
    b = list(b)
    a_ids, b_ids = {s.sample_id for s in a}, {s.sample_id for s in b}
    by_id = {s.sample_id: s for s in pool}
    if not (a_ids | b_ids) <= set(by_id):
        raise ValueError("selection contains samples outside the pool")
    a_tokens, b_tokens = _tokens(a, tokenizer), _tokens(b, tokenizer)
    swaps = 0
    while a_tokens and abs(b_tokens - a_tokens) / a_tokens > tolerance and swaps < max_swaps:
        larger, smaller = (b, a) if b_tokens > a_tokens else (a, b)
        larger_ids = {s.sample_id for s in larger}
        chosen_ids = a_ids | b_ids
        candidates = [s for s in pool if s.sample_id not in chosen_ids]
        best = None
        # Greedy best-improvement swap within the same stratum.
        for index, old in enumerate(larger):
            stratum = strata[old.sample_id]
            for new in candidates:
                if strata[new.sample_id] != stratum:
                    continue
                delta = (token_length(old.prompt + "\n" + old.answer, tokenizer)
                         - token_length(new.prompt + "\n" + new.answer, tokenizer))
                if delta <= 0:
                    continue
                new_gap = abs((b_tokens - delta if larger is b else b_tokens) -
                              (a_tokens - delta if larger is a else a_tokens))
                if best is None or new_gap < best[0]:
                    best = (new_gap, index, new)
            if best is not None and best[0] == 0:
                break
        if best is None:
            break
        _, index, replacement = best
        old = larger[index]
        larger[index] = replacement
        if larger is b:
            b_ids.remove(old.sample_id)
            b_ids.add(replacement.sample_id)
        else:
            a_ids.remove(old.sample_id)
            a_ids.add(replacement.sample_id)
        a_tokens, b_tokens = _tokens(a, tokenizer), _tokens(b, tokenizer)
        swaps += 1
    gap = abs(b_tokens - a_tokens) / a_tokens if a_tokens else 0
    if gap > tolerance:
        raise ValueError(f"cannot match token budgets within {tolerance:.1%}: {a_tokens} vs {b_tokens} ({gap:.3%})")
    return a, b, {"a_tokens": a_tokens, "b_tokens": b_tokens, "relative_gap": gap, "swaps": swaps}


def selection_record(name: str, selected: list[Sample], pool: list[Sample], seed: int, strata: dict[str, int],
                     weights: dict[str, float] | None = None, tokenizer=None, extra: dict | None = None) -> dict:
    weights = weights or {}
    per_stratum = Counter(strata[s.sample_id] for s in selected)
    pool_per_stratum = Counter(strata[s.sample_id] for s in pool)
    return {
        "name": name,
        "seed": seed,
        "count": len(selected),
        "tokens": _tokens(selected, tokenizer),
        "ids": [s.sample_id for s in selected],
        "ids_sha256": digest([s.sample_id for s in selected]),
        "per_stratum": dict(sorted(per_stratum.items())),
        "pool_per_stratum": dict(sorted(pool_per_stratum.items())),
        "weighted": any(v != 1 for v in weights.values()) if weights else False,
        "max_weight": max(weights.values()) if weights else 1,
        **(extra or {}),
    }


def build_b1_b2(pool: list[Sample], diagnosis: dict[str, bool], count: int, seeds: Iterable[int],
                strata: dict[str, int], tokenizer=None, tolerance: float = 0.01,
                wrong_weight: float = 3.0, correct_weight: float = 1.0):
    """Build paired B1/B2 selections for each seed.

    `diagnosis` maps training-pool sample_id -> whether the base model answered
    correctly. B2 requires a decision for every pool sample; missing labels are
    an error rather than an implicit "correct".
    """
    pool_ids = {s.sample_id for s in pool}
    missing = sorted(pool_ids - set(diagnosis))
    if missing:
        raise ValueError(f"B2 requires a diagnosis for every pool sample; {len(missing)} missing (first: {missing[0]})")
    outputs = []
    for seed in seeds:
        b1 = select_subset(pool, count, seed, strata, {})
        weights = {s.sample_id: (wrong_weight if diagnosis[s.sample_id] is False else correct_weight) for s in pool}
        b2 = select_subset(pool, count, seed, strata, weights)
        b1, b2, budget = match_token_budget(b1, b2, pool, strata, tokenizer, tolerance=tolerance)
        wrong_in_b2 = sum(diagnosis[s.sample_id] is False for s in b2)
        wrong_in_b1 = sum(diagnosis[s.sample_id] is False for s in b1)
        outputs.append({
            "seed": seed,
            "b1": selection_record("B1", b1, pool, seed, strata, tokenizer=tokenizer),
            "b2": selection_record("B2", b2, pool, seed, strata, tokenizer=tokenizer,
                                   extra={"wrong_samples": wrong_in_b2, "wrong_rate": wrong_in_b2 / len(b2)}),
            "b1_wrong_samples": wrong_in_b1,
            "b1_wrong_rate": wrong_in_b1 / len(b1),
            "budget": budget,
            "overlap_with_other_arm": len({s.sample_id for s in b1} & {s.sample_id for s in b2}),
            "diagnosed_pool": len(diagnosis),
            "correct_pool": sum(diagnosis.values()),
        })
    return outputs


def contamination_report(splits: dict[str, list[Sample]], fewshot: list[Sample] | None = None,
                          threshold: float = 0.8):
    """Exact, near-duplicate (5-word shingles, Jaccard) and same-group leakage.

    This checks only data visible to this project. It cannot prove that a public
    pretraining corpus never saw a benchmark item.
    """
    fewshot = fewshot or []
    exact: dict[str, list[str]] = defaultdict(list)
    shingles: dict[str, set[str]] = {}
    index: dict[str, list[str]] = defaultdict(list)
    owners: dict[str, str] = {}
    for split, samples in {**splits, "fewshot": fewshot}.items():
        for sample in samples:
            if sample.sample_id in owners:
                raise ValueError("duplicate sample across splits")
            owners[sample.sample_id] = split
            key = hashlib.sha256(normalize_text(sample.prompt).encode()).hexdigest()
            exact[key].append(sample.sample_id)
            sh = {" ".join(gram) for gram in zip(*[words(sample.prompt)[i:] for i in range(5)])}
            shingles[sample.sample_id] = sh
            for gram in sh:
                index[gram].append(sample.sample_id)
    exact_pairs = []
    for ids in exact.values():
        if len({owners[i] for i in ids}) > 1:
            exact_pairs.append(sorted(ids))
    near_pairs = []
    seen = set()
    for sample_id, sh in shingles.items():
        if not sh:
            continue
        counts = Counter()
        for gram in sh:
            counts.update(index[gram])
        for other, overlap in counts.items():
            if other <= sample_id or owners[other] == owners[sample_id]:
                continue
            pair = tuple(sorted((sample_id, other)))
            if pair in seen:
                continue
            union = len(sh | shingles[other])
            if union and overlap / union >= threshold:
                near_pairs.append({"a": pair[0], "b": pair[1], "jaccard": overlap / union,
                                   "splits": sorted((owners[pair[0]], owners[pair[1]]))})
            seen.add(pair)
    group_sets = {split: {s.group_id for s in samples if s.group_id} for split, samples in splits.items()}
    leaked_groups = []
    names = sorted(group_sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            shared = group_sets[left] & group_sets[right]
            if shared:
                leaked_groups.append({"splits": [left, right], "groups": sorted(shared)[:20],
                                      "count": len(shared)})
    report = {"threshold": threshold, "shingle_size": 5,
              "exact": exact_pairs, "near": near_pairs, "groups": leaked_groups,
              "counts": {"exact": len(exact_pairs), "near": len(near_pairs), "groups": len(leaked_groups)},
              "scope_note": "Only project-visible data is checked; no claim about pretraining contamination."}
    return report


def assert_no_contamination(report: dict):
    if any(report["counts"].values()):
        raise ValueError(f"contamination findings: {report['counts']}")
    return report


def export_sft(samples: list[Sample], output: Path, system_prompt: str):
    if output.exists():
        raise ValueError("SFT export already exists")
    if len({s.sample_id for s in samples}) != len(samples):
        raise ValueError("duplicate SFT samples")
    rows = []
    for sample in samples:
        rows.append({"messages": [{"role": "system", "content": system_prompt},
                                  {"role": "user", "content": sample.prompt},
                                  {"role": "assistant", "content": sample.answer}],
                     "metadata": {"sample_id": sample.sample_id, "task": sample.task}})
    write_jsonl(output, [{"messages": row["messages"]} for row in rows])
    write_jsonl(output.with_suffix(".provenance.jsonl"), [{"sample_id": row["metadata"]["sample_id"],
                                                           "task": row["metadata"]["task"],
                                                           "prompt_sha256": digest(row["messages"][1]["content"])}
                                                          for row in rows])
    write_json(output.with_suffix(".manifest.json"), {"rows": len(rows), "ids_sha256": digest([r["metadata"]["sample_id"] for r in rows]),
                                                      "targets": "gsm8k-completion-sft"})
    return len(rows)


def write_llamafactory_config(seed: int, data_path: Path, output: Path, base_model: str, config: dict):
    if output.exists():
        raise ValueError("refusing to overwrite an existing training config")
    sft = config["sft"]
    training = {
        "model_name_or_path": base_model,
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": sft["lora_rank"],
        "lora_alpha": sft["lora_alpha"],
        "lora_dropout": sft["lora_dropout"],
        "dataset": "gsm8k_light_sft",
        "dataset_dir": str(data_path.parent.resolve()),
        "template": sft["template"],
        "cutoff_len": sft["cutoff_len"],
        "overwrite_cache": False,
        "preprocessing_num_workers": 8,
        "output_dir": f"outputs/{Path(base_model).name.lower()}-{data_path.stem}-seed{seed}",
        "logging_steps": 10,
        "save_steps": 100,
        "plot_loss": True,
        "overwrite_output_dir": False,
        "per_device_train_batch_size": sft["per_device_batch"],
        "gradient_accumulation_steps": sft["gradient_accumulation"],
        "learning_rate": sft["learning_rate"],
        "num_train_epochs": sft["epochs"],
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "bf16": sft["bf16"],
        "ddp_timeout": 180000000,
        "report_to": "none",
        "seed": seed,
        "data_seed": seed,
    }
    write_json(output / f"seed{seed}.yaml", training)
    dataset_info = {"gsm8k_light_sft": {"file_name": data_path.name, "formatting": "sharegpt",
                                        "columns": {"messages": "messages"}}}
    write_json(data_path.parent / "dataset_info.json", dataset_info)
    write_json(output / "dataset_info.json", dataset_info)
    return training


def sample_from_row(row: dict) -> Sample:
    return Sample.model_validate(row)


def load_diagnosis(path: Path) -> dict[str, bool]:
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError("diagnosis contains a non-ok score; fix the scorer before selection")
        result[row["sample_id"]] = bool(row["correct"])
    return result


def _pad_to_length(needle: str, filler: str, target: int, tokenizer=None) -> str:
    """Grow text geometrically, then trim to a small tolerance."""
    body = needle
    block = filler
    while token_length(body, tokenizer) < target:
        body += block
        if token_length(body, tokenizer) < target * 0.75:
            block = block + block
    while token_length(body, tokenizer) > target * 1.05 and len(body) > len(needle) * 2:
        body = body[: int(len(body) * 0.98)]
    return body


def build_ruler_samples(length: int, task: str, count: int, seed: int, tokenizer=None) -> list[Sample]:
    """Synthetic fixed-length diagnostics, not an official RULER submission."""
    if length not in {4096, 8192, 16384}:
        raise ValueError("registered RULER diagnostic lengths are 4096/8192/16384")
    if task not in {"niah_single_1", "niah_multikey_1", "vt"}:
        raise ValueError("registered RULER diagnostic tasks are niah_single_1/niah_multikey_1/vt")
    rng = random.Random(f"{seed}:{length}:{task}")
    filler = ("The archive records routine inventory checks, delivery notes, and supplier messages from the "
              "regional operations team. ")
    samples = []
    for index in range(count):
        number = rng.randint(1000, 9999)
        if task == "niah_single_1":
            needle = f"The magic number is {number}."
            question = "What is the magic number? Answer with the number."
            answer = str(number)
        elif task == "niah_multikey_1":
            keys = ["alpha", "beta", "gamma"]
            values = {key: rng.randint(1000, 9999) for key in keys}
            needle = " ".join(f"The {key} key is {value}." for key, value in values.items())
            question = f"What is the beta key? Answer with the number."
            answer = str(values["beta"])
        else:
            start = rng.randint(10, 99)
            chain = [f"A = {start}"]
            names = ["B", "C", "D", "E"]
            for previous, name in zip(["A", "B", "C", "D"], names):
                chain.append(f"{name} = {previous} + {rng.randint(1, 9)}")
            needle = "; ".join(chain) + "."
            question = "What is the final value of E? Answer with the number."
            value = start
            for step in chain[1:]:
                value += int(step.split("+")[-1])
            answer = str(value)
        body = _pad_to_length(needle, filler, length, tokenizer)
        position = rng.randint(0, max(0, len(body) - len(needle)))
        body = body[:position] + needle + body[position + len(needle):]
        prompt = f"{body}\n\nQuestion: {question}"
        samples.append(Sample(sample_id=f"ruler-{task}-{length}-{index:03d}", task="ruler", split="diagnostic",
                              prompt=prompt, answer=answer, metadata={"target_length": length, "task": task,
                                                                      "seed": seed, "token_counter": "hf" if tokenizer else "chars/4"}))
    return samples
