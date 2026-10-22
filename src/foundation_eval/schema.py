"""Explicit, auditable contracts shared by data, collection and analysis.

The contracts are deliberately small: a task, a sample, a raw model response,
one scored record and a run manifest. There is no composite "capability score"
because no defensible weighting exists for these tasks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["demo", "train", "dev", "test", "diagnostic"]
Protocol = Literal["completion", "chat"]
ScoreStatus = Literal["ok", "extract_error", "unsupported", "execution_error", "judge_error", "error"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSpec(Contract):
    key: str
    name: str
    dataset_id: str
    revision: str
    config_name: str = "default"
    train_split: str | None = None
    test_split: str = "test"
    native_metric: str
    primary_protocol: Protocol = "completion"
    fewshot: int = 0
    max_new_tokens: int = 1024
    sample_count: int | None = None
    role: Literal["primary", "transfer", "auxiliary", "diagnostic"] = "auxiliary"
    notes: str = ""


class Sample(Contract):
    sample_id: str
    task: str
    split: Split
    prompt: str
    answer: str
    group_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def identity(self):
        if not self.sample_id or "/" in self.sample_id:
            raise ValueError("sample_id must be non-empty and contain no slash")
        return self


class ModelResponse(Contract):
    sample_id: str
    model_label: str
    protocol: Protocol
    text: str
    usage: dict[str, Any] = Field(default_factory=dict)
    elapsed_seconds: float = 0
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class Score(Contract):
    sample_id: str
    task: str
    metric: str
    score: float | None = None
    correct: bool | None = None
    status: ScoreStatus = "ok"
    detail: dict[str, Any] = Field(default_factory=dict)


class RunManifest(Contract):
    schema_version: int = 1
    kind: str
    status: Literal["running", "complete", "interrupted", "rejected"] = "running"
    created_utc: str
    git_head: str = ""
    source_hashes: dict[str, str] = Field(default_factory=dict)
    data_hashes: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    endpoint: dict[str, Any] = Field(default_factory=dict)
    protocol_signature: str = ""
    expected_records: int = 0
    actual_records: int = 0
    complete_records: int = 0
    notes: str = ""


class WeightRecord(Contract):
    schema_version: int = 1
    model_label: str
    base_model: str
    adapter_path: str
    adapter_sha256: str
    training_data_sha256: str
    training_config_sha256: str
    training_seed: int
    git_head: str
    created_utc: str
    eval_status: Literal["pending", "running", "complete"] = "pending"
    notes: str = ""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row.model_dump() if isinstance(row, BaseModel) else row,
                                       ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def wilson_interval(successes: int, n: int, z: float = 1.96):
    if n <= 0:
        return [None, None]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def validate_score(score: Score) -> Score:
    if score.status == "ok":
        if score.correct is None:
            raise ValueError("an ok score must decide correctness")
    elif score.correct is not None:
        raise ValueError("a non-ok score cannot claim a correctness decision")
    if score.correct is True and score.score not in {None, 1, 1.0}:
        raise ValueError("correct score must be 1")
    if score.correct is False and score.score not in {None, 0, 0.0}:
        raise ValueError("incorrect score must be 0")
    return score
