"""Offline contracts for measurement, selection, scoring and reporting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from foundation_eval.agent import ScriptedAdapter
from foundation_eval.data import (build_b1_b2, build_ruler_samples, contamination_report,
                                  extract_choice, extract_final_number, length_strata,
                                  prepare_gsm8k_split, select_subset)
from foundation_eval.judge import score_sample, verify_instruction
from foundation_eval.report import eval_matrix, make_report, paired_comparison, summarize_run
from foundation_eval.run import ROOT, load_config
from foundation_eval.schema import ModelResponse, Sample, Score, read_jsonl, write_json, write_jsonl
from foundation_eval.selection import make_selections, register_weight, training_export


def gsm_row(index: int):
    return {"id": str(index), "question": f"Question {index}: what is {index} + 1?",
            "answer": f"Add one.\n#### {index + 1}"}


def make_pool(n=60, wrong_every=2):
    samples = []
    for index in range(n):
        samples.append(Sample(sample_id=f"gsm8k-{index:03d}", task="gsm8k", split="train",
                              prompt=f"Question {index}: " + ("word " * (index % 7 + 1)) + f"what is {index} + 1?",
                              answer=f"Add one.\n#### {index + 1}", group_id=f"g{index // 5}"))
    diagnosis = {s.sample_id: (index % wrong_every != 0) for index, s in enumerate(samples)}
    return samples, diagnosis


def test_numeric_and_choice_extraction():
    assert extract_final_number("reasoning\n#### 1,024") == "1024"
    assert extract_final_number("so the answer is -3.") == "-3"
    assert extract_final_number("no number here") is None
    assert extract_choice("The answer is (C).", 4) == "C"
    assert extract_choice("A", 10) == "A"


def test_gsm8k_split_is_pinned_and_disjoint():
    rows = [gsm_row(i) for i in range(1200)]
    first = prepare_gsm8k_split(rows, 500, "salt")
    second = prepare_gsm8k_split(rows, 500, "salt")
    assert [s.sample_id for s in first["dev"]] == [s.sample_id for s in second["dev"]]
    assert len(first["dev"]) == 500 and len(first["train"]) == 700
    assert not ({s.sample_id for s in first["dev"]} & {s.sample_id for s in first["train"]})


def test_contamination_finds_exact_near_and_group_leaks():
    train = [Sample(sample_id="train-1", task="gsm8k", split="train", prompt="the quick brown fox jumps over the lazy dog",
                    answer="1", group_id="same")]
    test = [Sample(sample_id="test-1", task="gsm8k", split="test", prompt="the quick brown fox jumps over the lazy dog",
                   answer="1", group_id="same"),
            Sample(sample_id="test-2", task="gsm8k", split="test", prompt="the quick brown fox jumps over a lazy dog",
                   answer="1", group_id="other")]
    report = contamination_report({"train": train, "test": test})
    assert report["counts"]["exact"] == 1
    assert report["counts"]["groups"] == 1
    assert report["counts"]["near"] >= 0


def test_b1_b2_selection_is_matched_and_error_weighted():
    pool, diagnosis = make_pool(120, wrong_every=2)
    strata = length_strata(pool, 5)
    rows = build_b1_b2(pool, diagnosis, 20, [42, 43], strata, tolerance=0.01)
    for row in rows:
        assert row["b1"]["count"] == row["b2"]["count"] == 20
        assert len(set(row["b1"]["ids"])) == len(set(row["b2"]["ids"])) == 20
        assert row["budget"]["relative_gap"] <= 0.01
        assert row["b2"]["wrong_rate"] >= row["b1_wrong_rate"]
    assert rows[0]["b1"]["ids"] != rows[1]["b1"]["ids"]


def test_b2_requires_full_diagnosis():
    pool, diagnosis = make_pool(40)
    diagnosis.pop(next(iter(diagnosis)))
    with pytest.raises(ValueError, match="diagnosis for every pool"):
        build_b1_b2(pool, diagnosis, 8, [42], length_strata(pool, 4))


def test_ifeval_rules_and_unsupported_are_not_pass():
    assert verify_instruction("The sky is blue. It is clear.", "length_constraints:number_sentences",
                              {"relation": "exactly", "num_sentences": 2})[1] is True
    assert verify_instruction("one two three", "length_constraints:number_words",
                              {"relation": "more than", "num_words": 10})[1] is False
    assert verify_instruction('{"ok": true}', "detectable_format:json_format", {})[1] is True
    assert verify_instruction("a,b", "punctuation:no_comma", {})[1] is False
    supported, passed, _ = verify_instruction("x", "synthetic:unknown", {})
    assert supported is False and passed is False


def test_scoring_native_tasks():
    sample = Sample(sample_id="s1", task="gsm8k", split="test", prompt="q", answer="#### 42")
    response = ModelResponse(sample_id="s1", model_label="m", protocol="completion", text="reasoning\n#### 42")
    assert score_sample(sample, response, "exact_match_final_answer").correct is True
    wrong = response.model_copy(update={"text": "#### 41"})
    assert score_sample(sample, wrong, "exact_match_final_answer").correct is False
    broken = response.model_copy(update={"text": "no answer"})
    score = score_sample(sample, broken, "exact_match_final_answer")
    assert score.status == "extract_error" and score.correct is None

    choice = Sample(sample_id="m1", task="mmlu_pro", split="test", prompt="q",
                    answer="B", metadata={"options": 4})
    scored = score_sample(choice, ModelResponse(sample_id="m1", model_label="m", protocol="completion",
                                                text="B"), "exact_match_option")
    assert scored.correct is True
    bbh = Sample(sample_id="b1", task="bbh", split="test", prompt="q", answer="no")
    marked = score_sample(bbh, ModelResponse(sample_id="b1", model_label="m", protocol="completion",
                                             text="Reasoning.\nANSWER: no"), "task_exact_match")
    assert marked.correct is True


def test_code_scoring_local_mode(monkeypatch):
    monkeypatch.setenv("FOUNDATION_ALLOW_LOCAL_CODE", "1")
    sample = Sample(sample_id="c1", task="humaneval_plus", split="test", prompt="write add_one", answer="code",
                    metadata={"entry_point": "add_one",
                              "test_code": "def check(candidate):\n    assert candidate(1) == 2\n"})
    good = ModelResponse(sample_id="c1", model_label="m", protocol="completion",
                         text="```python\ndef add_one(x):\n    return x + 1\n```")
    assert score_sample(sample, good, "pass_at_1_functional", code_mode="local").correct is True
    bad = good.model_copy(update={"text": "```python\ndef add_one(x):\n    return x\n```"})
    assert score_sample(sample, bad, "pass_at_1_functional", code_mode="local").correct is False


def test_ruler_generation_is_fixed_length_and_seeded():
    first = build_ruler_samples(4096, "niah_single_1", 2, 2026)
    second = build_ruler_samples(4096, "niah_single_1", 2, 2026)
    assert [s.prompt for s in first] == [s.prompt for s in second]
    assert all(s.answer.isdigit() for s in first)
    with pytest.raises(ValueError, match="registered RULER"):
        build_ruler_samples(2048, "niah_single_1", 1, 2026)


def make_run(folder: Path, label: str, pattern: list[bool], seed=42, protocol="completion"):
    folder.mkdir(parents=True)
    scores = [Score(sample_id=f"s{i}", task="gsm8k", metric="exact_match_final_answer",
                    correct=value, status="ok", score=1.0 if value else 0.0)
              for i, value in enumerate(pattern)]
    write_jsonl(folder / "scores.jsonl", scores)
    responses = [{"sample_id": f"s{i}", "usage": {"total_tokens": 10}, "elapsed_seconds": 0.1}
                 for i in range(len(pattern))]
    write_jsonl(folder / "responses.jsonl", responses)
    write_json(folder / "manifest.json", {"status": "complete", "task": "gsm8k", "split": "test",
                                          "protocol": protocol, "model_label": label, "training_seed": seed,
                                          "protocol_signature": "sig", "not_a_benchmark": False,
                                          "benchmark_complete": True, "expected_records": len(pattern),
                                          "actual_records": len(pattern)})
    return folder


def test_bbh_uses_macro_average_not_sample_count():
    from foundation_eval.report import summarize
    scores = [Score(sample_id=f"a{i}", task="bbh", metric="task_exact_match", correct=True, status="ok", score=1.0,
                    detail={"subtask": "A"}) for i in range(3)]
    scores += [Score(sample_id="b0", task="bbh", metric="task_exact_match", correct=False, status="ok", score=0.0,
                     detail={"subtask": "B"})]
    result = summarize(scores, "bbh")
    assert result["accuracy"] == 0.75
    assert result["native_accuracy"] == 0.5
    assert result["subtask_n"] == 2


def test_report_keeps_unsupported_out_of_the_denominator(tmp_path):
    run = make_run(tmp_path / "run", "B0", [True, False, True])
    with (run / "scores.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(Score(sample_id="s3", task="gsm8k", metric="x", status="unsupported").model_dump_json() + "\n")
    payload = make_report(run)
    assert payload["native_metric"]["n"] == 4
    assert payload["native_metric"]["decided"] == 3
    assert payload["native_metric"]["accuracy"] == 2 / 3
    assert payload["native_metric"]["status"]["unsupported"] == 1
    assert "composite" not in payload


def test_paired_comparison_requires_complete_matched_runs(tmp_path):
    a = make_run(tmp_path / "a", "B1", [False] * 10)
    b = make_run(tmp_path / "b", "B2", [True] * 7 + [False] * 3)
    result = paired_comparison([a], [b], samples=200)
    assert result["difference"] == pytest.approx(0.7)
    assert result["ci95"][0] > 0
    with pytest.raises(ValueError, match="mismatch"):
        paired_comparison([a], [make_run(tmp_path / "c", "B2", [True] * 10, protocol="chat")])
    incomplete = make_run(tmp_path / "d", "B2", [True] * 10)
    manifest = json.loads((incomplete / "manifest.json").read_text(encoding="utf-8"))
    manifest["benchmark_complete"] = False
    write_json(incomplete / "manifest.json", manifest)
    with pytest.raises(ValueError, match="benchmark-complete"):
        paired_comparison([a], [incomplete])


def test_eval_matrix_and_summary(tmp_path):
    a = make_run(tmp_path / "a", "B1", [True, False])
    b = make_run(tmp_path / "b", "B2", [True, True])
    rows = eval_matrix([a, b], tmp_path / "matrix")
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"B1", "B2"}
    assert summarize_run(a)["accuracy"] == 0.5


def test_sft_export_and_llamafactory_config(tmp_path):
    pool, diagnosis = make_pool(30)
    from foundation_eval.data import export_sft, write_llamafactory_config
    samples = pool[:6]
    count = export_sft(samples, tmp_path / "sft.jsonl", "solve")
    assert count == 6
    config = load_config()[0]
    write_llamafactory_config(42, tmp_path / "sft.jsonl", tmp_path / "cfg", "Qwen/Qwen3-8B-Base", config)
    training = yaml.safe_load((tmp_path / "cfg/seed42.yaml").read_text(encoding="utf-8"))
    assert training["lora_rank"] == 32 and training["learning_rate"] == 1e-4
    assert training["per_device_train_batch_size"] * training["gradient_accumulation_steps"] * 4 == 16


def test_scripted_adapter_is_explicitly_not_a_benchmark():
    sample = Sample(sample_id="s1", task="gsm8k", split="test", prompt="q", answer="#### 1")
    adapter = ScriptedAdapter({"s1": "#### 1"})
    response = adapter.generate(sample, "completion", [], "solve")
    assert response.usage == {"scripted": True}


def test_end_to_end_selection_and_training_export(tmp_path):
    import copy
    config = copy.deepcopy(load_config()[0])
    config.update({"selection_count": 8, "seeds": [42], "length_bins": 4, "token_tolerance": 0.01})
    pool, diagnosis = make_pool(80, wrong_every=2)
    pool_path = tmp_path / "pool.jsonl"
    write_jsonl(pool_path, pool)
    diagnosis_path = tmp_path / "diagnosis.jsonl"
    write_jsonl(diagnosis_path, [{"sample_id": key, "status": "ok", "correct": value,
                                  "task": "gsm8k"} for key, value in diagnosis.items()])
    selection_dir = tmp_path / "selection"
    manifest = make_selections(pool_path, diagnosis_path, config, selection_dir)
    assert len(manifest["seeds"]) == 1
    assert manifest["seeds"][0]["budget"]["relative_gap"] <= 0.01
    plan = training_export(selection_dir, config, tmp_path / "training")
    assert len(plan) == 4  # B1/B2 x 8B/4B
    assert all((Path(row["config_dir"]) / "seed42.yaml").exists() for row in plan)


def test_weight_registration_hashes_adapter_and_refuses_duplicate(tmp_path):
    from foundation_eval.schema import WeightRecord
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"r": 32}', encoding="utf-8")
    record = WeightRecord(model_label="B1-s42", base_model="Qwen/Qwen3-8B-Base",
                          adapter_path=str(adapter), adapter_sha256="pending",
                          training_data_sha256="data", training_config_sha256="config",
                          training_seed=42, git_head="test", created_utc="2026-10-22T00:00:00Z")
    saved = register_weight(tmp_path / "weights.jsonl", record)
    assert len(saved.adapter_sha256) == 64
    with pytest.raises(ValueError, match="already registered"):
        register_weight(tmp_path / "weights.jsonl", record)


def test_load_config_registers_primary_and_transfer():
    config, tasks, models = load_config()
    assert tasks["gsm8k"]["role"] == "primary" and tasks["svamp"]["role"] == "transfer"
    assert config["primary_protocol"] == "completion"
    assert models["B2"]["base_model"] == "Qwen/Qwen3-8B-Base"
    assert config["b2"]["wrong_weight"] == 3
