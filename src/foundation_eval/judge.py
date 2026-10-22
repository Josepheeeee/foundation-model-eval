"""Deterministic scoring first; an LLM judge is only an auxiliary error analyst.

Native metrics are never replaced by a model's opinion. Unsupported IFEval
instructions and failed code execution are reported as such, never as passes.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .data import extract_choice, extract_code, extract_final_number, normalize_text, words
from .schema import ModelResponse, Sample, Score, validate_score


def _score(sample: Sample, metric: str, correct: bool | None, status="ok", detail=None, score=None) -> Score:
    return validate_score(Score(sample_id=sample.sample_id, task=sample.task, metric=metric,
                                correct=correct, status=status, detail=detail or {},
                                score=(1.0 if correct else 0.0) if correct is not None and score is None else score))


def _numeric(sample: Sample, response: ModelResponse, metric: str) -> Score:
    predicted = extract_final_number(response.text)
    expected = extract_final_number(sample.answer)
    if predicted is None or expected is None:
        return _score(sample, metric, None, "extract_error",
                      {"predicted": predicted, "expected": expected, "raw_tail": response.text[-200:]})
    return _score(sample, metric, predicted == expected, detail={"predicted": predicted, "expected": expected})


def _choice(sample: Sample, response: ModelResponse, metric: str) -> Score:
    predicted = extract_choice(response.text, int(sample.metadata.get("options", 10)))
    expected = extract_choice(sample.answer, int(sample.metadata.get("options", 10)))
    if predicted is None or expected is None:
        return _score(sample, metric, None, "extract_error",
                      {"predicted": predicted, "expected": expected})
    return _score(sample, metric, predicted == expected, detail={"predicted": predicted, "expected": expected})


def _bbh(sample: Sample, response: ModelResponse, metric: str) -> Score:
    expected = sample.answer.strip()
    if re.fullmatch(r"[A-J]", expected, re.I):
        return _choice(sample, response, metric)
    predicted = normalize_text(response.text)
    target = normalize_text(expected)
    return _score(sample, metric, predicted == target, detail={"predicted": predicted, "expected": target})


def _count_sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def _relation(value: int, relation: str, target: int) -> bool:
    relation = relation.lower()
    if relation in {"at least", ">="}:
        return value >= target
    if relation in {"less than", "<"}:
        return value < target
    if relation in {"more than", ">"}:
        return value > target
    if relation in {"exactly", "=="}:
        return value == target
    raise ValueError(f"unsupported relation: {relation}")


def verify_instruction(text: str, instruction_id: str, kwargs: dict):
    """Return (supported, passed, reason)."""
    lower = text.lower()
    try:
        if instruction_id == "length_constraints:number_words":
            return True, _relation(len(words(text)), kwargs["relation"], int(kwargs["num_words"])), "word count"
        if instruction_id == "length_constraints:number_sentences":
            return True, _relation(_count_sentences(text), kwargs["relation"], int(kwargs["num_sentences"])), "sentence count"
        if instruction_id == "length_constraints:number_paragraphs":
            count = len([p for p in re.split(r"\n\s*\n", text.strip()) if p])
            return True, _relation(count, kwargs["relation"], int(kwargs["num_paragraphs"])), "paragraph count"
        if instruction_id == "keywords:existence":
            return True, all(keyword.lower() in lower for keyword in kwargs["keywords"]), "keywords present"
        if instruction_id == "keywords:frequency":
            count = lower.count(kwargs["keyword"].lower())
            return True, _relation(count, kwargs["relation"], int(kwargs["frequency"])), "keyword frequency"
        if instruction_id == "keywords:forbidden_words":
            return True, all(word.lower() not in lower for word in kwargs["forbidden_words"]), "forbidden words absent"
        if instruction_id == "detectable_format:title":
            return True, bool(re.search(r"<<[^<>]+>>", text)), "title in double angle brackets"
        if instruction_id == "detectable_format:json_format":
            value = json.loads(text.strip())
            return True, isinstance(value, dict), "valid JSON object"
        if instruction_id == "detectable_format:multiple_sections":
            splitter = kwargs.get("section_spliter", "SECTION")
            count = text.count(splitter)
            return True, count >= int(kwargs.get("num_sections", 1)), "section count"
        if instruction_id == "detectable_format:number_bullet_lists":
            count = len(re.findall(r"^\s*[\*\-]\s+", text, re.M))
            return True, count >= int(kwargs["num_bullets"]), "bullet count"
        if instruction_id == "detectable_format:numbered_list":
            count = len(re.findall(r"^\s*\d+[\.\)]\s+", text, re.M))
            return True, count >= int(kwargs.get("num_numbers", kwargs.get("num_bullets", 1))), "numbered list count"
        if instruction_id == "detectable_format:highlighted_sections":
            count = len(re.findall(r"\*[^*\n]+\*", text))
            return True, count >= int(kwargs.get("num_highlights", 1)), "highlight count"
        if instruction_id == "detectable_content:number_placeholders":
            count = len(re.findall(r"\[[^\[\]]+\]", text))
            return True, count >= int(kwargs["num_placeholders"]), "placeholder count"
        if instruction_id == "detectable_content:postscript":
            return True, "P.S." in text or "PS:" in text, "postscript present"
        if instruction_id == "startend:end_checker":
            return True, text.strip().endswith(kwargs["end_phrase"]), "end phrase"
        if instruction_id == "change_case:english_lowercase":
            return True, not re.search(r"[A-Z]", text), "no uppercase letters"
        if instruction_id == "change_case:english_uppercase":
            return True, not re.search(r"[a-z]", text), "no lowercase letters"
        if instruction_id == "punctuation:no_comma":
            return True, "," not in text, "no comma"
        if instruction_id == "combination:repeat_prompt":
            return True, normalize_text(kwargs.get("prompt", "")) in normalize_text(text), "prompt repeated"
        return False, False, "unsupported instruction type"
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return True, False, f"rule error: {type(exc).__name__}"


def _ifeval(sample: Sample, response: ModelResponse, metric: str) -> Score:
    ids = sample.metadata.get("instruction_id_list") or []
    kwargs_list = sample.metadata.get("kwargs") or [{} for _ in ids]
    if len(ids) != len(kwargs_list):
        return _score(sample, metric, None, "unsupported", {"reason": "instruction/kwargs length mismatch"})
    checks = []
    for instruction_id, kwargs in zip(ids, kwargs_list):
        supported, passed, reason = verify_instruction(response.text, instruction_id, kwargs)
        checks.append({"instruction": instruction_id, "supported": supported, "passed": passed, "reason": reason})
    if any(not check["supported"] for check in checks):
        return _score(sample, metric, None, "unsupported", {"checks": checks})
    correct = all(check["passed"] for check in checks)
    return _score(sample, metric, correct, detail={"checks": checks})


def run_code_tests(code: str, test_code: str, entry_point: str, mode: str = "docker", timeout: int = 15):
    """Execute generated code with no network and bounded resources.

    `docker` is the default. `local` is for development only and is refused on
    Windows unless FOUNDATION_ALLOW_LOCAL_CODE=1 is explicitly set.
    """
    if not entry_point or "def check" not in test_code:
        return {"passed": False, "status": "execution_error", "reason": "test code must define check(candidate)"}
    harness = f"""{code}

{test_code}

if __name__ == "__main__":
    import json
    try:
        check({entry_point})
        print(json.dumps({{"passed": True}}))
    except Exception as exc:
        print(json.dumps({{"passed": False, "error": type(exc).__name__}}))
"""
    with tempfile.TemporaryDirectory(prefix="foundation-code-") as folder:
        path = Path(folder)
        (path / "run.py").write_text(harness, encoding="utf-8")
        if mode == "docker":
            command = ["docker", "run", "--rm", "--network", "none", "--memory", "512m", "--pids-limit", "64",
                       "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
                       "-v", f"{path.resolve()}:/work:ro", "-w", "/work", "python:3.12-slim",
                       "python", "-I", "/work/run.py"]
        elif mode == "local":
            if os.name == "nt" and os.getenv("FOUNDATION_ALLOW_LOCAL_CODE") != "1":
                raise RuntimeError("local code execution is disabled on Windows; use the Docker runner")
            command = [sys.executable, "-I", "run.py"]
        else:
            raise ValueError("code mode must be docker or local")
        try:
            result = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=timeout,
                                    env={"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"})
        except FileNotFoundError:
            return {"passed": False, "status": "execution_error", "reason": "runner not available"}
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "execution_error", "reason": "timeout"}
        lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
        if not lines:
            return {"passed": False, "status": "execution_error", "reason": f"exit={result.returncode}",
                    "stderr": result.stderr[-500:]}
        payload = json.loads(lines[-1])
        return {"passed": bool(payload.get("passed")), "status": "ok", **payload}


def _code(sample: Sample, response: ModelResponse, metric: str, mode: str) -> Score:
    code = extract_code(response.text)
    if not code.strip():
        return _score(sample, metric, None, "extract_error", {"reason": "empty code"})
    outcome = run_code_tests(code, sample.metadata.get("test_code", ""),
                             sample.metadata.get("entry_point", ""), mode=mode)
    if outcome.get("status") != "ok":
        return _score(sample, metric, None, "execution_error", outcome)
    return _score(sample, metric, outcome["passed"], detail=outcome)


def score_sample(sample: Sample, response: ModelResponse, metric: str | None = None,
                 code_mode: str = "docker") -> Score:
    metric = metric or sample.metadata.get("metric", "native")
    if sample.task in {"gsm8k", "svamp", "ruler"}:
        return _numeric(sample, response, metric)
    if sample.task == "mmlu_pro":
        return _choice(sample, response, metric)
    if sample.task == "bbh":
        return _bbh(sample, response, metric)
    if sample.task == "ifeval":
        return _ifeval(sample, response, metric)
    if sample.task == "humaneval_plus":
        return _code(sample, response, metric, code_mode)
    return _score(sample, metric, None, "error", {"reason": "unknown task"})


def auxiliary_error_analysis(endpoint, sample: Sample, response: ModelResponse, score: Score,
                             prompt: str, generation: dict):
    """Optional LLM error note. Never enters the native metric."""
    from .agent import format_chat
    messages = format_chat(sample, [], prompt)
    messages.append({"role": "user", "content": f"MODEL RESPONSE:\n{response.text}\n\nSCORE: {json.dumps(score.model_dump())}"})
    from .agent import ModelAdapter
    adapter = ModelAdapter(endpoint, generation)
    # Reuse the transport with a synthetic sample so the auxiliary call is auditable.
    synthetic = sample.model_copy(update={"sample_id": sample.sample_id + "-error-analysis"})
    generated = adapter.generate(synthetic, "chat", [], prompt, "auxiliary-judge")
    return {"analysis": generated.text, "usage": generated.usage}
