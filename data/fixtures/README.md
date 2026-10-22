# Fixtures

These files are hand-written, tiny, and used only for offline engineering checks.

- `demo_samples.jsonl`: 8 GSM8K-style train items plus a few MMLU-Pro/BBH/IFEval/HumanEval+/RULER items. They are not benchmark samples.
- `responses.jsonl`: scripted model outputs, including two deliberate GSM8K errors so the demo exercises failure reporting.

`foundation-eval demo` writes a new `runs/` directory and labels every artifact `scripted`/`not_a_benchmark`. Never cite fixture accuracy as a model result, and never mix these rows into a real training pool.
