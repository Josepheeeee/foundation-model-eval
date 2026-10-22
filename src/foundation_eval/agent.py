"""The model under test, isolated behind one small completion/chat adapter.

`agent.py` keeps the same reading position as the retail project, but its role
is now the *subject model adapter*, not an agent harness. No tools, memory or
planning are provided: this project measures base-model capability.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

from .schema import ModelResponse, Sample


@dataclass(frozen=True)
class ModelEndpoint:
    role: str
    model: str
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls, role: str = "TARGET"):
        prefix = role.upper()
        model = os.getenv(f"{prefix}_MODEL", "").strip()
        base = (os.getenv(f"{prefix}_BASE_URL") or os.getenv("API_BASE_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        key = os.getenv(f"{prefix}_API_KEY") or os.getenv("API_KEY") or "local"
        if not model:
            raise ValueError(f"Configure {prefix}_MODEL, {prefix}_BASE_URL and {prefix}_API_KEY in .env")
        url = httpx.URL(base)
        if url.scheme not in {"http", "https"} or url.userinfo or url.query:
            raise ValueError("endpoint must be a plain http(s) base URL")
        return cls(role, model, base, key)

    def public(self):
        return {"role": self.role, "model": self.model, "base_url": self.base_url}


def format_completion(sample: Sample, fewshot: list[Sample], system_prompt: str) -> str:
    parts = [system_prompt.strip(), ""]
    for example in fewshot:
        parts += [f"Question:\n{example.prompt.strip()}", f"Answer:\n{example.answer.strip()}", ""]
    parts += [f"Question:\n{sample.prompt.strip()}", "Answer:"]
    return "\n".join(parts)


def format_chat(sample: Sample, fewshot: list[Sample], system_prompt: str) -> list[dict]:
    messages = [{"role": "system", "content": system_prompt.strip()}]
    for example in fewshot:
        messages += [{"role": "user", "content": example.prompt.strip()},
                     {"role": "assistant", "content": example.answer.strip()}]
    messages.append({"role": "user", "content": sample.prompt.strip()})
    return messages


class ModelAdapter:
    def __init__(self, endpoint: ModelEndpoint, generation: dict, timeout: float = 180, retries: int = 2):
        self.endpoint = endpoint
        self.generation = generation
        self.timeout = timeout
        self.retries = retries

    def _body(self, prompt, messages, protocol):
        body = {"model": self.endpoint.model, "temperature": self.generation.get("temperature", 0.0),
                "top_p": self.generation.get("top_p", 1.0),
                "max_tokens": self.generation.get("max_new_tokens", 1024),
                "seed": self.generation.get("seed", 2026)}
        if self.generation.get("stop"):
            body["stop"] = self.generation["stop"]
        if protocol == "completion":
            body["prompt"] = prompt
        else:
            body["messages"] = messages
        return body

    def generate(self, sample: Sample, protocol: str, fewshot: list[Sample], system_prompt: str,
                 model_label: str) -> ModelResponse:
        prompt = format_completion(sample, fewshot, system_prompt) if protocol == "completion" else ""
        messages = format_chat(sample, fewshot, system_prompt) if protocol == "chat" else []
        url = self.endpoint.base_url + ("/completions" if protocol == "completion" else "/chat/completions")
        started = time.perf_counter()
        for attempt in range(self.retries + 1):
            try:
                response = httpx.post(url, json=self._body(prompt, messages, protocol),
                                      headers={"Authorization": "Bearer " + self.endpoint.api_key},
                                      timeout=self.timeout, trust_env=True)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < self.retries:
                        time.sleep(min(2 ** attempt, 4))
                        continue
                response.raise_for_status()
                data = response.json()
                if protocol == "completion":
                    text = (data.get("choices") or [{}])[0].get("text", "")
                    finish = (data.get("choices") or [{}])[0].get("finish_reason")
                else:
                    text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
                    finish = (data.get("choices") or [{}])[0].get("finish_reason")
                return ModelResponse(sample_id=sample.sample_id, model_label=model_label, protocol=protocol,
                                     text=text, usage=data.get("usage", {}), finish_reason=finish,
                                     elapsed_seconds=time.perf_counter() - started,
                                     raw={"endpoint": self.endpoint.public(), "response_id": data.get("id")})
            except httpx.TransportError:
                if attempt == self.retries:
                    raise RuntimeError("model transport failed after bounded retries") from None
                time.sleep(min(2 ** attempt, 4))
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"model endpoint returned HTTP {exc.response.status_code}") from None
        raise RuntimeError("model retry limit exhausted")


class ScriptedAdapter:
    """Offline demo adapter; never a benchmark result."""

    def __init__(self, responses: dict[str, str], model_label="scripted"):
        self.responses = responses
        self.model_label = model_label

    def generate(self, sample: Sample, protocol: str, fewshot: list[Sample], system_prompt: str,
                 model_label: str | None = None) -> ModelResponse:
        text = self.responses.get(sample.sample_id)
        if text is None:
            raise KeyError(f"no scripted response for {sample.sample_id}")
        return ModelResponse(sample_id=sample.sample_id, model_label=model_label or self.model_label,
                             protocol=protocol, text=text, usage={"scripted": True})
