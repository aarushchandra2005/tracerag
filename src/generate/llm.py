"""LLM backends for the generator. Plain HTTPS calls, no vendor SDKs.

  anthropic   Claude Messages API            key: ANTHROPIC_API_KEY
  openai      any OpenAI-compatible endpoint  key: OPENAI_API_KEY, base: OPENAI_BASE_URL
  groq        OpenAI-compatible preset        key: GROQ_API_KEY
  gemini      OpenAI-compatible preset        key: GEMINI_API_KEY
  openrouter  OpenAI-compatible preset        key: OPENROUTER_API_KEY
  ollama      local OpenAI-compatible server  no key (http://localhost:11434/v1)

The model name is always required for OpenAI-compatible backends (pass
--model or set TRACERAG_LLM_MODEL), because providers rename models often.
Responses are cached on disk by (backend, model, prompt), so reruns are free
and reproducible.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import config

PRESETS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}
API_BACKENDS = ("anthropic",) + tuple(PRESETS)


class LLMError(RuntimeError):
    pass


class _Cached:
    """Disk cache shared by the API backends."""

    def __init__(self, backend: str, model: str):
        self.backend, self.model = backend, model
        self.path = config.CACHE_DIR / "llm" / f"{backend}.jsonl"
        self._mem: dict[str, str] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["text"]

    def key(self, system: str, user: str) -> str:
        blob = json.dumps([self.backend, self.model, config.LLM_TEMPERATURE, system, user], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        return self._mem.get(key)

    def put(self, key: str, text: str) -> None:
        self._mem[key] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"key": key, "model": self.model, "text": text}, ensure_ascii=False) + "\n")


def _post(url: str, headers: dict, body: dict, retries: int = 5) -> dict:
    import requests

    delay = 2.0
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise LLMError(f"network error calling {url}: {e}") from e
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504, 529) and attempt < retries - 1:
            wait = float(resp.headers.get("retry-after") or delay)
            time.sleep(min(wait, 60.0))
            delay *= 2
            continue
        raise LLMError(f"{url} returned {resp.status_code}: {resp.text[:500]}")
    raise LLMError(f"giving up on {url}")


class AnthropicLLM:
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str = "", api_key: str | None = None):
        self.name = "anthropic"
        self.model = model or config.ANTHROPIC_DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise LLMError("set ANTHROPIC_API_KEY to use the anthropic backend")
        self.cache = _Cached(self.name, self.model)

    def generate(self, system: str, user: str) -> str:
        key = self.cache.key(system, user)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": self.model, "max_tokens": config.LLM_MAX_TOKENS, "temperature": config.LLM_TEMPERATURE,
                "system": system, "messages": [{"role": "user", "content": user}]}
        try:
            data = _post(self.url, headers, body)
        except LLMError as e:
            if "temperature" not in str(e):
                raise
            body.pop("temperature")              # some models fix the temperature themselves
            data = _post(self.url, headers, body)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
        self.cache.put(key, text)
        return text


class OpenAICompatibleLLM:
    def __init__(self, backend: str = "openai", model: str = "", api_key: str | None = None,
                 base_url: str | None = None):
        if backend not in PRESETS:
            raise LLMError(f"unknown backend {backend!r}")
        default_url, key_env = PRESETS[backend]
        self.name = backend
        self.model = model
        if not self.model:
            raise LLMError(f"pass --model (or set TRACERAG_LLM_MODEL) for the {backend} backend; "
                           "use a chat model name from your provider's model list")
        self.base_url = (base_url or (os.environ.get("OPENAI_BASE_URL") if backend == "openai" else None)
                         or default_url).rstrip("/")
        self.api_key = api_key or (os.environ.get(key_env) if key_env else "ollama")
        if not self.api_key:
            raise LLMError(f"set {key_env} to use the {backend} backend")
        self.cache = _Cached(self.name, self.model)

    def generate(self, system: str, user: str) -> str:
        key = self.cache.key(system, user)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {"model": self.model, "temperature": config.LLM_TEMPERATURE, "max_tokens": config.LLM_MAX_TOKENS,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        url = f"{self.base_url}/chat/completions"
        for _ in range(3):                       # adapt to models that reject an optional field
            try:
                data = _post(url, headers, body)
                break
            except LLMError as e:
                msg = str(e)
                if "max_tokens" in msg and "max_tokens" in body:
                    body["max_completion_tokens"] = body.pop("max_tokens")
                elif "temperature" in msg and "temperature" in body:
                    body.pop("temperature")
                else:
                    raise
        else:
            raise LLMError(f"{url} kept rejecting the request")
        text = (data["choices"][0]["message"].get("content") or "").strip()
        self.cache.put(key, text)
        return text


def make_llm(backend: str = config.LLM_BACKEND, model: str = config.LLM_MODEL):
    """Return an object with .name, .model and .generate(system, user), or None for 'extractive'."""
    if backend == "extractive":
        return None
    if backend == "anthropic":
        return AnthropicLLM(model)
    return OpenAICompatibleLLM(backend, model)
