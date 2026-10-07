import zlib

import numpy as np
import pytest

import config
from src.generate import llm as llm_mod
from src.generate import rag
from src.retrieve.bm25 import BM25Retriever


def _bow(texts, model=None):
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, zlib.crc32(w.encode()) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_extractive_answer_cites_only_retrieved_chunks(monkeypatch, toy_sent_index):
    monkeypatch.setattr(rag, "encode", _bow)
    out = rag.answer("swim training suppresses tumor growth in mice", BM25Retriever(toy_sent_index), k=2)
    retrieved = {r["chunk_id"] for r in out["retrieved"]}
    assert out["backend"] == "extractive" and out["cited_chunks"]
    assert set(out["cited_chunks"]) <= retrieved
    assert all(p["cited_ids"] and not p["invalid_ids"] for p in out["sentences"])
    assert "Swim training suppresses tumor growth in mice" in out["answer"]


def test_extractive_abstains_when_nothing_is_close(monkeypatch, toy_sent_index):
    monkeypatch.setattr(rag, "encode", _bow)
    gen = rag.ExtractiveGenerator(min_similarity=0.99)
    assert gen.generate_from_chunks("quantum gravity", toy_sent_index.chunks[:2]) == rag.NOT_ENOUGH


def test_prompt_lists_chunk_ids(toy_index):
    prompt = rag.build_user_prompt("a claim", toy_index.chunks[:2])
    assert prompt.startswith("Claim: a claim") and "[d1]" in prompt and "[d2]" in prompt


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text, self.headers = status, payload, text, {}

    def json(self):
        return self._payload


def test_anthropic_request_and_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return FakeResponse(200, {"content": [{"type": "text", "text": "It holds [1]."}]})

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    model = llm_mod.AnthropicLLM()
    assert model.generate("sys", "user") == "It holds [1]."
    url, headers, body = calls[0]
    assert url == "https://api.anthropic.com/v1/messages"
    assert headers["x-api-key"] == "test-key" and headers["anthropic-version"] == "2023-06-01"
    assert body["system"] == "sys" and body["messages"] == [{"role": "user", "content": "user"}]
    assert body["model"] == config.ANTHROPIC_DEFAULT_MODEL
    assert llm_mod.AnthropicLLM().generate("sys", "user") == "It holds [1]." and len(calls) == 1  # disk cache


def test_openai_compatible_switches_to_max_completion_tokens(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    bodies = []

    def fake_post(url, headers, json, timeout):
        bodies.append(dict(json))
        if "max_tokens" in json:
            return FakeResponse(400, text='{"error": "Unsupported parameter: max_tokens"}')
        return FakeResponse(200, {"choices": [{"message": {"content": "Answer [2]."}}]})

    import requests
    monkeypatch.setattr(requests, "post", fake_post)
    model = llm_mod.OpenAICompatibleLLM("groq", "some-model")
    assert model.generate("s", "u") == "Answer [2]."
    assert "max_completion_tokens" in bodies[-1] and model.base_url == "https://api.groq.com/openai/v1"


def test_missing_model_or_key_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm_mod.LLMError, match="--model"):
        llm_mod.OpenAICompatibleLLM("openai", "")
    with pytest.raises(llm_mod.LLMError, match="OPENAI_API_KEY"):
        llm_mod.OpenAICompatibleLLM("openai", "m")
    assert llm_mod.make_llm("extractive") is None
