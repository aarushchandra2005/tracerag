import random
import zlib

import numpy as np
import pytest

from src.ingest.chunking import Chunk
from src.verify import inject, splitter
from src.verify.checker import CitationChecker, evidence_windows


# ----------------------------------------------------------------- splitter
def test_citations_before_and_after_the_full_stop():
    parsed = splitter.split("Mice ran faster [12_0]. Tumors shrank. [55] Both held [1][2].")
    assert [p["cited_ids"] for p in parsed] == [["12_0"], ["55"], ["1", "2"]]
    assert [p["text"] for p in parsed] == ["Mice ran faster.", "Tumors shrank.", "Both held."]


def test_no_space_after_full_stop_and_comma_lists():
    parsed = splitter.split("It worked.[12] It failed [13, 14].")
    assert [p["cited_ids"] for p in parsed] == [["12"], ["13", "14"]]


def test_chemistry_brackets_are_not_citations():
    (p,) = splitter.split("[Ca2+]i rose sharply [77_3].")
    assert p["cited_ids"] == ["77_3"] and p["text"] == "[Ca2+]i rose sharply."


def test_invalid_ids_and_abstention():
    parsed = splitter.split("A holds [1]. B holds [999]. Not enough evidence.", valid_ids=["1"])
    assert parsed[1]["invalid_ids"] == ["999"]
    assert parsed[2]["abstain"] and not parsed[0]["abstain"]
    assert not splitter.split("No evidence of harm was seen [1].")[0]["abstain"]


def test_with_citations_round_trip():
    s = splitter.with_citations("Cells died.", ["4_1", "9"])
    assert s == "Cells died [4_1][9]." and splitter.split(s)[0]["cited_ids"] == ["4_1", "9"]


# ----------------------------------------------------------------- negation rules
@pytest.mark.parametrize("src,expected", [
    ("Exercise increased survival.", "Exercise decreased survival."),
    ("The drug did not reduce pain.", "The drug did reduce pain."),
    ("Smoking is linked to cancer.", "Smoking is not linked to cancer."),
    ("Mice ate the food.", "It is not true that mice ate the food."),
    ("Higher doses helped.", "Lower doses helped."),
])
def test_negate_sentence(src, expected):
    assert inject.negate_sentence(src) == expected


RETRIEVED = [
    Chunk("10_0", "10", "Paper A", "Exercise increased survival in mice. Survival rose by half."),
    Chunk("10_1", "10", "Paper A", "Survival rose by half. Effects lasted a year."),
    Chunk("20_0", "20", "Paper B", "Diet had no effect on weight."),
]
ANSWER = "Exercise increased survival in mice [10_0]. Effects lasted a year [10_1]."


@pytest.mark.parametrize("kind", ["negate", "swap_citation", "add_fabricated"])
def test_corruptions_point_at_the_changed_sentence(kind):
    rng = random.Random(1)
    valid = [c.chunk_id for c in RETRIEVED]
    if kind == "negate":
        corr = inject.negate(ANSWER, rng, valid)
    elif kind == "swap_citation":
        corr = inject.swap_citation(ANSWER, rng, RETRIEVED, valid)
    else:
        corr = inject.add_fabricated(ANSWER, rng, "Coffee cures colds.", ["10_0"], valid)
    parsed = splitter.split(corr.answer, valid)
    target = parsed[corr.index]
    assert target["sentence"] == corr.corrupted
    if kind == "swap_citation":
        assert target["cited_ids"] == ["20_0"]           # the only chunk from another paper
    if kind == "negate":
        assert "decreased" in target["text"] or "not" in target["text"]


def test_nothing_to_corrupt_in_an_abstention():
    assert inject.negate("Not enough evidence.", random.Random(0)) is None


# ----------------------------------------------------------------- checker logic with stub models
def _bow(texts):
    out = np.zeros((len(texts), 256))
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").split():
            out[i, zlib.crc32(w.encode()) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class StubNLI:
    """Entails if every hypothesis word is in the premise; contradicts if it adds 'not'."""

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += len(pairs)
        rows = []
        for prem, hyp in pairs:
            p, h = set(prem.lower().replace(".", "").split()), set(hyp.lower().replace(".", "").split())
            if "not" in h and "not" not in p and (h - {"not"}) <= p:
                rows.append([0.05, 0.05, 0.90])
            elif h <= p:
                rows.append([0.90, 0.05, 0.05])
            else:
                rows.append([0.10, 0.80, 0.10])
        return np.array(rows)


@pytest.fixture
def checker(monkeypatch):
    c = CitationChecker("cascade", thresholds_override={"cosine": 0.5, "nli": 0.5, "gate": 0.2, "cascade_nli": 0.5})
    monkeypatch.setattr(c, "_embed", _bow)
    c._nli = StubNLI()
    return c


def test_evidence_windows_include_title_and_pairs():
    wins = evidence_windows("T", "A one. B two. C three.", window=2)
    assert wins == ["T", "A one. B two.", "B two. C three."]


def test_verdict_rules(checker):
    chunk = RETRIEVED[0]
    assert checker.verdict("Anything.", [], "cosine").reason == "no citation"
    assert checker.verdict("x", [chunk], "nli", invalid_ids=["99"]).label == "unsupported"
    assert checker.verdict("Not enough evidence.", [], abstain=True).label == "abstain"
    ok = checker.verdict("Exercise increased survival in mice.", [chunk], "nli")
    assert ok.label == "supported" and ok.nli_called
    neg = checker.verdict("Exercise not increased survival in mice.", [chunk], "nli")
    assert neg.label == "unsupported" and "contradicted" in neg.reason


def test_cascade_skips_nli_for_off_topic_citations(checker):
    before = checker.nli.calls
    v = checker.verdict("Quantum chromodynamics predicts gluons.", [RETRIEVED[2]], "cascade")
    assert v.label == "unsupported" and not v.nli_called and checker.nli.calls == before


def test_check_answer_end_to_end(checker):
    rows = checker.check_answer(ANSWER + " Coffee cures colds [20_0].", RETRIEVED, "nli")
    assert [r["verdict"].label for r in rows] == ["supported", "supported", "unsupported"]


# ----------------------------------------------------------------- real NLI plumbing on a tiny random model
def test_nli_wrapper_loads_a_deberta_checkpoint(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    text = tmp_path / "t.txt"
    text.write_text("\n".join(["the drug reduced tumor growth in mice", "cats are mammals",
                               "survival increased after exercise"] * 50), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(text), model_prefix=str(tmp_path / "spm"), vocab_size=60,
                                   pad_id=0, pad_piece="[PAD]", bos_id=1, bos_piece="[CLS]", eos_id=2,
                                   eos_piece="[SEP]", unk_id=3, unk_piece="[UNK]", user_defined_symbols=["[MASK]"],
                                   hard_vocab_limit=False, minloglevel=2)
    tok = transformers.DebertaV2Tokenizer(vocab_file=str(tmp_path / "spm.model"))
    cfg = transformers.DebertaV2Config(
        vocab_size=len(tok) + 8, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        type_vocab_size=0, relative_attention=True, position_buckets=16, pos_att_type=["p2c", "c2p"], num_labels=3,
        id2label={0: "contradiction", 1: "entailment", 2: "neutral"},
        label2id={"contradiction": 0, "entailment": 1, "neutral": 2})
    model_dir = tmp_path / "nli"
    transformers.DebertaV2ForSequenceClassification(cfg).save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    from src.verify.nli import NLIModel

    m = NLIModel(str(model_dir), device="cpu", batch_size=2, cache_dir=tmp_path)
    assert m.column == {"entail": 1, "neutral": 2, "contradict": 0}
    pairs = [("the drug reduced tumor growth", "tumors shrank"), ("cats are mammals", "dogs bark"),
             ("word " * 900, "long premise gets truncated"), ("cats are mammals", "dogs bark")]
    probs = m.predict(pairs)
    assert probs.shape == (4, 3) and np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], probs[3])
    again = NLIModel(str(model_dir), device="cpu", cache_dir=tmp_path)    # answers come from the disk cache
    assert np.allclose(again.predict(pairs[:2]), probs[:2], atol=1e-6)
    assert len(again._disk) == 3


def test_swap_falls_back_to_another_paper_when_all_chunks_share_one():
    same_paper = RETRIEVED[:2]
    other = Chunk("30", "30", "Paper C", "Unrelated finding about diet.")
    valid = [c.chunk_id for c in same_paper]
    assert inject.swap_citation(ANSWER, random.Random(0), same_paper, valid) is None
    corr = inject.swap_citation(ANSWER, random.Random(0), same_paper, valid, fallback=[other])
    assert corr.extra_chunk is other
    assert splitter.split(corr.answer, valid + ["30"])[corr.index]["cited_ids"] == ["30"]
