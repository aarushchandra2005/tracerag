from collections import Counter

import numpy as np

from src.index.inverted_index import InvertedIndex
from tests.conftest import TOY_CORPUS


def _expected_postings(index, zone):
    """Brute force: tokenize every chunk again and count."""
    exp = {}
    for c in index.chunks:
        text = {"title": c.title, "body": c.text, "all": f"{c.title} {c.text}"}[zone]
        for term, tf in Counter(index.tokenizer.tokenize(text)).items():
            exp.setdefault(term, []).append((c.chunk_id, tf))
    return exp


def test_postings_match_brute_force(toy_index):
    for zone in ("title", "body", "all"):
        exp = _expected_postings(toy_index, zone)
        assert set(exp) == set(toy_index.zones[zone].terms)
        for term, plist in exp.items():
            assert toy_index.get_postings(term, zone) == plist
            assert toy_index.zones[zone].doc_freq(term) == len(plist)


def test_postings_sorted_and_lengths(toy_index):
    z = toy_index.zones["body"]
    for i in range(len(z.terms)):
        docs = z.post_doc[z.offsets[i]:z.offsets[i + 1]]
        assert np.all(np.diff(docs) > 0)
    for c, chunk in enumerate(toy_index.chunks):
        assert z.length[c] == len(toy_index.tokenizer.tokenize(chunk.text))


def test_lnc_norms(toy_index):
    z = toy_index.zones["all"]
    c = toy_index.chunk_pos["d1"]
    tf = Counter(toy_index.tokenizer.tokenize(TOY_CORPUS["d1"]["title"] + " " + TOY_CORPUS["d1"]["text"]))
    expected = np.sqrt(sum((1 + np.log10(v)) ** 2 for v in tf.values()))
    assert np.isclose(z.lnc_norm[c], expected)


def test_dictionary_and_analyzed_lookup(toy_index):
    assert toy_index.dictionary("all")["insur"] == 2          # d1 and d4
    assert toy_index.get_postings("Insurance", analyze=True) == [("d1", 3), ("d4", 2)]


def test_save_load_roundtrip(tmp_path, toy_sent_index):
    path = toy_sent_index.save(tmp_path / "idx.pkl")
    loaded = InvertedIndex.load(path)
    assert loaded.chunk_ids == toy_sent_index.chunk_ids
    assert loaded.tokenizer.settings == toy_sent_index.tokenizer.settings
    for zone in ("title", "body", "all"):
        a, b = loaded.zones[zone], toy_sent_index.zones[zone]
        assert a.terms == b.terms
        assert np.array_equal(a.post_doc, b.post_doc) and np.array_equal(a.post_tf, b.post_tf)
        assert np.allclose(a.lnc_norm, b.lnc_norm)


def test_chunk_to_paper_mapping(toy_sent_index):
    for i, c in enumerate(toy_sent_index.chunks):
        assert toy_sent_index.doc_ids[toy_sent_index.chunk_doc[i]] == c.doc_id


def test_print_postings_shows_df(toy_index):
    out = toy_index.print_postings("insurance")
    assert "df = 2 of N = 4" in out and "d1" in out
    assert "removed by the tokenizer" in toy_index.print_postings("the")
