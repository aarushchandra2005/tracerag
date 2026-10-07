from src.index.tokenizer import STOPWORDS, Tokenizer, tokenize
from src.ingest.chunking import chunk_sentences, chunk_whole, split_sentences


# ----------------------------------------------------------------- tokenizer
def test_case_fold_punctuation_stopwords_and_stemming():
    assert tokenize("The Tumors were GROWING, rapidly!") == ["tumor", "grow", "rapidli"]


def test_toggles_for_ablation():
    text = "The tumors were growing"
    assert Tokenizer(stem=False, stopwords=True).tokenize(text) == ["tumors", "growing"]
    assert Tokenizer(stem=True, stopwords=False).tokenize(text) == ["the", "tumor", "were", "grow"]
    assert Tokenizer(stem=False, stopwords=False).tokenize(text) == ["the", "tumors", "were", "growing"]


def test_scientific_tokens_survive():
    toks = Tokenizer(stem=False, stopwords=False).tokenize("TNF-α and IL-6 rose by 2.5-fold (p<0.05).")
    assert "α" in toks and "il" in toks and "6" in toks and "tnf" in toks


def test_stopword_list_is_nltk_without_apostrophes():
    assert len(STOPWORDS) == 153 and "not" in STOPWORDS and "the" in STOPWORDS


def test_empty_input():
    assert tokenize("") == [] and tokenize(None) == []


# ----------------------------------------------------------------- sentence splitter
def test_splits_on_sentence_ends():
    assert split_sentences("Cells died. The effect was large! Was it? Yes.") == [
        "Cells died.", "The effect was large!", "Was it?", "Yes."]


def test_keeps_abbreviations_decimals_and_species():
    text = "Smith et al. showed a 2.5-fold rise, e.g. in E. coli strains. Results were clear."
    assert split_sentences(text) == [
        "Smith et al. showed a 2.5-fold rise, e.g. in E. coli strains.", "Results were clear."]


def test_splits_before_lowercase_gene_names_and_digits():
    assert split_sentences("Levels rose. mRNA fell. p53 was stable. 32 mice died.") == [
        "Levels rose.", "mRNA fell.", "p53 was stable.", "32 mice died."]


def test_closing_parenthesis_stays_with_sentence():
    assert split_sentences("Risk fell (95% CI 1.2-3.4). The trial ended.") == [
        "Risk fell (95% CI 1.2-3.4).", "The trial ended."]


# ----------------------------------------------------------------- chunkers
DOC = {"doc_id": "7", "title": "T", "text": "S0 a. S1 b. S2 c. S3 d. S4 e."}


def test_whole_chunk_keeps_doc_id():
    (c,) = chunk_whole(DOC)
    assert c.chunk_id == "7" and c.doc_id == "7" and c.text == DOC["text"] and c.end == 5


def test_windows_overlap_and_cover_every_sentence():
    chunks = chunk_sentences(DOC, window=3, stride=1)
    assert [c.chunk_id for c in chunks] == ["7_0", "7_1", "7_2"]
    assert chunks[0].text == "S0 a. S1 b. S2 c." and chunks[-1].text == "S2 c. S3 d. S4 e."
    assert all(c.doc_id == "7" and c.title == "T" for c in chunks)


def test_stride_that_skips_the_tail_gets_a_final_window():
    chunks = chunk_sentences(DOC, window=2, stride=2)
    assert [(c.start, c.end) for c in chunks] == [(0, 2), (2, 4), (3, 5)]


def test_short_doc_is_one_window():
    chunks = chunk_sentences({"doc_id": "9", "title": "T", "text": "Only one sentence."}, window=3)
    assert len(chunks) == 1 and chunks[0].chunk_id == "9_0"
