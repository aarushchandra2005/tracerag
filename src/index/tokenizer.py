"""Text -> index terms: case fold, strip punctuation, drop stopwords, Porter stem.

Stemming and stopword removal can each be switched off for the ablation study.
The same Tokenizer must be used for documents and queries, which is why the
settings are stored inside every saved index.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.stem import PorterStemmer

# NLTK's English stopword list (179 words), minus the forms with apostrophes:
# the tokenizer splits on punctuation, so "don't" can never reach this filter.
STOPWORDS = frozenset("""
i me my myself we our ours ourselves you your yours yourself yourselves he him
his himself she her hers herself it its itself they them their theirs themselves
what which who whom this that these those am is are was were be been being have
has had having do does did doing a an the and but if or because as until while
of at by for with about against between into through during before after above
below to from up down in out on off over under again further then once here
there when where why how all any both each few more most other some such no nor
not only own same so than too very s t can will just don should now d ll m o re
ve y ain aren couldn didn doesn hadn hasn haven isn ma mightn mustn needn shan
shouldn wasn weren won wouldn
""".split())

# Runs of letters or digits in any script (keeps Greek letters such as α in TNF-α).
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

_STEMMER = PorterStemmer(mode=PorterStemmer.ORIGINAL_ALGORITHM)


@lru_cache(maxsize=None)
def porter(token: str) -> str:
    """Porter (1980) stem of one lowercase token, memoised: the corpus repeats words a lot."""
    return _STEMMER.stem(token)


class Tokenizer:
    def __init__(self, stem: bool = True, stopwords: bool = True):
        self.stem = stem
        self.stopwords = stopwords

    @property
    def settings(self) -> dict:
        return {"stem": self.stem, "stopwords": self.stopwords}

    @property
    def tag(self) -> str:
        return f"stem{int(self.stem)}_stop{int(self.stopwords)}"

    def tokenize(self, text: str) -> list[str]:
        text = unicodedata.normalize("NFKC", text or "").casefold()
        tokens = _TOKEN_RE.findall(text)
        if self.stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        if self.stem:
            tokens = [porter(t) for t in tokens]
        return tokens

    __call__ = tokenize

    def __repr__(self) -> str:
        return f"Tokenizer(stem={self.stem}, stopwords={self.stopwords})"


def tokenize(text: str, stem: bool = True, stopwords: bool = True) -> list[str]:
    """Functional form used in the roadmap: tokenize(text) -> list[str]."""
    return Tokenizer(stem, stopwords).tokenize(text)
