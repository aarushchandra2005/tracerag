# Retrieval results (BEIR SciFact)

Test split: 300 claims. Settings tuned on the 809 train claims: `{'zone_weights_whole': [0.1, 1.0], 'hybrid_alpha_whole': 0.5, 'zone_weights_sent': [0.25, 1.0], 'hybrid_alpha_sent': 0.4}`.

## Main table (whole-abstract chunks, test)

| Retriever | P@5 | P@10 | R@10 | MRR@10 | nDCG@10 | nDCG@10 95% CI |
|---|---|---|---|---|---|---|
| tf-idf lnc.ltc (zones) | 0.163 | 0.093 | 0.846 | 0.644 | 0.687 | 0.646–0.728 |
| BM25 | 0.162 | 0.091 | 0.820 | 0.644 | 0.683 | 0.639–0.725 |
| Dense (MiniLM) | 0.164 | 0.088 | 0.783 | 0.605 | 0.645 | 0.599–0.691 |
| Hybrid RRF (BM25 + dense) | 0.175 | 0.096 | 0.858 | 0.683 | 0.720 | 0.677–0.761 |
| Hybrid weighted (BM25 + dense) | 0.174 | 0.096 | 0.857 | 0.700 | 0.733 | 0.691–0.774 |

## nDCG@10 by claim type (test)

| Retriever | SUPPORT (n=124) | CONTRADICT (n=64) | NEI (n=112) |
|---|---|---|---|
| tf-idf lnc.ltc (zones) | 0.861 | 0.733 | 0.469 |
| BM25 | 0.867 | 0.750 | 0.440 |
| Dense (MiniLM) | 0.784 | 0.770 | 0.421 |
| Hybrid RRF (BM25 + dense) | 0.884 | 0.792 | 0.496 |
| Hybrid weighted (BM25 + dense) | 0.900 | 0.822 | 0.498 |

## Paired randomization tests on per-query nDCG@10 (test)

| A | B | chunking | nDCG@10 A | nDCG@10 B | diff | p (randomization, 10k) |
|---|---|---|---|---|---|---|
| BM25 | tf-idf lnc.ltc (zones) | whole | 0.683 | 0.687 | -0.005 | 0.571 |
| BM25 | tf-idf lnc.ltc (zones) | sent | 0.680 | 0.672 | 0.009 | 0.309 |
| tf-idf lnc.ltc (zones) | tf-idf lnc.ltc (one field) | whole | 0.687 | 0.674 | 0.013 | 0.016 |
| tf-idf lnc.ltc (zones) | tf-idf lnc.ltc (one field) | sent | 0.672 | 0.674 | -0.003 | 0.701 |
| Dense (MiniLM) | BM25 | whole | 0.645 | 0.683 | -0.038 | 0.036 |
| Dense (MiniLM) | BM25 | sent | 0.682 | 0.680 | 0.001 | 0.944 |
| Hybrid RRF (BM25 + dense) | BM25 | whole | 0.720 | 0.683 | 0.037 | 0.001 |
| Hybrid RRF (BM25 + dense) | BM25 | sent | 0.723 | 0.680 | 0.043 | <0.001 |
| Hybrid RRF (BM25 + dense) | Dense (MiniLM) | whole | 0.720 | 0.645 | 0.075 | <0.001 |
| Hybrid RRF (BM25 + dense) | Dense (MiniLM) | sent | 0.723 | 0.682 | 0.041 | 0.001 |
| Hybrid weighted (BM25 + dense) | Hybrid RRF (BM25 + dense) | whole | 0.733 | 0.720 | 0.013 | 0.026 |
| Hybrid weighted (BM25 + dense) | Hybrid RRF (BM25 + dense) | sent | 0.733 | 0.723 | 0.010 | 0.124 |
| Hybrid RRF (BM25 + dense) | Hybrid RRF (tf-idf + dense) | whole | 0.720 | 0.714 | 0.006 | 0.324 |
| Hybrid RRF (BM25 + dense) | Hybrid RRF (tf-idf + dense) | sent | 0.723 | 0.723 | 0.000 | 0.925 |

## Ablations

| ablation | setting | retriever | chunking | nDCG@10 train | nDCG@10 test | MRR@10 test | R@10 test | P@5 test |
|---|---|---|---|---|---|---|---|---|
| tokenizer | stemming on, stopwords removed | tf-idf lnc.ltc (zones) | whole | 0.683 | 0.687 | 0.644 | 0.846 | 0.163 |
| tokenizer | stemming on, stopwords removed | BM25 | whole | 0.696 | 0.683 | 0.644 | 0.820 | 0.162 |
| tokenizer | stemming off, stopwords removed | tf-idf lnc.ltc (zones) | whole | 0.652 | 0.659 | 0.622 | 0.794 | 0.157 |
| tokenizer | stemming off, stopwords removed | BM25 | whole | 0.668 | 0.666 | 0.634 | 0.786 | 0.158 |
| tokenizer | stemming on, stopwords kept | tf-idf lnc.ltc (zones) | whole | 0.689 | 0.691 | 0.650 | 0.835 | 0.162 |
| tokenizer | stemming on, stopwords kept | BM25 | whole | 0.698 | 0.684 | 0.648 | 0.815 | 0.161 |
| tokenizer | stemming off, stopwords kept | tf-idf lnc.ltc (zones) | whole | 0.657 | 0.656 | 0.621 | 0.785 | 0.153 |
| tokenizer | stemming off, stopwords kept | BM25 | whole | 0.668 | 0.663 | 0.631 | 0.788 | 0.157 |
| title weight | w_title=0 x w_body | tf-idf lnc.ltc (zones) | whole | 0.666 | 0.670 | 0.628 | 0.825 | 0.161 |
| title weight | w_title=0.05 x w_body | tf-idf lnc.ltc (zones) | whole | 0.677 | 0.677 | 0.635 | 0.829 | 0.161 |
| title weight | w_title=0.1 x w_body | tf-idf lnc.ltc (zones) | whole | 0.683 | 0.687 | 0.644 | 0.846 | 0.163 |
| title weight | w_title=0.15 x w_body | tf-idf lnc.ltc (zones) | whole | 0.682 | 0.683 | 0.640 | 0.839 | 0.165 |
| title weight | w_title=0.2 x w_body | tf-idf lnc.ltc (zones) | whole | 0.681 | 0.680 | 0.639 | 0.834 | 0.167 |
| title weight | w_title=0.25 x w_body | tf-idf lnc.ltc (zones) | whole | 0.678 | 0.673 | 0.631 | 0.828 | 0.163 |
| title weight | w_title=0.5 x w_body | tf-idf lnc.ltc (zones) | whole | 0.648 | 0.647 | 0.611 | 0.783 | 0.155 |
| title weight | w_title=1 x w_body | tf-idf lnc.ltc (zones) | whole | 0.590 | 0.594 | 0.559 | 0.723 | 0.145 |
| title weight | w_title=2 x w_body | tf-idf lnc.ltc (zones) | whole | 0.541 | 0.541 | 0.504 | 0.681 | 0.135 |
| title weight | w_title=3 x w_body | tf-idf lnc.ltc (zones) | whole | 0.517 | 0.514 | 0.477 | 0.645 | 0.131 |
| title weight | one merged field | tf-idf lnc.ltc (one field) | whole | 0.682 | 0.674 | 0.632 | 0.827 | 0.165 |
| title weight | w_title=0 x w_body | tf-idf lnc.ltc (zones) | sent | 0.645 | 0.637 | 0.595 | 0.787 | 0.155 |
| title weight | w_title=0.05 x w_body | tf-idf lnc.ltc (zones) | sent | 0.655 | 0.651 | 0.607 | 0.808 | 0.157 |
| title weight | w_title=0.1 x w_body | tf-idf lnc.ltc (zones) | sent | 0.665 | 0.663 | 0.620 | 0.815 | 0.160 |
| title weight | w_title=0.15 x w_body | tf-idf lnc.ltc (zones) | sent | 0.671 | 0.672 | 0.633 | 0.816 | 0.161 |
| title weight | w_title=0.2 x w_body | tf-idf lnc.ltc (zones) | sent | 0.673 | 0.672 | 0.633 | 0.813 | 0.164 |
| title weight | w_title=0.25 x w_body | tf-idf lnc.ltc (zones) | sent | 0.673 | 0.672 | 0.632 | 0.817 | 0.163 |
| title weight | w_title=0.5 x w_body | tf-idf lnc.ltc (zones) | sent | 0.657 | 0.649 | 0.607 | 0.798 | 0.160 |
| title weight | w_title=1 x w_body | tf-idf lnc.ltc (zones) | sent | 0.608 | 0.610 | 0.574 | 0.741 | 0.145 |
| title weight | w_title=2 x w_body | tf-idf lnc.ltc (zones) | sent | 0.556 | 0.553 | 0.518 | 0.684 | 0.139 |
| title weight | w_title=3 x w_body | tf-idf lnc.ltc (zones) | sent | 0.530 | 0.533 | 0.500 | 0.660 | 0.133 |
| title weight | one merged field | tf-idf lnc.ltc (one field) | sent | 0.678 | 0.674 | 0.637 | 0.819 | 0.160 |
| chunking | whole (whole abstract) | tf-idf lnc.ltc (zones) | whole | 0.683 | 0.687 | 0.644 | 0.846 | 0.163 |
| chunking | sent (sentence windows) | tf-idf lnc.ltc (zones) | sent | 0.673 | 0.672 | 0.632 | 0.817 | 0.163 |
| chunking | whole (whole abstract) | BM25 | whole | 0.696 | 0.683 | 0.644 | 0.820 | 0.162 |
| chunking | sent (sentence windows) | BM25 | sent | 0.684 | 0.680 | 0.648 | 0.805 | 0.162 |
| chunking | whole (whole abstract) | Dense (MiniLM) | whole | 0.660 | 0.645 | 0.605 | 0.783 | 0.164 |
| chunking | sent (sentence windows) | Dense (MiniLM) | sent | 0.690 | 0.682 | 0.642 | 0.818 | 0.166 |
| chunking | whole (whole abstract) | Hybrid RRF (BM25 + dense) | whole | 0.719 | 0.720 | 0.683 | 0.858 | 0.175 |
| chunking | sent (sentence windows) | Hybrid RRF (BM25 + dense) | sent | 0.726 | 0.723 | 0.684 | 0.860 | 0.173 |
| chunking | whole (whole abstract) | Hybrid weighted (BM25 + dense) | whole | 0.736 | 0.733 | 0.700 | 0.857 | 0.174 |
| chunking | sent (sentence windows) | Hybrid weighted (BM25 + dense) | sent | 0.738 | 0.733 | 0.697 | 0.858 | 0.176 |
| fusion | Hybrid RRF (BM25 + dense) | Hybrid RRF (BM25 + dense) | whole | 0.719 | 0.720 | 0.683 | 0.858 | 0.175 |
| fusion | Hybrid RRF (tf-idf + dense) | Hybrid RRF (tf-idf + dense) | whole | 0.711 | 0.714 | 0.674 | 0.860 | 0.171 |
| fusion | Hybrid weighted (BM25 + dense) | Hybrid weighted (BM25 + dense) | whole | 0.736 | 0.733 | 0.700 | 0.857 | 0.174 |
| BM25 k1/b | k1=0.9, b=0.4 | BM25 | whole | 0.697 | 0.676 | 0.641 | 0.803 | 0.162 |
| BM25 k1/b | k1=1.2, b=0.75 | BM25 | whole | 0.696 | 0.683 | 0.644 | 0.820 | 0.162 |
| BM25 k1/b | k1=1.5, b=0.75 | BM25 | whole | 0.698 | 0.689 | 0.652 | 0.827 | 0.162 |
| BM25 k1/b | k1=2.0, b=0.75 | BM25 | whole | 0.694 | 0.690 | 0.649 | 0.838 | 0.163 |
| BM25 k1/b | k1=1.2, b=1.0 | BM25 | whole | 0.696 | 0.686 | 0.648 | 0.820 | 0.165 |
| BM25 k1/b | k1=1.2, b=0.5 | BM25 | whole | 0.700 | 0.675 | 0.636 | 0.811 | 0.163 |
