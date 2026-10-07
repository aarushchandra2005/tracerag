# TraceRAG results

Generated 2026-10-06 11:37 on Python 3.13.16, numpy 2.5.3.


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


# Sparse vs dense: winner analysis (test, whole abstracts)

Rank of the first relevant paper under BM25 and MiniLM (a paper outside the top 100 counts as 101 when deciding the winner); overlap = share of claim terms present in the relevant paper.

| winner | claims | share | mean overlap | median overlap | mean claim terms missing |
|---|---|---|---|---|---|
| sparse | 87 | 0.290 | 0.559 | 0.545 | 3.770 |
| tie | 140 | 0.467 | 0.697 | 0.714 | 2.843 |
| dense | 73 | 0.243 | 0.430 | 0.400 | 5.164 |

Permutation test, mean overlap of sparse wins vs dense wins: p = 0.0005

## Three clear sparse wins

**Claim 660** (NEI): Ivermectin is used to treat onchocerciasis.
- first relevant paper 1215116: BM25 rank 1, dense rank >100, tf-idf rank 1
- lexical overlap 25%; matched: `onchocerciasi`; missing: `ivermectin us treat`
- BM25 term shares for the paper: onchocerciasi 100%

**Claim 48** (CONTRADICT): A total of 1,000 people in the UK are asymptomatic carriers of vCJD infection.
- first relevant paper 13734012: BM25 rank 2, dense rank >100, tf-idf rank 2
- lexical overlap 44%; matched: `uk carrier vcjd infect`; missing: `total 1 000 peopl asymptomat`
- BM25 term shares for the paper: vcjd 49%, uk 19%, carrier 19%, infect 14%

**Claim 785** (NEI): Microarray results from culture-amplified mixtures of serotypes correlate poorly with microarray results from uncultured mixtures.
- first relevant paper 12471115: BM25 rank 2, dense rank >100, tf-idf rank 2
- lexical overlap 56%; matched: `microarrai result cultur serotyp poorli`; missing: `amplifi mixtur correl uncultur`
- BM25 term shares for the paper: serotyp 44%, microarrai 35%, poorli 10%, cultur 8%, result 4%

## Three clear dense wins

**Claim 1191** (NEI): The amount of publicly available DNA data doubles every 10 years.
- first relevant paper 30655442: BM25 rank >100, dense rank 2, tf-idf rank >100
- lexical overlap 22%; matched: `dna data`; missing: `amount publicli avail doubl everi 10 year`
- closest sentence in embedding space (cosine 0.44): "DNA and RNA sequences are directly submitted from researchers and genome sequencing groups and collected from the scientific literature and patent applications (Fig. 1)."

**Claim 535** (NEI): Hypertension is frequently observed in type 1 diabetes patients.
- first relevant paper 39368721: BM25 rank >100, dense rank 4, tf-idf rank >100
- lexical overlap 29%; matched: `hypertens 1`; missing: `frequent observ type diabet patient`
- closest sentence in embedding space (cosine 0.65): "Blood glucose concentration one hour after a glucose load was an independent predictor of future hypertension."

**Claim 1** (NEI): 0-dimensional biomaterials show inductive properties.
- first relevant paper 31715818: BM25 rank >100, dense rank 5, tf-idf rank >100
- lexical overlap 0%; matched: ``; missing: `0 dimension biomateri show induct properti`
- closest sentence in embedding space (cosine 0.36): "Examples include magnetic nanoparticles and quantum dots for stem cell labeling and in vivo tracking; nanoparticles, carbon nanotubes, and polyplexes for the intracellular delivery of genes/oligonucleotides and protein/peptides; and engineered nanometer-scale scaffolds for stem cell differentiation and transplantation."



# Citation verifier results

## A. SciFact expert labels (claim vs cited paper)

Pairs: {"train": {"cited_no_evidence": 355, "contradict": 194, "support": 370, "wrong_paper": 332}, "test": {"cited_no_evidence": 130, "support": 138, "wrong_paper": 124, "contradict": 71}}. Thresholds tuned on train (macro-F1), scores on test. precision/recall/f1 are for the 'unsupported' flag; macro_f1 averages it with the F1 of the 'supported' class, so flagging everything (first row) does not score well.

| checker | threshold (train) | precision | recall | f1 | macro_f1 | balanced_acc | AUROC | NLI calls |
|---|---|---|---|---|---|---|---|---|
| Baseline: flag every sentence | - | 0.702 | 1.000 | 0.825 | 0.412 | 0.500 | 0.500 | 0% |
| Cosine (MiniLM) | cos < 0.70 | 0.828 | 0.757 | 0.791 | 0.682 | 0.694 | 0.782 | 0% |

Share of test pairs flagged as unsupported, by kind (support = false-alarm rate):

| checker | support (n=138) | contradict (n=71) | cited_no_evidence (n=130) | wrong_paper (n=124) |
|---|---|---|---|---|
| cosine | 0.370 | 0.479 | 0.846 | 0.823 |

Cosine checker: how much evidence to compare against (config.PREMISE_WINDOW, picked on train):

| evidence window | AUROC train | AUROC test | used |
|---|---|---|---|
| title + 1-sentence windows | 0.736 | 0.792 |  |
| title + 2-sentence windows | 0.747 | 0.782 | yes |
| title + 3-sentence windows | 0.745 | 0.771 |  |

NLI rows not run (--no-nli). Run `python -m src.eval.run_verifier_eval` with the model available to add them.

## B. Injected corruptions on generated answers

Answers: `extractive`. Original sentences without a hand label (87 items) are presumed supported.

| checker | answers | precision | recall | f1 | macro_f1 | balanced_acc | accuracy | tp | fp | fn | tn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cosine | extractive | 0.966 | 0.655 | 0.781 | 0.811 | 0.816 | 0.816 | 57 | 2 | 30 | 85 |

| answers | checker | original flagged (n=87) | negate flagged (n=29) | swap_citation flagged (n=29) | add_fabricated flagged (n=29) |
|---|---|---|---|---|---|
| extractive | cosine | 0.023 | 0.034 | 0.931 | 1.000 |

## Failure cases (gold benchmark, test)

### cosine: unsupported claims it let through (most confident first)

- claim 354 vs paper 8774475 (cited_no_evidence), cosine = 0.90
  - claim: Downregulation and mislocalization of Scribble prevents cell transformation and mammary tumorigenesis.
  - closest evidence: Like depletion, mislocalization of Scribble from cell-cell junction was sufficient to promote cell transformation. Interestingly, spontaneous mammary tumors in mice and humans possess both downregulated and mislocalized Scribble.
- claim 759 vs paper 1805641 (contradict), cosine = 0.89
  - claim: Mathematical models predict that using Artemisinin-based combination therapy over nongametocytocidal drugs have a dramatic impact in reducing malaria transmission.
  - closest evidence: Modelling the Impact of Artemisinin Combination Therapy and Long-Acting Treatments on Malaria Transmission Intensity
- claim 274 vs paper 11614737 (contradict), cosine = 0.86
  - claim: Combination nicotine replacement therapies with varenicline or bupropion lead to significantly higher long-term abstinence rates at 52 weeks than varenicline monotherapy.
  - closest evidence: CONCLUSIONS AND RELEVANCE Among cigarette smokers, combined use of varenicline and bupropion, compared with varenicline alone, increased prolonged abstinence but not 7-day point prevalence at 12 and 26 weeks. Neither outcome was significantly different at 52 weeks.

### cosine: supported claims it flagged

- claim 133 vs paper 16280642, cosine = 0.46
  - claim: Assembly of invadopodia is triggered by focal generation of phosphatidylinositol-3,4-biphosphate and the activation of the nonreceptor tyrosine kinase Src.
  - closest evidence: The expression of various phosphoinositide-binding domains revealed that the podosomes in Src-transformed NIH3T3 (NIH-src) cells are enriched with PtdIns(3,4)P2, suggesting an important role of this phosphoinositide in podosome formation. Live-cell imaging analysis revealed that Src-expression stimulated podosome formation at focal adhesions of NIH3T3 cells after PtdIns(3,4)P2 accumulation.
- claim 314 vs paper 4347374, cosine = 0.47
  - claim: Deamination of cytidine to uridine on the minus strand of viral DNA results in catastrophic G-to-A mutations in the viral genome.
  - closest evidence: APOBEC family members also have potent DNA mutator activity through dC deamination; however, whether the editing potential of APOBEC3G has any relevance to HIV inhibition is unknown. Here, we demonstrate that it does, as APOBEC3G exerts its antiviral effect during reverse transcription to trigger G-to-A hypermutation in the nascent retroviral DNA.
- claim 808 vs paper 36606083, cosine = 0.48
  - claim: Most termination events in Okazaki fragments are sequence specific.
  - closest evidence: Many fundamental aspects of DNA replication, such as the exact locations where DNA synthesis is initiated and terminated, how frequently origins are used, and how fork progression is influenced by transcription, are poorly understood. Via the deep sequencing of Okazaki fragments, we comprehensively document replication fork directionality throughout the S. cerevisiae genome, which permits the systematic analysis of initiation, origin efficiency, fork progression, and termination.

