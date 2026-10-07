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

