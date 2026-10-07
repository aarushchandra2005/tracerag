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

