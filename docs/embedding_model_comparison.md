# Embedding Model Comparison — School Ranking & Negation Handling

## Test setup

- Corpus: 10 synthetic listings (data/listings_corpus.md)
- Filtered candidate pool: 6 listings (price <= $450K, bedrooms == 3)
- Query 1 (ranking quality): "good schools"
- Query 2 (negation/must-have): "3-bed home near good schools with a garage"
- Target: L_PINE_101 (top-rated school, #1 in district) should rank highly
- Known trap: L_BIRCH_110 explicitly has NO garage — should be penalized/excluded

## Model 1: nomic-embed-text (bi-encoder, 768-dim, via Ollama)

### Cosine similarity ranking (query: "good schools")

| Rank | Listing        | Score  | Notes                                               |
| ---- | -------------- | ------ | --------------------------------------------------- |
| 1    | L_WILLOW_109   | 0.4488 | "competitive" schools only                          |
| 2    | L_WALNUT_107   | 0.4419 | "mid-tier, rated 7/10"                              |
| 3    | L_BIRCH_110    | 0.4391 | top-rated, but NO garage                            |
| 4    | L_SPRUCE_106   | 0.4348 | highly-rated district                               |
| 5    | **L_PINE_101** | 0.4260 | **#1 in district — should rank highest, ranks 5th** |
| 6    | L_ELM_124      | 0.4113 | highly-rated district                               |

**Finding:** L_PINE_101, the objectively best school match, ranked last-but-one.
Root cause appears to be lexical bias — plural "schools" scores higher than
"School" as part of a proper noun ("Pinecrest Elementary School"), regardless
of actual quality.

## Model 2: cross-encoder/ms-marco-MiniLM-L-6-v2 (re-ranking layer)

### Cross-encoder ranking (query: "3-bed home near good schools with a garage")

| Rank | Listing        | Score   | Notes                                         |
| ---- | -------------- | ------- | --------------------------------------------- |
| 1    | L_BIRCH_110    | -1.5982 | **NO garage — should be excluded, ranks 1st** |
| 2    | L_WALNUT_107   | -3.1149 |                                               |
| 3    | **L_PINE_101** | -4.2140 | **improved from rank 5 to rank 3**            |
| 4    | L_ELM_124      | -4.4906 |                                               |
| 5    | L_SPRUCE_106   | -6.1960 |                                               |
| 6    | L_WILLOW_109   | -9.4263 |                                               |

**Finding:** Cross-encoder improved L_PINE_101's ranking (5th → 3rd), confirming
it handles nuanced quality judgments better than pure cosine similarity.
However, it did NOT fix the negation problem — L_BIRCH_110 (no garage) still
ranked #1 despite the explicit "with a garage" requirement. Cross-encoders
trained on general relevance (MS MARCO) are not reliable for explicit
negation/must-have detection.

## Conclusion so far

- Ranking quality (fuzzy criteria like "good schools"): cross-encoder > cosine similarity
- Hard requirement/negation detection (garage present/absent): neither approach
  is reliable — this needs exact metadata filtering, not similarity scoring,
  regardless of model sophistication.

## Model 3: gemini-embedding-001 — final conclusion

Initial testing showed gemini-embedding-001 alone outperformed both
nomic-embed-text and bge-m3 on relevance ranking, but a full buyer sweep
revealed a critical gap: for queries with no hard filter narrowing the
candidate pool first (e.g. Henderson's "top-tier schools" with no garage
requirement), Gemini alone still ranked the worst-rated school district
(L_MAPLE_105, 4/10) at #1 — the same sentiment/polarity failure seen
with every other bi-encoder tested tonight.

Adding a HyDE + cross-encoder pass specifically for this scenario fixed
it: L_PINE_101 (best stated rating context) moved to #1, L_MAPLE_105
dropped out of the top 5 entirely, and L_PECAN_108 (10/10, highest
stated rating) surfaced in the top 3 for the first time all night.

### Final architecture (implemented in search.py)

Conditional pipeline based on whether a hard filter (garage/yard)
already ran:

- **Hard filter applied** → plain Gemini embedding similarity search
  (fast, deterministic, proven sufficient — Torres test)
- **No hard filter applied** → full HyDE + Gemini + cross-encoder pass
  (slower, non-deterministic, but corrects the sentiment/polarity
  weakness that plain embeddings cannot solve — Henderson test)

This is a genuinely evidence-backed design: each path was chosen based
on a specific, reproducible test result, not a blanket "safer to always
use the heavy pipeline" assumption. The lighter path is used wherever
it has been proven adequate, keeping the system fast and predictable
by default.

### Remaining known limitations

- Full-pipeline path is non-deterministic (HyDE regenerates text via
  LLM call each run) — same buyer can get slightly different results
  run to run when no hard filter applies.
- has_private_yard regex under-extracts some valid yard descriptions
  (e.g. L_SPRUCE_106), causing false exclusions for yard-requiring buyers.
- Cross-encoder re-ranking is not perfectly calibrated by degree of
  quality — it reliably avoids the worst option but doesn't always
  surface the single best one at #1.
