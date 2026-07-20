# Task 7: Retrieval Test Log

## Query

"3-bed homes under $450K near good schools"

## Approach A: Pure semantic search (no metadata filtering)

Top-3 results: L_WALNUT_107 ($385K), L_PECAN_108 ($455K), L_MAPLE_105 ($415K, 4-bed)

**Judgment: FAIL**

- L_PECAN_108 violates the $450K budget constraint outright.
- L_MAPLE_105 violates the 3-bedroom constraint (it's 4-bed) and is zoned
  for the lowest-rated school district in the corpus (4/10) — nearly the
  opposite of "good schools."
- L_PINE_101, the clearest actual match (3-bed, $425K, adjacent to a
  #1-ranked elementary school), does not appear in the top 3 at all.
- Root cause: embedding similarity has no concept of numeric comparison
  (< or <=), so it cannot reliably enforce hard constraints like price
  or bedroom count.

## Approach B: Metadata filter first, then semantic search on survivors

Filter (price <= 450000 AND bedrooms == 3) → 6 candidates:
L_PINE_101, L_ELM_124, L_SPRUCE_106, L_WALNUT_107, L_WILLOW_109, L_BIRCH_110

Semantic search ("good schools") on those 6 → top-3:
L_WILLOW_109 ($399K), L_WALNUT_107 ($385K), L_BIRCH_110 ($430K)

**Judgment: PARTIAL PASS**

- All hard constraints (price, bedrooms) are correctly satisfied — a
  major improvement over Approach A.
- However, the semantic ranking is not exhaustive/reliable: L_PINE_101
  (adjacent to a #1-ranked elementary) and L_ELM_124 (highly-rated
  district) — arguably the two strongest "good schools" matches among
  the 6 candidates — did not surface in the top 3.
- Likely cause: lexical/surface-level bias. L_WILLOW_109 and L_WALNUT_107
  both literally contain the plural word "schools," while L_PINE_101 and
  L_ELM_124 reference "School" as part of a proper noun (e.g. "Pinecrest
  Elementary School") — the embedding model appears to weight word-form
  overlap alongside true semantic meaning, not perfectly separating the two.
- Implication: pure vector retrieval alone is not sufficient to reliably
  rank fuzzy/qualitative criteria. This points toward needing an LLM
  reasoning layer (or a broader k + re-ranking step) on top of raw
  retrieval — likely relevant for Week 1's Task 8 prototype.

## Note

One additional gap identified during this task: `city` was never extracted
into metadata during ingestion (Task 6). L_WILLOW_109 (Round Rock, not
Austin) was not excluded by either approach despite most buyer profiles
specifying `preferred_city: Austin`. To be added as a metadata field in
a future revision of ingest.py.
