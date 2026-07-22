# RAG Retrieval Investigation — Findings & Fixes

## Problem

Initial retrieval (pure bi-encoder cosine similarity via nomic-embed-text)
consistently failed two ways:

1. Failed to rank L_PINE_101 (objectively the best school match — "#1 in
   district") near the top for "good schools" queries — it ranked 5th of 6.
2. Failed to detect negation — listings explicitly lacking a must-have
   (e.g. L_BIRCH_110 has no garage, L_WALNUT_107 has no private yard)
   still surfaced as matches, since embeddings can't distinguish
   "has X" from "does not have X" when the same words appear either way.

## Techniques tested (ranking problem)

| Technique                                          | L_PINE_101 rank (of 6) | L_MAPLE_105 rank (worst school, of 6) |
| -------------------------------------------------- | ---------------------- | ------------------------------------- |
| Cosine similarity (nomic-embed-text)               | 5                      | 1 (wrong)                             |
| Cosine similarity (bge-m3, different model)        | 5                      | 1 (wrong)                             |
| HyDE + cosine similarity                           | 2                      | 1 (wrong)                             |
| Contrastive HyDE (good vs bad hypothetical)        | 4                      | 1 (wrong)                             |
| Cross-encoder re-rank alone (ms-marco-MiniLM)      | 3                      | 4 (correct-ish)                       |
| **HyDE + cosine + cross-encoder (final pipeline)** | **1**                  | **3-4 (varies)**                      |

**Conclusion:** two different bi-encoder models produced identical failures,
ruling out "weak model" as the cause. The failure is structural to
bi-encoders comparing independently-computed vectors — they respond to
lexical/topical overlap, not sentiment or quality judgment. A cross-encoder
(reads query+document together) measurably improved results. The winning
pipeline — HyDE-generated query expansion, then bi-encoder narrowing, then
cross-encoder final re-rank — reliably surfaces the correct top match, but
does not perfectly and consistently exclude weaker matches from the rest
of the top-3 (see Known Limitations).

## Negation problem — solved via metadata, not similarity

Tested: bi-encoder (failed), cross-encoder (failed — L_BIRCH_110 still
ranked #1 for a garage query). Similarity-based techniques, however
sophisticated, could not reliably detect negation.

**Fix:** extract objective, stated facts into structured metadata during
ingestion, with explicit negation-aware regex (checks "does not have"
before checking presence). Added fields: `garage_spaces` (int/None),
`has_private_yard` (bool/None), `has_hoa` (bool), `city`, `bathrooms`,
`square_footage`. `search_listings()` now checks these via exact filters
for any must-have that maps to a known checkable attribute, before
falling back to fuzzy search for the remainder.

## Design principle established

- Objective, stated fact (garage, yard, HOA) → metadata, exact filter
- Partially stated (school rating — some listings give a number) →
  hybrid, not yet implemented
- Genuinely subjective / unverifiable from corpus (e.g. "safe
  neighborhood") → stays in fuzzy/RAG path. Note: "safe neighborhood"
  appears specifically in the Anonymous Guardrail Test Profile — likely
  intentional bait for Week 3's discrimination-guardrail testing, not
  just an ordinary fuzzy preference.

## Known limitations (unresolved)

1. **Non-determinism:** HyDE regenerates hypothetical text via LLM call
   on every search, so the same buyer/preferences can yield slightly
   different rankings across runs. Observed across 3 repeated Torres
   test runs: L_PINE_101 stayed #1 consistently, but L_MAPLE_105
   (worst school) inconsistently appeared in #2 or #3.
2. **Regex under-extraction:** `has_private_yard` only catches specific
   phrasing ("private/fenced/expansive/large yard/backyard") — listings
   using other wording (e.g. "oversized backyard with mature oak trees")
   return `None` rather than `True`, causing false exclusions when a
   buyer requires a yard (confirmed: L_SPRUCE_106 wrongly excluded for
   Vance profile).
3. **`city` field never enforced as a filter** — L_WILLOW_109 (Round
   Rock, not Austin) still surfaces despite most buyers preferring
   Austin.
4. **Rejected listings (memory) not yet integrated** — L_ELM_124 still
   surfaces for Torres despite being previously rejected in their
   session history. Deferred to Week 2, Task 15 by design.

## Next steps

- Week 2: wire session_history.rejected_listings into search_listings()
- Consider hybrid school_rating metadata (numeric where corpus states
  one, fuzzy fallback otherwise)
- Consider enforcing `city` as a hard filter
- Consider fixing HyDE's temperature/determinism if consistency matters
  more than variety
