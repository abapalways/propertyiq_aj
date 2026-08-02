# Task 27: Error Analysis

## Definition of Done

Every failing case is categorized (retrieval miss, wrong tool call, guardrail
miss, fabricated comp/price, latency) with a root cause and prioritized fix.

## Methodology

This analysis is built entirely from live-tested failures discovered during
real search sessions today (not synthetic/hypothetical cases), each verified
via terminal logs, the Agent Trace panel, and/or the eval harness
(`scripts/run_evals.py`) before being recorded here. Several were caught by
the eval suite (Task 25/26); most were caught through direct, repeated
manual testing against both the real corpus and the eval corpus.

## Error catalog

| #   | Case                                                                                                                                                                                                                                                                    | Category                                                                                                   | Root cause                                                                                                                                                                                                                                                                                                                                                                                        | Status                                                                                                                                                                                                                                                                                                                                                   |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | LLM invented school district names, elementary/high school pairings, and letter grades not present in any real listing (e.g. "Eanes ISD," "Westlake High," "A-rated")                                                                                                   | **Fabricated comp/price**                                                                                  | `_search_listings_tool` truncated each listing's description to 400 characters — the `### Key Features & Context` section (containing real school/condition data) fell after that cutoff for every listing tested. The LLM, asked for school info it never received, invented plausible-sounding specifics instead of reporting the gap.                                                          | ✅ **Fixed** — tool now surfaces `dimension_enrichments['school_quality']['raw_text']` and `['condition']['raw_text']` directly in its output, alongside an updated system-prompt instruction requiring the LLM to say "not available" rather than invent when these fields are null. Re-verified clean across 3 independent live queries after the fix. |
| 2   | Original cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-ranking promoted a listing with an explicitly poor (4/10) school district above genuinely good matches                                                                                                             | **Retrieval miss**                                                                                         | The cross-encoder was trained for general relevance, not sentiment/quality discrimination — "poor schools" and "good schools" both score as topically relevant to a "good schools" query.                                                                                                                                                                                                         | ✅ **Fixed** (architectural) — replaced with cosine-similarity dimension classification + NLI contradiction-checking against pre-verified `dimension_enrichments`. Verified across dozens of live test cases with 0 recurrences of this specific failure mode.                                                                                           |
| 3   | NLI scored "needs renovation" backwards — a genuinely needs-work listing came back as `contradiction`, a move-in-ready listing came back as `entailment`                                                                                                                | **Retrieval miss**                                                                                         | The hypothesis template `"This home satisfies the requirement: {criterion}"` was grammatically awkward for verb-phrase criteria ("needs renovation"), degrading NLI reliability specifically for negated/need-based phrasing.                                                                                                                                                                     | ✅ **Fixed** — simplified template to `"This home {criterion}."`. Regression-tested against all previously-passing positive-phrasing cases (no regressions) before confirming the fix.                                                                                                                                                                   |
| 4   | NLI failed to exclude a listing explicitly described as needing significant work ("outdated electrical system, roof at end of life... sold strictly as-is") from a "move-in ready" search                                                                               | **Retrieval miss**                                                                                         | NLI contradiction detection is sensitive to specific phrasing/verbosity, not purely to underlying meaning. Deeper investigation (below) found the true root cause runs even further upstream: `enrich_dimension()`'s LLM call is non-deterministic even at `temperature=0`, so the exact enrichment wording — and therefore the NLI outcome for borderline cases — varies between ingestion runs. | ⚠️ **Attempted, reverted, deeper cause found** — see "Task 28: fix, regression, revert" below.                                                                                                                                                                                                                                                           |
| 5   | BM25 keyword search returns zero matches for "fixer-upper" against a listing whose real text says "diamond in the rough" (same meaning, no literal overlap)                                                                                                             | **Retrieval miss**                                                                                         | BM25 is pure exact-term matching with no synonym/paraphrase understanding — a structural limitation of the technique, not a bug.                                                                                                                                                                                                                                                                  | ⚠️ **Not fixed** — documented, known limitation. Candidate fix: a small synonym-expansion step before BM25 tokenization, or routing very-low-BM25-score keyword criteria through the dimension classifier as a fallback.                                                                                                                                 |
| 6   | `L_PECAN_108`'s ingested `school_quality.raw_text` contains an internal dev/test note ("Price Test: At $455,000, this property sits just above the $450,000 figure used in requirements.md's flagship sample query...") that was never meant to be real listing content | **Fabricated comp/price** (data hygiene, not model behavior)                                               | The dev note was pasted directly into the source markdown file rather than kept in a separate test-planning document, so `extract_dimension_text()` faithfully extracted it as if it were real listing text.                                                                                                                                                                                      | ⚠️ **Not fixed** — deferred. Fix is straightforward (remove the note from the source `.md` file and re-run `ingest.py`), just not yet done.                                                                                                                                                                                                              |
| 7   | Agent Trace panel printed the "Search results" block multiple times for a single search                                                                                                                                                                                 | **Wrong tool call** (display/state bug, not a real extra tool call)                                        | `if tool_name == "_search_listings_tool":` sat outside the `elif msg_type == "ToolMessage":` block it was meant to be scoped to, so it re-fired on every subsequent message in the loop, not just the one that triggered the search.                                                                                                                                                              | ✅ **Fixed** — block moved inside the correct `elif` branch.                                                                                                                                                                                                                                                                                             |
| 8   | Cache hit/miss badge failed to render in the UI despite correct `CACHE HIT`/`CACHE MISS` terminal logs                                                                                                                                                                  | **Latency** (observability gap, not a real perf regression)                                                | `_last_cache_stats` reset inside `search_listings()` was not reliably scoped with `global`, leaving the module-level dict's state inconsistent with what the UI read.                                                                                                                                                                                                                             | ✅ **Fixed** — explicit `global _last_cache_stats` added to both the reset site and the increment site.                                                                                                                                                                                                                                                  |
| 9   | NLI contradiction on one listing caused the `reason` field of five _unrelated, already hard-filtered_ listings to be silently overwritten with that listing's contradiction text                                                                                        | **Retrieval miss** (display/audit-trail correctness, not a ranking bug — final result set was unaffected)  | The `breakdown` update loop matched on `r["tier"] == "hard_filter_rejected"` (matching _every_ previously-rejected listing) instead of `r["listing_id"] == listing_id` (matching only the specific listing being processed).                                                                                                                                                                      | ✅ **Fixed** — corrected to match by `listing_id`.                                                                                                                                                                                                                                                                                                       |
| 10  | `search_listings()` crashed with `NameError: name '_tokenize' is not defined` after a "duplicate code cleanup" accidentally deleted the only real definitions of `_tokenize()` and `_SimpleBM25`                                                                        | **Wrong tool call** (search failed entirely, not a wrong result)                                           | A cleanup pass misidentified genuinely duplicate content (import statements) alongside content that only appeared once (`_tokenize`/`_SimpleBM25`), deleting the latter along with the former.                                                                                                                                                                                                    | ✅ **Fixed** — both restored above their first point of use.                                                                                                                                                                                                                                                                                             |
| 11  | A search run under Torres's buyer profile continued returning a listing he'd rejected moments earlier, even after switching the buyer dropdown to Guest and back                                                                                                        | **Retrieval miss** (initially suspected; ultimately a state-management limitation, not a search-logic bug) | `buyer_profiles` is loaded into memory once at process start; switching buyers in the UI rebuilds the system prompt but does not re-read `buyer_profiles.json` from disk. Only a full process restart reloads current data.                                                                                                                                                                       | ⚠️ **Not fixed** — documented as a known limitation for any long-running server deployment (not just today's testing artifact). Confirmed working correctly across a genuine process restart.                                                                                                                                                            |

## Top 3 prioritized fixes (of the ones still open)

1. **NLI phrasing sensitivity (#4)** — highest priority of the remaining issues, because it's a silent correctness failure with no error or warning; a buyer could be shown a listing that directly contradicts what they asked for, with the system offering no signal anything went wrong.
2. **In-memory buyer-profile staleness (#11)** — highest priority from a deployment-readiness standpoint; this would affect _every_ multi-user session in a real, long-running deployment, not just today's manual testing.
3. **`L_PECAN_108` data contamination (#6)** — lowest effort of the three (a five-minute data edit + re-ingest), and worth doing simply because it's a known, easily-fixed data-quality issue sitting in the corpus.

## Note on the fabrication bug's priority

Item #1 (fabricated school/condition details) is the most severe failure
found today by impact — it produced confidently-stated, false factual
claims shown directly to a buyer — but it is listed as resolved rather than
prioritized here, since it was root-caused and fixed the same day it was
discovered, with the fix re-verified across multiple independent test
queries.

## Task 28: Fix attempted, regression found, reverted — and a deeper root cause

**First fix attempted:** reordered the text fed into the NLI contradiction
check (`_check_contradiction()` in `search.py`), putting the enrichment
sentence before the raw extracted text instead of after
(`f"{enrichment_text} {raw_text}"`).

**Initial result looked clean:** eval suite went from 6/7 to 7/7, with no
regressions on any of the 6 previously-passing cases (after also fixing an
unrelated eval-corpus authoring bug — see below).

**A newly-built LLM-as-judge groundedness eval (extending Task 25) caught
a real regression this fix introduced on the real production corpus,
which the narrower 7-case eval suite never exercised.** Specifically:
`L_PINE_101` — a listing genuinely, unambiguously zoned for a top-rated
school — flipped from a correct `entailment` to an incorrect
`contradiction` against a "top-rated schools" search, after the reordering
fix. This was caught because the groundedness eval runs real queries
through the full chat agent against the real corpus, not just the
isolated eval corpus.

**Decision: reverted the reordering fix.** A regression on a flagship,
extensively-tested real-corpus query was judged more damaging than
leaving the narrower `E_HARBOR_202`-style eval-corpus case unresolved.
Reverting restored `L_PINE_101` to `entailment` and `L_MAPLE_105`/
`L_WALNUT_107` to their correct `contradiction` results, confirmed
against scores identical to the original, extensively-verified baseline.

**Investigating why the same case behaved inconsistently across multiple
test runs led to a deeper finding:** `enrich_dimension()`'s LLM call
(used during ingestion to generate the one-sentence quality judgment) is
**non-deterministic even at `temperature=0`**. Re-running `ingest.py`
twice in a row on the identical source corpus produced different exact
wording for nearly every listing's enrichment sentence (e.g. "The school
district is top-rated" vs. "The school district is excellent" for the
same listing, same input). This means NLI outcomes for borderline cases
(like `E_HARBOR_202`, where contradiction and neutral scores are close)
are not fully controllable by anything in `search.py` alone — the
upstream enrichment text itself isn't stable across ingestion runs.

**Current state:** reverted to the original `raw_text`-first ordering —
the safest, most extensively-verified baseline. The `E_HARBOR_202`-style
borderline case remains open; it is not reliably fixed, but also not
guaranteed to fail on every run, given the underlying non-determinism.

**Not yet done:** caching the enrichment sentence at first generation
(keyed on the `raw_text` it was derived from) and reusing it on
subsequent ingestion runs, rather than regenerating it every time, would
likely resolve the run-to-run instability at its root. This was
identified but not implemented, given time constraints.

**Methodological lesson:** an eval suite is only as trustworthy as its
coverage. The original 7-case suite, built against an isolated synthetic
corpus, could not have caught this regression — it took a second,
differently-designed eval (LLM-as-judge groundedness, running the real
agent against the real corpus) to surface it. Relying on a single eval
type or a single corpus is a real risk; this incident is direct evidence
for using multiple, complementary eval strategies rather than trusting
one green scorecard in isolation.
