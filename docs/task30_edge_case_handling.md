# Task 30: Edge Case Handling

## Definition of Done

Each edge case produces a graceful fallback instead of a crash.

## Edge case 1: Comps tool failure

**Test:** "Can you find comparable sold homes for listing L_FAKE_999" (a
listing ID that was never searched for, with no metadata in context).

**Result:** ✅ Graceful. The LLM recognized it had no metadata for this
listing and asked the buyer directly for city, bedrooms, and square
footage, rather than calling `get_comps()` with incomplete data or
inventing plausible-sounding comps.

**Caveat:** `get_comps()`'s own explicit null-handling
(`if target_listing_metadata is None: return {"comps": [], "message": ...}`)
was not directly exercised by this test, since the LLM avoided the call
entirely rather than triggering the tool's fallback path. This is a
reasonable, arguably better outcome for the buyer, but it means the tool's
own error-return path remains formally unverified from a live chat
session — it has only been confirmed by direct unit-style testing earlier
in the project.

**A separate, real, confirmed failure — now fixed:** a transient Google
Generative AI embedding API `503 UNAVAILABLE` error, encountered live
during testing, was NOT originally caught anywhere in the pipeline. It
propagated all the way up through `classify_fuzzy_dimension()` →
`_check_contradiction()` → `search_listings()` → `_search_listings_tool()`
→ Gradio's event handler, surfacing as a raw Python traceback shown
directly to the user, with no graceful message and no retry.

**Fix applied:** a `SearchUnavailableError` exception was added to
`search.py`, raised by all three embedding-API call sites
(`_get_doc_embedding()`, `_get_query_embedding()`,
`classify_fuzzy_dimension()`) on any embedding API failure.
`_search_listings_tool()` in `chat_agent.py` catches this specifically and
returns a structured `{"error": "..."}` result instead — the same pattern
`get_comps()` already uses for its own "listing not found" case — so the
LLM can phrase a natural, graceful message to the buyer instead of the
request crashing outright.

**Not independently re-verified:** since the original failure was a
transient third-party API outage, it cannot be reliably forced on demand
to confirm the fix triggers correctly end-to-end. The fix has been
verified for syntactic correctness and does not affect the normal
successful-search code path (confirmed via a standard search after
applying the change), but the actual exception-handling branch itself has
not been exercised by a real API failure since the fix was applied.

## Edge case 2: No listings match criteria

**Test:** "Find me a 6-bedroom home in Austin under $50,000" (constraints
guaranteed to return zero results against the corpus).

**Result:** ✅ Graceful. The system correctly reported that no matching
listings exist, and proactively suggested reasonable next steps (broaden
price range, adjust city, adjust bedroom count) rather than returning an
empty or confusing response.

## Edge case 3: Ambiguous rejection reference

**Test:** "I don't want that one." sent as the first message of a fresh
Guest session, with no prior search results in context for "that one" to
resolve against.

**Result:** ✅ Graceful. The system correctly recognized the ambiguity and
asked for the specific listing ID or description, rather than guessing,
hallucinating a listing to reject, or calling `reject_listing` with
incorrect or fabricated arguments.

**Not yet tested:** the same scenario for a real buyer profile (not
Guest), where `reject_listing`'s actual persistence path is involved. The
Guest-session test confirms the LLM's reference-resolution judgment is
sound, but doesn't exercise the real buyer-scoped rejection tool under
ambiguity.

## Summary

| Edge case                                          | Status                                                         |
| -------------------------------------------------- | -------------------------------------------------------------- |
| Comps tool failure (bad/missing listing reference) | ✅ Graceful (LLM avoided the failure path directly)            |
| Embedding API transient failure (503)              | ✅ Fixed (not independently re-verified against a live outage) |
| No listings match criteria                         | ✅ Graceful                                                    |
| Ambiguous rejection reference                      | ✅ Graceful (Guest session; real-buyer path untested)          |

## Not yet done / follow-up

1. Directly exercise `get_comps()`'s `None`-metadata return path from a
   live chat session (harder to force reliably, since it depends on the
   LLM's own decision to call the tool with incomplete data).
2. Re-test ambiguous rejection reference against a real buyer profile,
   to confirm `reject_listing` is never called with guessed/incorrect
   arguments under ambiguity.
3. If a real embedding-API outage occurs again, confirm the
   `SearchUnavailableError` handling path actually produces the intended
   graceful message rather than a traceback, since this could not be
   forced on demand for testing.
