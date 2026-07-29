# Task 17: Guardrail Rules Checklist

Each rule below is mapped directly to requirements.md, Section 5 (Guardrail Requirements).

## Rule 1: No protected-characteristic filtering or steering

**Source:** "Must refuse any search, filter, or recommendation criterion based on
protected characteristics (race, religion, national origin, familial status,
disability, etc.) per fair-housing principles; no steering."

**Implementation requirement:** Before any search executes, the buyer's request
must be checked for explicit or proxy references to protected characteristics
(e.g. "fewer immigrant families," "family-friendly" used as a demographic proxy,
"safe neighborhood" when used to imply racial composition). If detected, refuse
the specific criterion, explain why, and offer a legitimate alternative (price,
size, commute, amenities). This check happens BEFORE the query reaches the vector
store or search_listings() - not as a post-hoc filter on results.

**Status:** Not yet implemented. Current system prompt (Week 1) states this
principle in prose but has no enforced code-level check.

## Rule 2: No fabricated comps, prices, or mortgage figures

**Source:** "Must never fabricate comps, prices, or mortgage figures; all numeric
claims must come from a tool call or cited RAG source."

**Implementation requirement:** Every price, comp, or mortgage number in an agent
response must be traceable to an actual calc_mortgage/get_comps tool result or
real listing metadata - never invented or estimated by the LLM directly.

**Status:** Partially addressed. Week 2 testing found and fixed a related,
adjacent failure (LLM fabricating school-district ratings/names not present in
tool output) via explicit grounding instructions in the system prompt. The same
principle needs to be verified specifically for price/comp/mortgage figures,
which haven't yet been stress-tested for fabrication the way school data was.

## Rule 3: Affordability estimates must be labeled as approximate

**Source:** "Must clearly label affordability estimates as approximate and
dependent on the buyer's actual credit/financial situation, not a guaranteed
loan offer."

**Implementation requirement:** Any calc_mortgage output presented to the buyer
must include a disclaimer that it's an estimate, not a guaranteed offer.

**Status:** Partially addressed - calc_mortgage's docstring scopes it as
"principal + interest only," and LLM responses tonight have organically included
caveats ("this is an estimate assuming a fixed-rate mortgage..."). Not yet
enforced as a hard requirement in the system prompt.

## Rule 4: Respect stored rejections/preferences

**Source:** "Must respect stored buyer rejections/preferences and not resurface
explicitly rejected listings without the user re-requesting them."

**Status:** ✅ Implemented and tested (Week 2, Task 15) - rejected_listing_ids
filtering happens before any ranking logic runs, verified across multiple
sessions and app restarts.

## Rule 5: Log every filter and flag protected-characteristic pattern matches

**Source:** "Observability must log every filter applied to a search and
flag/reject any filter matching a protected-characteristic pattern, for
compliance review."

**Implementation requirement:** Every search's actual filter criteria (hard
filters + fuzzy terms) must be logged with enough detail for compliance review,
and any protected-characteristic pattern match must be flagged distinctly from
normal filters.

**Status:** Partially addressed - the Agent Trace panel (Week 2, Task 16) already
logs every tool call's arguments, including search filters. Does not yet
distinguish/flag protected-characteristic pattern matches specifically.

## Summary

- 1 of 5 rules fully implemented (Rule 4, rejection memory)
- 2 of 5 rules partially implemented (Rules 3, 5 - existing infrastructure covers
  part of the requirement)
- 2 of 5 rules not yet implemented (Rules 1, 2 - these are Task 18's primary focus)
