# Task 8: Prototype Test Log — Criteria → Shortlist

## Buyer tested

B001, The Torres Family
Preferences: min_bedrooms=3, max_budget=450000,
must_haves=["good school district", "garage"], preferred_city=Austin

## Result

The program ran successfully with no crash, and the returned shortlist
does reflect real listing data pulled from the corpus (3 results, all
correctly within budget and bedroom count).

However, the results were not up to the mark:

1. L_PINE_101 — the listing explicitly identified as Torres's ideal
   match in the dataset's own test notes (3-bed, $425K, adjacent to a
   #1-ranked elementary school) — did not appear in the top 3 at all.

2. L_BIRCH_110 was returned as result #3, despite its corpus entry
   explicitly stating it does NOT have a garage — one of Torres's two
   stated must-haves. The system silently recommended a listing that
   fails an explicit hard requirement.

## Root cause

Both issues trace back to the decision (made for simplicity in this
task) to treat all must_haves as one joined fuzzy semantic-search
string ("good school district garage") rather than splitting checkable
items like "garage" into a hard boolean metadata filter. Because the
word "garage" still appears elsewhere in L_BIRCH_110's text (describing
its driveway as an alternative), it scored well enough on lexical/
semantic similarity to rank despite failing the actual requirement.

## Next step (not part of this task, noted for later)

Must-haves should likely be split into two categories at the schema
level: hard/checkable (garage, HOA, yard) vs. genuinely fuzzy (school
quality, "family-friendly" feel) — hard ones enforced as metadata
filters, not folded into the semantic search string.
