# Task 14: Memory Schema Design

## Memory types used, mapped to the four agentic memory categories

### Semantic memory — standing facts about the buyer

**Storage:** `buyer_profiles.json` → `preferences` object
**Fields:** `min_bedrooms`, `max_budget`, `must_haves`, `preferred_city`
**Read path:** Loaded once per session into the agent's system prompt
(`build_system_prompt()` in chat_agent.py), and passed directly into
`search_listings()` as `buyer_preferences`.
**Write path:** Not yet implemented as a live tool (buyer profiles are
currently static). Planned extension: an `update_preference` tool,
mirroring `reject_listing`'s pattern, so a buyer stating a new
requirement mid-conversation ("actually I need 4 bedrooms") updates
`preferences` directly, persisted to disk.

### Episodic memory — specific, dated events

**Storage:** `buyer_profiles.json` → `session_history.rejected_listings`
**Schema per entry:** `{listing_id, reason, timestamp}`
**Read path:** Loaded into the system prompt as a list of excluded IDs,
and passed to `search_listings()`/`analyze_listings()` as
`rejected_listing_ids` — filtered out before any ranking logic runs.
**Write path:** `reject_listing(buyer_id, listing_id, reason)` tool.
Called by the LLM when it recognizes a natural-language rejection in
conversation (e.g. "I don't want that one, the HOA is too high").
Includes a duplicate guard (won't re-add an already-rejected listing)
and writes back to buyer_profiles.json immediately — real persistence,
not session-only.

### Procedural memory — behavioral rules

**Storage:** System prompt text (prompts/system_prompt.md, and the
per-buyer prompt built in build_system_prompt())
**Examples:** Fair-housing compliance rules (never filter/steer on
protected characteristics), instructions to never fabricate numbers,
instructions to ask rather than assume when a guest has no preferences
on file.
**Note:** Not currently learned/adapted automatically (e.g. no
"buyer rejects garage-less homes 3x in a row, weight garage more
heavily" logic) — the rules are hand-written, static instructions,
not derived from behavior. A genuinely adaptive procedural memory
system is out of scope for this project's timeline.

### Working memory — current conversation context

**Storage:** `messages_state` (a gr.State holding the full LangChain
message list) — lives only for the duration of one browser session,
lost on refresh/restart.
**Contains:** every HumanMessage, AIMessage (including tool-call
requests), and ToolMessage in order — the complete, literal transcript
the LLM re-reads on every turn. This is what enables cross-turn
reference resolution (e.g. "the first listing" correctly resolving to
a specific listing_id/price from several messages earlier).

## Design decision: read-before-write ordering

On every search call, rejected_listing_ids are filtered out of
all_docs before any hard or fuzzy filtering logic runs — a rejected
listing never enters the candidate pool at any stage, rather than
being filtered out at the end. This ensures a rejected listing cannot
accidentally reappear due to a bug in a later pipeline stage.

## Verified test case (Task 15)

Buyer: The Torres Family (B001). Pre-existing rejection: L_ELM_124
(from initial test data, Task 4). New rejection recorded live via
natural conversation during this session: L_WALNUT_107 ("HOA fee is
too high"). Confirmed via cat buyer_profiles.json: both entries
persisted correctly, no duplicates. Confirmed via live chat: a
follow-up "show me updated listings" request correctly excluded both,
promoting the next-best candidate (L_SPRUCE_106) into the shortlist.
