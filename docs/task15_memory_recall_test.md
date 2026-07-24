# Task 15: Memory Recall Test — "Don't Show Rejected Listings Again"

## Definition of Done
Session 1 rejects a listing; Session 2 (same buyer) does not surface it.

## Test 1 — Pre-existing rejection (Task 4 seed data)
Buyer: The Torres Family (B001)
Pre-existing rejection on file: L_ELM_124, reason "Yard is too small", 
recorded 2026-07-11.

Ran search_listings() fresh with Torres's real preferences. Result: 
L_ELM_124 did not appear in results, in the debug analysis tier 
breakdown, or anywhere in the candidate pool.

## Test 2 — Live rejection, same session
Within a chat_agent.py conversation, after receiving a shortlist 
including L_WALNUT_107, said "I don't want L_WALNUT_107, the HOA fee 
is too high." Agent correctly called reject_listing(buyer_id="B001", 
listing_id="L_WALNUT_107", reason="HOA fee is too high"). Verified via 
buyer_profiles.json: a new entry was appended, alongside L_ELM_124, 
no duplicates, persisted to disk.

Follow-up "show me updated listings" correctly excluded both.

## Test 3 — Cross-session persistence
Restarted the app entirely (new process). Selected Torres again. Both 
rejections remained excluded, confirming durable disk persistence.

## Test 4 — Undo and reset operations
Added additional test rejections for B001 (L_PINE_101, L_SPRUCE_106) 
via tools.py. Ran undo_last_rejection("B001") - correctly restored 
L_SPRUCE_106 (the most recently added). Ran reset_rejected_listings("B001") 
- correctly cleared all remaining rejections, confirmed via disk read. 
Original seed data (L_ELM_124) was manually restored afterward since 
this test intentionally wiped it.

## Test 5 — Agent trace verification (Task 16 crossover)
Live chat session showed, via the Agent Trace panel, the actual 
_search_listings_tool call including rejected_listing_ids=['L_ELM_124', 
'L_SPRUCE_106'] being passed in - direct visual proof that memory was 
read and applied, not just inferred from output.

## Conclusion
All tests pass. Memory persists correctly across pre-seeded data, live 
natural-language rejections, full application restarts, and supports 
undo/reset operations. Verified both via direct file inspection and via 
the agent trace panel showing memory being actively used in tool calls.
