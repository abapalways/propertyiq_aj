# Task 24: Observability — LangSmith Tracing

## Definition of Done

Every search's filters and any guardrail trigger are logged with a trace ID.

## Implementation

Observability is provided by [LangSmith](https://smith.langchain.com), via
LangChain's built-in tracing integration — no custom instrumentation code
was needed, since `chat_agent.py` already uses `llm.bind_tools()` and
`llm.invoke()` throughout, and LangSmith auto-instruments these calls when
tracing is enabled via environment variables:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=PropertyIQ_AJ
LANGCHAIN_API_KEY=<key>
```

These are read via `load_dotenv("../.env")` at the top of both `ingest.py`
and `chat_app.py`, and confirmed printed at every app startup
(`TRACING_V2: true`, `PROJECT: PropertyIQ_AJ`).

## What gets captured, per trace

Each LLM invocation (including every tool-calling round) is captured as its
own trace, with a unique trace ID, containing:

- The full system prompt in effect for that turn
- The user's message
- Any tool call the LLM issued, with its exact structured arguments
  (e.g. `_search_listings_tool(k=3, max_budget=10000000, min_bedrooms=0,
must_haves=["move-in ready"], preferred_city=null, rejected_listing_ids=null)`)
- The tool's real JSON result
- The LLM's final response
- Latency and token/cost breakdown for that specific call

This means every search's applied filters (`max_budget`, `min_bedrooms`,
`must_haves`, `preferred_city`, `rejected_listing_ids`) are captured
automatically as structured tool-call arguments on every trace — satisfying
this task's requirement without any additional logging code.

Guardrail checks (`_fair_housing_guard.validate()`, `_on_topic_guard.validate()`
in `chat_app.py`) run as plain Python function calls _before_ the traced
LLM invocation begins, so a blocked message never reaches a LangSmith trace
at all — which is itself the correct behavior (no search or LLM call should
occur for a blocked request) but means guardrail _blocks_ are not currently
visible inside LangSmith the way search filters are. This is a known gap;
see "Not yet done" below.

## Evidence

Project-level view, confirming tracing has been active throughout today's
testing session:

- **Project:** `PropertyIQ_AJ`
- **Trace count (7 days):** 754
- **Error rate (7 days):** ~0%
- **P50 latency:** 0.63s
- **P99 latency:** 25.92s
- **Total tokens (7 days):** 724,344
- **Total cost (7 days):** $0.20

Individual trace, exported as evidence of a single search's full detail
(system prompt, user message, tool call with exact filter arguments, tool
result, final response, latency, and cost):

- Query: _"Show me homes that are move-in ready, no budget limit."_
- Tool call: `_search_listings_tool(k=3, max_budget=10000000, min_bedrooms=0, must_haves=["move-in ready"], preferred_city=null, rejected_listing_ids=null)`
- Latency: 2.54s | Tokens: 2.6K | Cost: $0.0008
- Trace: viewable in the `PropertyIQ_AJ` LangSmith project (see "How to
  reproduce this evidence" below); not linked directly here since this
  repository is public and the link would expose the LangSmith
  organization's internal project URL.

## How to reproduce this evidence

Anyone with access to the project's LangSmith account can view equivalent
traces directly:

1. Log into [smith.langchain.com](https://smith.langchain.com)
2. Open the `PropertyIQ_AJ` project under Tracing
3. Any `ChatGroq` trace shows the full detail described above — system
   prompt, user message, tool call arguments, tool result, final response,
   latency, and cost.

## Not yet done

- **Guardrail blocks are not currently traced.** Since `_fair_housing_guard.validate()`
  and `_on_topic_guard.validate()` run before any LLM call, a blocked
  request produces no LangSmith trace at all. A future improvement would
  wrap the guardrail check itself in a traced span (e.g. via
  `@traceable` from `langsmith`), so blocked requests are visible in the
  same project alongside successful searches, rather than only appearing
  in terminal output.
- **The 25.92s P99 latency outlier has not yet been root-caused.** Likely
  candidates, based on today's testing: a Google embedding API 503 retry,
  or a turn involving multiple sequential tool calls (e.g. the LLM trying
  "needs renovation" then "fixer-upper" as two separate searches before
  answering). Worth identifying the specific trace and investigating
  further as a Stretch Goal (latency budget) follow-up.
