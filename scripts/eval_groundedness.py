# %%
"""
LLM-as-judge groundedness eval (Task 25/27 extension).

Unlike run_evals.py (which calls search_listings() directly and checks
listing IDs deterministically), this script runs the FULL chat agent -
LLM tool-calling + final response generation - then uses a separate LLM
call to judge whether every factual claim in the generated response is
actually grounded in the tool's real output.

This specifically targets the failure mode that assertion-based evals
cannot catch: a response that lists the "correct" listings, but describes
them with fabricated details (the exact bug found and fixed earlier - see
docs/task27_error_analysis.md, item #1).

Run from the scripts/ directory:
    python eval_groundedness.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dotenv import load_dotenv
load_dotenv("../.env")

import json
from datetime import datetime, timezone
from langchain_core.messages import HumanMessage, SystemMessage

from chat_agent import llm_with_tools, build_system_prompt, TOOL_MAP

# %%
# --- Test queries designed to probe groundedness specifically ---
# Each one asks about details (school, condition) that live in real,
# verifiable tool output - exactly the kind of claim that got fabricated
# before the Task 27 fix.

GROUNDEDNESS_TEST_QUERIES = [
    "Show me homes with top-rated schools, no budget limit.",
    "Find me a 3-bedroom home in Austin under $450,000 with a garage.",
    "Show me homes that are move-in ready, no budget limit.",
]

JUDGE_PROMPT_TEMPLATE = """You are a strict fact-checker reviewing a real estate 
assistant's response for fabrication.

Below is the RAW TOOL OUTPUT the assistant received (the ground truth - the only 
real data available), and the assistant's FINAL RESPONSE to the buyer.

Your job: identify any factual claim in the response about a specific listing 
(school names, ratings, letter grades, condition details, neighborhood facts, 
or any other specific detail) that does NOT appear in or is not a reasonable 
paraphrase of the raw tool output. Prices, addresses, bedroom/bathroom counts, 
and square footage are structured fields already present in every tool result - 
do not flag these as unsupported unless the response states an amount that 
contradicts the tool data.

RAW TOOL OUTPUT:
{tool_output}

ASSISTANT'S FINAL RESPONSE:
{final_response}

Respond with ONLY a JSON object, no other text, in this exact format:
{{
  "grounded": true or false,
  "unsupported_claims": ["claim 1 that isn't in the tool output", "claim 2", ...],
  "reasoning": "one or two sentence explanation"
}}

If there are no unsupported claims, "grounded" should be true and 
"unsupported_claims" should be an empty list."""


def run_query_through_agent(query, buyer=None):
    """Runs one query through the real chat agent (LLM + tool calling),
    returns the raw tool output and final response text."""
    system_prompt = build_system_prompt(buyer)
    messages = [SystemMessage(system_prompt), HumanMessage(query)]

    response = llm_with_tools.invoke(messages)
    messages.append(response)

    tool_outputs = []
    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_fn = TOOL_MAP[tool_call["name"]]
            result = tool_fn(**tool_call["args"])
            tool_outputs.append({"tool": tool_call["name"], "args": tool_call["args"], "result": result})
            messages.append({"role": "tool", "content": json.dumps(result, default=str), "tool_call_id": tool_call["id"]})
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return tool_outputs, response.content


def judge_groundedness(tool_outputs, final_response, judge_llm):
    """Uses a separate LLM call to check whether final_response's factual
    claims are all grounded in tool_outputs. Returns the parsed judge verdict."""
    tool_output_str = json.dumps(tool_outputs, default=str, indent=2)
    prompt = JUDGE_PROMPT_TEMPLATE.format(tool_output=tool_output_str, final_response=final_response)

    judge_response = judge_llm.invoke(prompt).content.strip()

    # Strip markdown code fences if the judge wrapped its JSON in them
    if judge_response.startswith("```"):
        judge_response = judge_response.split("```")[1]
        if judge_response.startswith("json"):
            judge_response = judge_response[4:]
        judge_response = judge_response.strip()

    try:
        return json.loads(judge_response)
    except json.JSONDecodeError:
        return {"grounded": None, "unsupported_claims": [], "reasoning": f"Judge output was not valid JSON: {judge_response[:200]}"}


# %%
def run_groundedness_evals():
    from langchain_groq import ChatGroq
    # NOTE: using the same model family as the system under test (ChatGroq
    # openai/gpt-oss-120b) is a known limitation - a model judging its own
    # outputs is less reliable than an independent judge model. Swap in a
    # different provider/model here if one is available, for a stronger check.
    judge_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "results": []}
    pass_count = 0

    print(f"Running {len(GROUNDEDNESS_TEST_QUERIES)} groundedness checks...\n")

    for query in GROUNDEDNESS_TEST_QUERIES:
        print(f"Query: {query!r}")
        tool_outputs, final_response = run_query_through_agent(query)

        if not tool_outputs:
            print("  (no tool call made - skipping groundedness check)\n")
            continue

        verdict = judge_groundedness(tool_outputs, final_response, judge_llm)
        status = "✅ GROUNDED" if verdict.get("grounded") else "❌ FABRICATION DETECTED"
        print(f"  {status}")
        print(f"  Reasoning: {verdict.get('reasoning')}")
        if verdict.get("unsupported_claims"):
            for claim in verdict["unsupported_claims"]:
                print(f"    ⚠️  Unsupported: {claim}")
        print()

        if verdict.get("grounded"):
            pass_count += 1

        report["results"].append({
            "query": query,
            "final_response": final_response,
            "verdict": verdict,
        })

    total = len([r for r in report["results"]])
    score_str = f"{pass_count}/{total}" if total else "0/0"
    report["score"] = score_str
    print(f"{'=' * 40}\nGroundedness score: {score_str}")

    os.makedirs("../evals/reports", exist_ok=True)
    report_filename = f"../evals/reports/groundedness_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_filename}")

    return report


if __name__ == "__main__":
    run_groundedness_evals()