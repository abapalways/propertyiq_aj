# %%
"""
Eval harness for PropertyIQ search (Task 25/26/27/28).

Runs a fixed set of test cases against the EVAL corpus (data/eval_corpus.md,
eval_faiss_index/) - NOT the real listings_corpus.md/faiss_index/ - and
scores search_listings() against known-correct expected results.

Each eval case is designed around a specific listing in eval_corpus.md that
was deliberately built to test one behavior (see eval_corpus.md's comments
for the intent behind each listing).

Run from the scripts/ directory:
    python run_evals.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dotenv import load_dotenv
load_dotenv("../.env")

import json
from datetime import datetime, timezone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from search import search_listings

EVAL_INDEX_PATH = "../eval_faiss_index"

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local(EVAL_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# %%
# --- Eval case definitions ---
# Each case names the specific behavior it tests and which listing(s) prove it.

EVAL_CASES = [
    {
        "id": "poor_schools_excluded",
        "description": "NLI must exclude a listing with explicitly poor schools (3/10) from a 'good schools' search, even though its embedding is topically similar.",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 0,
            "must_haves": ["top-rated schools"], "preferred_city": None,
        },
        "expected_excluded": ["E_SUNSET_203"],
        "expected_included": ["E_MAPLEWOOD_201", "E_QUARRY_207", "E_LINCOLN_206"],
    },
    {
        "id": "needs_work_excluded_from_move_in_ready",
        "description": "NLI must exclude a listing that explicitly needs significant work from a 'move-in ready' search.",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 0,
            "must_haves": ["move-in ready"], "preferred_city": None,
        },
        "expected_excluded": ["E_HARBOR_202"],
        "expected_included": ["E_MAPLEWOOD_201", "E_QUARRY_207"],
    },
    {
        "id": "budget_hard_filter",
        "description": "A listing priced above max_budget must be hard-filtered out, regardless of any other criteria.",
        "buyer_preferences": {
            "max_budget": 450_000, "min_bedrooms": 0,
            "must_haves": [], "preferred_city": None,
        },
        "expected_excluded": ["E_OVERBUDGET_208"],
        "expected_included": ["E_QUARRY_207"],  # exactly at the $450K boundary - must be INCLUDED, not excluded
    },
    {
        "id": "bedroom_hard_filter",
        "description": "A listing below min_bedrooms must be hard-filtered out.",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 3,
            "must_haves": [], "preferred_city": None,
        },
        "expected_excluded": ["E_COMPACT_209"],  # only 2 bedrooms
        "expected_included": ["E_MAPLEWOOD_201"],
    },
    {
        "id": "garage_checkable_attribute",
        "description": "A listing with no garage must be excluded when 'garage' is a stated must-have (checkable attribute, not fuzzy).",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 0,
            "must_haves": ["garage"], "preferred_city": None,
        },
        "expected_excluded": ["E_NOGARAGE_210"],
        "expected_included": ["E_MAPLEWOOD_201"],
    },
    {
        "id": "bm25_named_entity_match",
        "description": "A named, specific school should be found via BM25 keyword matching, not dense/NLI (the phrase won't classify as a sentiment dimension).",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 0,
            "must_haves": ["Lincoln Heights Academy"], "preferred_city": None,
        },
        "expected_included": ["E_LINCOLN_206"],
        "expected_excluded": [],  # no hard exclusion expected; this case checks E_LINCOLN_206 ranks at/near top
        "expect_top_rank": "E_LINCOLN_206",
    },
    {
        "id": "no_enrichment_passes_through",
        "description": "A listing with no school/condition data at all should NOT be excluded by NLI (nothing to contradict) - it should pass through unchecked.",
        "buyer_preferences": {
            "max_budget": 10_000_000, "min_bedrooms": 0,
            "must_haves": ["top-rated schools"], "preferred_city": None,
        },
        "expected_included": ["E_BIRCHFIELD_205"],
        "expected_excluded": [],
    },
]

# %%
# --- Run evals ---

def run_eval_case(case, vectorstore, embeddings, k=10):
    results = search_listings(case["buyer_preferences"], vectorstore, embeddings, k=k)
    result_ids = [doc.metadata.get("listing_id") for doc in results]

    failures = []

    for listing_id in case.get("expected_included", []):
        if listing_id not in result_ids:
            failures.append(f"expected '{listing_id}' in results, but it was missing")

    for listing_id in case.get("expected_excluded", []):
        if listing_id in result_ids:
            failures.append(f"expected '{listing_id}' to be excluded, but it appeared in results")

    if case.get("expect_top_rank"):
        expected_top = case["expect_top_rank"]
        if not result_ids or result_ids[0] != expected_top:
            actual_top = result_ids[0] if result_ids else "(no results)"
            failures.append(f"expected '{expected_top}' to rank #1, but got '{actual_top}'")

    passed = len(failures) == 0
    return passed, failures, result_ids


def run_all_evals():
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [],
    }

    print(f"Running {len(EVAL_CASES)} eval cases against {EVAL_INDEX_PATH}...\n")

    pass_count = 0
    for case in EVAL_CASES:
        passed, failures, result_ids = run_eval_case(case, vectorstore, embeddings)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {case['id']}")
        print(f"   {case['description']}")
        print(f"   Results: {result_ids}")
        if failures:
            for f in failures:
                print(f"   ⚠️  {f}")
        print()

        if passed:
            pass_count += 1

        report["results"].append({
            "id": case["id"],
            "passed": passed,
            "failures": failures,
            "result_ids": result_ids,
        })

    score_str = f"{pass_count}/{len(EVAL_CASES)}"
    report["score"] = score_str
    print(f"{'=' * 40}\nScore: {score_str} ({pass_count / len(EVAL_CASES) * 100:.0f}%)")

    os.makedirs("../evals/reports", exist_ok=True)
    report_filename = f"../evals/reports/{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}.json"
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_filename}")

    return report


if __name__ == "__main__":
    run_all_evals()