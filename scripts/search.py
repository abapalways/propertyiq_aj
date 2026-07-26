"""
Core search logic: takes buyer preferences + a vector store, 
returns a ranked shortlist of matching listings.

Uses a conditional pipeline: if hard filters (garage, yard) already 
narrowed the candidate pool, plain embedding similarity is sufficient. 
If no hard filter applies, the query relies entirely on fuzzy judgment 
(e.g. "top-tier schools"), so a HyDE + cross-encoder pass is added to 
correct for embeddings' weak sentiment/quality discrimination.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder

_llm = ChatOllama(model="qwen3:8b")
_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

CHECKABLE_ATTRIBUTES = {
    "garage": ("garage_spaces", lambda v: v is not None and v >= 1),
    "private yard": ("has_private_yard", lambda v: v is True),
    "yard": ("has_private_yard", lambda v: v is True),
}


def _matches_checkable(must_have_text):
    text_lower = must_have_text.lower()
    for phrase, (key, check_fn) in CHECKABLE_ATTRIBUTES.items():
        if phrase in text_lower:
            return key, check_fn
    return None


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _generate_hyde_text(query):
    prompt = f"""Write exactly 1-2 sentences describing a home that matches 
this buyer request: "{query}". Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
    return _llm.invoke(prompt).content

_last_rrf_scores = {}  # module-level, holds the real scores from the most recent RRF search

def search_listings(buyer_preferences, vectorstore, embeddings, k=3, rejected_listing_ids=None):
    global _last_rrf_scores
    # ... existing code up through candidates/fuzzy_must_haves ...

    if not fuzzy_must_haves:
        return candidates[:k]

    RRF_K = 60
    rrf_scores = {doc.metadata.get("listing_id"): 0.0 for doc in candidates}

    for criterion in fuzzy_must_haves:
        query_vector = embeddings.embed_query(criterion)
        scored = []
        for doc in candidates:
            doc_vector = embeddings.embed_documents([doc.page_content])[0]
            sim = _cosine_similarity(query_vector, doc_vector)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (sim, doc) in enumerate(scored, 1):
            listing_id = doc.metadata.get("listing_id")
            rrf_scores[listing_id] += 1.0 / (RRF_K + rank)

    _last_rrf_scores = rrf_scores  # stash the REAL scores, computed once, here

    doc_by_id = {doc.metadata.get("listing_id"): doc for doc in candidates}
# ... existing RRF scoring code stays the same ...
    doc_by_id = {doc.metadata.get("listing_id"): doc for doc in candidates}
    rrf_ranked_ids = sorted(rrf_scores.keys(), key=lambda lid: rrf_scores[lid], reverse=True)
    rrf_ranked_docs = [doc_by_id[lid] for lid in rrf_ranked_ids]

    # Final precision pass: cross-encoder re-ranks the RRF-ordered candidates
    combined_query = " ".join(fuzzy_must_haves)
    pairs = [[combined_query, doc.page_content] for doc in rrf_ranked_docs]
    cross_scores = _cross_encoder.predict(pairs)
    final_ranked = sorted(zip(cross_scores, rrf_ranked_docs), key=lambda x: x[0], reverse=True)

    return [doc for score, doc in final_ranked[:k]]

def analyze_listings(buyer_preferences, vectorstore, embeddings, k=3, rejected_listing_ids=None):
    if rejected_listing_ids is None:
        rejected_listing_ids = []

    all_docs = list(vectorstore.docstore._dict.values())
    all_docs = [doc for doc in all_docs if doc.metadata.get("listing_id") not in rejected_listing_ids]

    max_budget = buyer_preferences["max_budget"]
    min_bedrooms = buyer_preferences["min_bedrooms"]
    preferred_city = buyer_preferences.get("preferred_city")
    must_haves = buyer_preferences.get("must_haves", [])

    results = []
    candidates_passed = []

    for doc in all_docs:
        listing_id = doc.metadata.get("listing_id")
        price = doc.metadata.get("price")
        bedrooms = doc.metadata.get("bedrooms")

        reasons_failed = []
        if price > max_budget:
            reasons_failed.append(f"over budget (${price:,} > ${max_budget:,})")
        if bedrooms < min_bedrooms:
            reasons_failed.append(f"too few bedrooms ({bedrooms} < {min_bedrooms})")
        if preferred_city and doc.metadata.get("city") != preferred_city:
            reasons_failed.append(f"wrong city ({doc.metadata.get('city')} != {preferred_city})")

        for mh in must_haves:
            match = _matches_checkable(mh)
            if match:
                key, check_fn = match
                if not check_fn(doc.metadata.get(key)):
                    reasons_failed.append(f"fails must-have '{mh}' ({key}={doc.metadata.get(key)})")

        if reasons_failed:
            results.append({"listing_id": listing_id, "tier": "hard_filter_rejected", "reason": "; ".join(reasons_failed), "score": None, "doc": doc})
        else:
            candidates_passed.append(doc)
            results.append({"listing_id": listing_id, "tier": "passed_but_not_selected", "reason": "passed all hard filters", "score": None, "doc": doc})

    fuzzy_must_haves = [mh for mh in must_haves if _matches_checkable(mh) is None]

    if fuzzy_must_haves and candidates_passed:
        RRF_K = 60
        rrf_scores = {doc.metadata.get("listing_id"): 0.0 for doc in candidates_passed}

        for criterion in fuzzy_must_haves:
            query_vector = embeddings.embed_query(criterion)
            scored = []
            for doc in candidates_passed:
                doc_vector = embeddings.embed_documents([doc.page_content])[0]
                sim = _cosine_similarity(query_vector, doc_vector)
                scored.append((sim, doc))
            scored.sort(key=lambda x: x[0], reverse=True)
            for rank, (sim, doc) in enumerate(scored, 1):
                rrf_scores[doc.metadata.get("listing_id")] += 1.0 / (RRF_K + rank)

        for r in results:
            if r["tier"] != "hard_filter_rejected":
                r["score"] = rrf_scores.get(r["listing_id"])
                r["reason"] = f"RRF score: {r['score']:.5f}"

        active_results = [r for r in results if r["tier"] != "hard_filter_rejected"]
        active_results.sort(key=lambda r: r["score"], reverse=True)
        for r in active_results[:k]:
            r["tier"] = "matched"
            r["reason"] = f"selected — {r['reason']}"
    elif candidates_passed:
        for r in results:
            if r["tier"] == "passed_but_not_selected":
                r["tier"] = "matched"
                r["reason"] = "selected (no fuzzy criteria to rank by)"
                k -= 1
                if k <= 0:
                    break

    return results

def _judge_listing_relevance(fuzzy_query, description_text):
    """Ask the LLM a direct yes/no: does this listing's real text genuinely 
    satisfy the fuzzy criterion? Grounded, per-listing judgment - replaces 
    trusting a similarity score alone."""
    prompt = f"""Given this real estate listing description, does it genuinely 
satisfy the buyer's requirement: "{fuzzy_query}"?

Answer with exactly one word on the first line: YES or NO.
If the description does not mention this topic at all, or contradicts it, answer NO.

Listing description:
{description_text}

Answer:"""
    response = _llm.invoke(prompt).content.strip()
    first_line = response.split("\n")[0].upper()
    return "YES" in first_line


def _apply_llm_judgment(ranked_docs, fuzzy_query, k):
    """Filter a ranked list down to only those an LLM confirms genuinely 
    satisfy the fuzzy_query, stopping once k confirmed matches are found."""
    confirmed = []
    for doc in ranked_docs:
        if len(confirmed) >= k:
            break
        if _judge_listing_relevance(fuzzy_query, doc.page_content):
            confirmed.append(doc)
    return confirmed