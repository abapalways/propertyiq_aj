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


def search_listings(buyer_preferences, vectorstore, embeddings, k=3):
    all_docs = list(vectorstore.docstore._dict.values())
    max_budget = buyer_preferences["max_budget"]
    min_bedrooms = buyer_preferences["min_bedrooms"]
    must_haves = buyer_preferences.get("must_haves", [])

    candidates = [
        doc for doc in all_docs
        if doc.metadata.get("price") <= max_budget
        and doc.metadata.get("bedrooms") >= min_bedrooms
    ]

    hard_filter_applied = False
    fuzzy_must_haves = []
    for mh in must_haves:
        match = _matches_checkable(mh)
        if match:
            key, check_fn = match
            candidates = [doc for doc in candidates if check_fn(doc.metadata.get(key))]
            hard_filter_applied = True
        else:
            fuzzy_must_haves.append(mh)

    if not candidates:
        return []

    if not fuzzy_must_haves:
        return candidates[:k]

    fuzzy_query = " ".join(fuzzy_must_haves)

    if hard_filter_applied:
        # Simple path: a hard filter already reduced risk of a bad match dominating
        filtered_vectorstore = FAISS.from_documents(candidates, embeddings)
        results = filtered_vectorstore.similarity_search(fuzzy_query, k=k)
        return results

    # No hard filter applied - full pipeline to guard against sentiment/quality errors
    hyde_text = _generate_hyde_text(fuzzy_query)
    hyde_vector = embeddings.embed_query(hyde_text)

    scored = []
    for doc in candidates:
        doc_vector = embeddings.embed_documents([doc.page_content])[0]
        sim = _cosine_similarity(hyde_vector, doc_vector)
        scored.append((sim, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    narrowed = [doc for sim, doc in scored]

    pairs = [[hyde_text, doc.page_content] for doc in narrowed]
    cross_scores = _cross_encoder.predict(pairs)
    reranked = sorted(zip(cross_scores, narrowed), key=lambda x: x[0], reverse=True)

    return [doc for score, doc in reranked[:k]]
def analyze_listings(buyer_preferences, vectorstore, embeddings):
    all_docs = list(vectorstore.docstore._dict.values())
    max_budget = buyer_preferences["max_budget"]
    min_bedrooms = buyer_preferences["min_bedrooms"]
    must_haves = buyer_preferences.get("must_haves", [])

    results = []
    hard_filter_applied = False
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

        for mh in must_haves:
            match = _matches_checkable(mh)
            if match:
                key, check_fn = match
                hard_filter_applied = True
                if not check_fn(doc.metadata.get(key)):
                    reasons_failed.append(f"fails must-have '{mh}' ({key}={doc.metadata.get(key)})")

        if reasons_failed:
            results.append({
                "listing_id": listing_id, "tier": "hard_filter_rejected",
                "reason": "; ".join(reasons_failed), "score": None, "doc": doc,
            })
        else:
            candidates_passed.append(doc)
            results.append({
                "listing_id": listing_id, "tier": "passed_but_not_selected",
                "reason": "passed all hard filters", "score": None, "doc": doc,
            })

    fuzzy_must_haves = [mh for mh in must_haves if _matches_checkable(mh) is None]

    if fuzzy_must_haves and candidates_passed:
        fuzzy_query = " ".join(fuzzy_must_haves)

        if hard_filter_applied:
            # SIMPLE PATH - matches search_listings' simple branch
            query_vector = embeddings.embed_query(fuzzy_query)
            for r in results:
                if r["tier"] != "hard_filter_rejected":
                    doc_vector = embeddings.embed_documents([r["doc"].page_content])[0]
                    r["score"] = _cosine_similarity(query_vector, doc_vector)
                    r["reason"] = f"cosine similarity to '{fuzzy_query}': {r['score']:.4f}"
        else:
            # FULL PATH - matches search_listings' HyDE + cross-encoder branch
            hyde_text = _generate_hyde_text(fuzzy_query)
            pairs = [[hyde_text, r["doc"].page_content] for r in results if r["tier"] != "hard_filter_rejected"]
            cross_scores = _cross_encoder.predict(pairs) if pairs else []

            score_idx = 0
            for r in results:
                if r["tier"] != "hard_filter_rejected":
                    r["score"] = float(cross_scores[score_idx])
                    r["reason"] = f"cross-encoder score (HyDE query: '{hyde_text[:60]}...'): {r['score']:.4f}"
                    score_idx += 1

    selected_results = search_listings(buyer_preferences, vectorstore, embeddings, k=3)
    selected_ids = [d.metadata.get("listing_id") for d in selected_results]

    for r in results:
        if r["listing_id"] in selected_ids:
            r["tier"] = "matched"
            if r["score"] is not None:
                r["reason"] = f"selected — {r['reason'].split(': ', 1)[0]}: {r['score']:.4f}"
            else:
                r["reason"] = "selected in final shortlist"

    return results