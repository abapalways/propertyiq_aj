"""
Core search logic: takes buyer preferences + a vector store, 
returns a ranked shortlist of matching listings.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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


def _generate_hyde_text(query):
    prompt = f"""Write exactly 1-2 sentences describing a home that matches 
this buyer request: "{query}". Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
    return _llm.invoke(prompt).content


def _cosine_similarity(vec_a, vec_b):
    import numpy as np
    vec_a, vec_b = np.array(vec_a), np.array(vec_b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


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

    fuzzy_must_haves = []
    for mh in must_haves:
        match = _matches_checkable(mh)
        if match:
            key, check_fn = match
            candidates = [doc for doc in candidates if check_fn(doc.metadata.get(key))]
        else:
            fuzzy_must_haves.append(mh)

    if not candidates:
        return []

    if not fuzzy_must_haves:
        return candidates[:k]

    fuzzy_query = " ".join(fuzzy_must_haves)
    hyde_text = _generate_hyde_text(fuzzy_query)

    hyde_vector = embeddings.embed_query(hyde_text)
    cosine_scored = []
    for doc in candidates:
        doc_vector = embeddings.embed_query(doc.page_content)
        sim = _cosine_similarity(hyde_vector, doc_vector)
        cosine_scored.append((sim, doc))
    cosine_scored.sort(key=lambda x: x[0], reverse=True)
    narrowed = [doc for sim, doc in cosine_scored]

    pairs = [[hyde_text, doc.page_content] for doc in narrowed]
    cross_scores = _cross_encoder.predict(pairs)
    reranked = sorted(zip(cross_scores, narrowed), key=lambda x: x[0], reverse=True)

    return [doc for score, doc in reranked[:k]]