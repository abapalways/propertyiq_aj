"""
Core search logic: takes buyer preferences + a vector store, 
returns a ranked shortlist of matching listings.

Pipeline:
1. Exact metadata filters (price, bedrooms, garage, yard, city)
2. Fuzzy criteria classified to a known FUZZY_DIMENSIONS entry via
   cosine similarity against canonical_phrases (deterministic, no LLM)
3. RRF (Reciprocal Rank Fusion) across fuzzy criteria for ranking
4. NLI contradiction-check using PRE-ENRICHED dimension text from
   ingest.py's dimension_enrichments metadata - hard-excludes listings
   whose enriched text contradicts a buyer's stated criterion
"""
import math
from collections import Counter

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
import numpy as np
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

from dimensions_config import FUZZY_DIMENSIONS

_nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
_NLI_LABELS = ["contradiction", "entailment", "neutral"]

CHECKABLE_ATTRIBUTES = {
    "garage": ("garage_spaces", lambda v: v is not None and v >= 1),
    "private yard": ("has_private_yard", lambda v: v is True),
    "yard": ("has_private_yard", lambda v: v is True),
}

# Cache for dimension canonical-phrase embeddings, built once per process
_dimension_embeddings_cache = None


def _tokenize(text):
    return re.findall(r"\w+", text.lower())


class _SimpleBM25:
    """Minimal BM25 (Okapi) implementation - no external dependency."""

    def __init__(self, tokenized_corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = tokenized_corpus
        self.doc_lens = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens) if self.doc_lens else 0
        self.doc_freqs = [Counter(doc) for doc in tokenized_corpus]
        self.n_docs = len(tokenized_corpus)
        self.idf = self._compute_idf()

    def _compute_idf(self):
        df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
        return idf

    def get_scores(self, tokenized_query):
        scores = []
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]
            for term in tokenized_query:
                if term not in doc_freqs:
                    continue
                freq = doc_freqs[term]
                idf = self.idf.get(term, 0.0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores


def _matches_checkable(must_have_text):
    text_lower = must_have_text.lower()
    for phrase, (key, check_fn) in CHECKABLE_ATTRIBUTES.items():
        if phrase in text_lower:
            return key, check_fn
    return None


def _build_bm25_index(candidates):
    """Build a BM25 index over candidate listings' page_content.
    Built fresh per search call since candidates vary with rejected_listing_ids."""
    tokenized_corpus = [_tokenize(doc.page_content) for doc in candidates]
    return _SimpleBM25(tokenized_corpus)


def _cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --- Cosine-similarity fuzzy dimension classification (deterministic, no LLM) ---

def build_dimension_embeddings(fuzzy_dimensions, embeddings_model):
    """
    Embeds each dimension's canonical_phrases once.
    Returns: {dimension_name: [(phrase, embedding_vector), ...]}
    """
    dimension_embeddings = {}
    for dim_name, config in fuzzy_dimensions.items():
        phrases = config["canonical_phrases"]
        vectors = embeddings_model.embed_documents(phrases)
        dimension_embeddings[dim_name] = list(zip(phrases, vectors))
    return dimension_embeddings


def _get_dimension_embeddings(embeddings_model):
    global _dimension_embeddings_cache
    if _dimension_embeddings_cache is None:
        _dimension_embeddings_cache = build_dimension_embeddings(FUZZY_DIMENSIONS, embeddings_model)
    return _dimension_embeddings_cache


def classify_fuzzy_dimension(criterion_text, dimension_embeddings, embeddings_model, threshold=0.75):
    """
    Maps a buyer's stated fuzzy criterion (e.g. "move-in ready") to the
    closest FUZZY_DIMENSIONS entry via cosine similarity - no LLM call.

    Returns (dimension_name, best_phrase, score) if confident,
    else (None, None, best_score_seen) so caller can fall back to plain RRF.
    """
    query_vec = embeddings_model.embed_query(criterion_text)

    best_dim = None
    best_phrase = None
    best_score = -1.0

    for dim_name, phrase_vectors in dimension_embeddings.items():
        for phrase, vec in phrase_vectors:
            score = _cosine_similarity(query_vec, vec)
            if score > best_score:
                best_score = score
                best_dim = dim_name
                best_phrase = phrase

    if best_score >= threshold:
        return best_dim, best_phrase, best_score
    return None, None, best_score


# --- NLI contradiction check using pre-enriched dimension_enrichments metadata ---

def _check_contradiction(doc, criterion, dimension_embeddings, embeddings_model, threshold=0.75):
    """
    Checks whether a listing's PRE-ENRICHED dimension text (from ingest.py)
    contradicts a buyer's fuzzy criterion. Uses cosine classification to pick
    the dimension, then looks up doc.metadata["dimension_enrichments"] -
    no live LLM extraction/enrichment at query time.

    Returns (is_contradiction: bool, detail: dict) - detail carries full
    reasoning trail (classification score, NLI scores, text used) for
    display in the Debug panel, not just a terminal print.
    """
    listing_id = doc.metadata.get("listing_id")
    dim_name, matched_phrase, class_score = classify_fuzzy_dimension(
        criterion, dimension_embeddings, embeddings_model, threshold=threshold
    )
    print(f"LOG [classify] '{criterion}' -> dimension={dim_name}, best_phrase='{matched_phrase}', score={class_score:.3f}, threshold={threshold}")

    detail = {
        "criterion": criterion,
        "classified_dimension": dim_name,
        "matched_canonical_phrase": matched_phrase,
        "classification_score": round(float(class_score), 3),
        "classification_threshold": threshold,
    }

    if dim_name is None:
        print(f"LOG [contradiction-check] {listing_id}: SKIPPED (no confident dimension match for '{criterion}')")
        detail["outcome"] = "skipped_no_dimension_match"
        return False, detail

    dimension_data = doc.metadata.get("dimension_enrichments", {}).get(dim_name)
    if not dimension_data or not dimension_data.get("raw_text"):
        print(f"LOG [contradiction-check] {listing_id}: SKIPPED (no '{dim_name}' enrichment on this listing)")
        detail["outcome"] = "skipped_no_enrichment"
        return False, detail

    enrichment_text = dimension_data["enrichment"]
    raw_text = dimension_data["raw_text"]
    combined = f"{raw_text} {enrichment_text}"
    hypothesis = f"This home {criterion}."
    scores = _nli_model.predict([(combined, hypothesis)])
    label = _NLI_LABELS[scores.argmax()]
    print(f"LOG [NLI] {listing_id} dim={dim_name}: text='{combined[:80]}...' hypothesis='{hypothesis}' -> {label} (scores={scores[0].round(3)})")

    detail.update({
        "outcome": "checked",
        "nli_label": label,
        "nli_scores": {lbl: round(float(s), 3) for lbl, s in zip(_NLI_LABELS, scores[0])},
        "raw_text_used": raw_text,
        "enrichment_used": enrichment_text,
    })
    return label == "contradiction", detail


_last_full_analysis = []  # module-level, built as a byproduct of the ONE real search_listings() run

def search_listings(buyer_preferences, vectorstore, embeddings, k=3, rejected_listing_ids=None):
    global _last_full_analysis
    if rejected_listing_ids is None:
        rejected_listing_ids = []

    dimension_embeddings = _get_dimension_embeddings(embeddings)

    all_docs = list(vectorstore.docstore._dict.values())
    all_docs = [doc for doc in all_docs if doc.metadata.get("listing_id") not in rejected_listing_ids]

    max_budget = buyer_preferences["max_budget"]
    min_bedrooms = buyer_preferences["min_bedrooms"]
    preferred_city = buyer_preferences.get("preferred_city")
    must_haves = buyer_preferences.get("must_haves", [])

    breakdown = []  # this becomes _last_full_analysis
    candidates = []

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
            breakdown.append({"listing_id": listing_id, "tier": "hard_filter_rejected", "reason": "; ".join(reasons_failed), "doc": doc})
        else:
            candidates.append(doc)
            breakdown.append({"listing_id": listing_id, "tier": "passed_but_not_selected", "reason": "passed all hard filters", "doc": doc})

    if not candidates:
        _last_full_analysis = breakdown
        return []

    fuzzy_must_haves = [mh for mh in must_haves if _matches_checkable(mh) is None]
    print(f"LOG [stage1] must_haves={must_haves} -> checkable(hard-filtered)={[mh for mh in must_haves if _matches_checkable(mh)]}, fuzzy={fuzzy_must_haves}")

    if not fuzzy_must_haves:
        matched_ids = {doc.metadata.get("listing_id") for doc in candidates[:k]}
        for r in breakdown:
            if r["listing_id"] in matched_ids:
                r["tier"] = "matched"
                r["reason"] = "selected (no fuzzy criteria to rank by)"
        _last_full_analysis = breakdown
        return candidates[:k]

    # Split fuzzy must_haves: confidently-classified dimension criteria (dense + NLI)
    # vs unclassified criteria (keyword/named-entity - use BM25 instead)
    dimension_criteria = []
    keyword_criteria = []
    for criterion in fuzzy_must_haves:
        dim_name, _, _ = classify_fuzzy_dimension(criterion, dimension_embeddings, embeddings)
        if dim_name is not None:
            dimension_criteria.append(criterion)
        else:
            keyword_criteria.append(criterion)
    print(f"LOG [stage2] dimension_criteria={dimension_criteria}, keyword_criteria={keyword_criteria}")

    for r in breakdown:
        r["dimension_criteria"] = dimension_criteria
        r["keyword_criteria"] = keyword_criteria

    RRF_K = 60
    rrf_scores = {doc.metadata.get("listing_id"): 0.0 for doc in candidates}

    # Dense RRF for dimension-classified criteria
    for criterion in dimension_criteria:
        query_vector = embeddings.embed_query(criterion)
        scored = []
        for doc in candidates:
            doc_vector = embeddings.embed_documents([doc.page_content])[0]
            sim = _cosine_similarity(query_vector, doc_vector)
            scored.append((sim, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        for rank, (sim, doc) in enumerate(scored, 1):
            rrf_scores[doc.metadata.get("listing_id")] += 1.0 / (RRF_K + rank)

    # BM25 RRF for keyword/named-entity criteria
    if keyword_criteria:
        bm25 = _build_bm25_index(candidates)
        for criterion in keyword_criteria:
            tokenized_query = _tokenize(criterion)
            bm25_scores = bm25.get_scores(tokenized_query)
            scored = list(zip(bm25_scores, candidates))
            scored.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, doc) in enumerate(scored, 1):
                rrf_scores[doc.metadata.get("listing_id")] += 1.0 / (RRF_K + rank)
            print(f"DEBUG BM25 '{criterion}': " + ", ".join(
                f"{doc.metadata.get('listing_id')}={s:.3f}" for s, doc in scored
            ))

    doc_by_id = {doc.metadata.get("listing_id"): doc for doc in candidates}
    rrf_ranked_ids = sorted(rrf_scores.keys(), key=lambda lid: rrf_scores[lid], reverse=True)
    rrf_ranked_docs = [doc_by_id[lid] for lid in rrf_ranked_ids]
    print(f"LOG [stage3] RRF scores after all criteria: {rrf_scores}")

    # NLI contradiction check: exclude listings whose pre-enriched dimension
    # text contradicts any fuzzy must-have. Each doc is matched back to its
    # OWN breakdown entry by listing_id - never by tier - so a contradiction
    # on one listing can't leak onto another's reason.
    non_contradicting_docs = []
    for doc in rrf_ranked_docs:
        listing_id = doc.metadata.get("listing_id")
        contradicts_any = False
        contradiction_detail = None
        all_checks = []

        for criterion in dimension_criteria:
            is_contradiction, detail = _check_contradiction(doc, criterion, dimension_embeddings, embeddings)
            all_checks.append(detail)
            print(f"DEBUG: {listing_id} vs '{criterion}': is_contradiction={is_contradiction}, label={detail}")

            if is_contradiction:
                contradicts_any = True
                contradiction_detail = detail
                break

        if contradicts_any:
            for r in breakdown:
                if r["listing_id"] == listing_id:
                    r["tier"] = "hard_filter_rejected"
                    r["reason"] = f"contradicts '{contradiction_detail['criterion']}' (NLI: {contradiction_detail['nli_label']})"
                    r["nli_checks"] = all_checks
        else:
            non_contradicting_docs.append(doc)
            for r in breakdown:
                if r["listing_id"] == listing_id:
                    r["nli_checks"] = all_checks

    final_docs = non_contradicting_docs[:k]
    final_ids = {doc.metadata.get("listing_id") for doc in final_docs}

    for r in breakdown:
        if r["tier"] == "passed_but_not_selected":
            r["_rrf_score"] = rrf_scores.get(r["listing_id"], 0)
            r["reason"] = f"RRF score: {rrf_scores.get(r['listing_id'], 0):.5f}"
            if r["listing_id"] in final_ids:
                r["tier"] = "matched"
                r["reason"] = f"selected — {r['reason']}"

    _last_full_analysis = breakdown
    return final_docs