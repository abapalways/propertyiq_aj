import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

all_docs = list(vectorstore.docstore._dict.values())
target_ids = ["L_PINE_101", "L_ELM_124", "L_MAPLE_105", "L_SPRUCE_106"]
candidates = [d for d in all_docs if d.metadata.get("listing_id") in target_ids]


def extract_school_sentence(doc_text):
    match = re.search(r"^.*[Ss]chool.*$", doc_text, re.MULTILINE)
    return match.group(0).strip() if match else doc_text[:300]


query = "good school district"

# Test 1: full document (what we had before)
print("=== Test 1: full document text ===")
pairs_full = [[query, doc.page_content] for doc in candidates]
scores_full = cross_encoder.predict(pairs_full)
ranked_full = sorted(zip(scores_full, candidates), key=lambda x: x[0], reverse=True)
for rank, (score, doc) in enumerate(ranked_full, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — score={score:.4f}")

# Test 2: just the school sentence
print("\n=== Test 2: extracted school sentence only ===")
pairs_targeted = [[query, extract_school_sentence(doc.page_content)] for doc in candidates]
for q, d in pairs_targeted:
    print(f"  Feeding: {d!r}")
scores_targeted = cross_encoder.predict(pairs_targeted)
ranked_targeted = sorted(zip(scores_targeted, candidates), key=lambda x: x[0], reverse=True)
print()
for rank, (score, doc) in enumerate(ranked_targeted, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — score={score:.4f}")

    from langchain_groq import ChatGroq

_enrich_llm = ChatGroq(model="openai/gpt-oss-120b")


def enrich_with_sentiment(school_text):
    prompt = f"""Read this real estate listing's school district description and 
summarize its quality in ONE short sentence using clear positive or negative 
language (e.g. "excellent, top-rated" or "poor, underperforming"). 
Be direct and unambiguous about whether it's good or bad.

Text: {school_text}

One-sentence summary:"""
    return _enrich_llm.invoke(prompt).content.strip()


print("\n=== Test 3: school sentence + LLM-generated sentiment adjective ===")
pairs_enriched = []
for doc in candidates:
    school_sentence = extract_school_sentence(doc.page_content)
    enrichment = enrich_with_sentiment(school_sentence)
    combined = f"{school_sentence} {enrichment}"
    pairs_enriched.append([query, combined])
    print(f"  {doc.metadata.get('listing_id')}: {combined!r}")

scores_enriched = cross_encoder.predict(pairs_enriched)
ranked_enriched = sorted(zip(scores_enriched, candidates), key=lambda x: x[0], reverse=True)
print()
for rank, (score, doc) in enumerate(ranked_enriched, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — score={score:.4f}")

    from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

print("\n=== Test 4: NLI cross-encoder (contradiction/entailment) ===")
hypothesis = "This home is in a good school district."

for doc in candidates:
    school_sentence = extract_school_sentence(doc.page_content)
    scores = nli_model.predict([(school_sentence, hypothesis)])
    # NLI models typically output [contradiction, entailment, neutral] logits
    labels = ["contradiction", "entailment", "neutral"]
    predicted_label = labels[scores.argmax()]
    print(f"  {doc.metadata.get('listing_id')}: {predicted_label} (scores={scores})")
    print(f"    Premise: {school_sentence!r}")

    from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

print("\n=== Test 4: NLI cross-encoder (contradiction/entailment) ===")
hypothesis = "This home is in a good school district."

for doc in candidates:
    school_sentence = extract_school_sentence(doc.page_content)
    scores = nli_model.predict([(school_sentence, hypothesis)])
    # NLI models typically output [contradiction, entailment, neutral] logits
    labels = ["contradiction", "entailment", "neutral"]
    predicted_label = labels[scores.argmax()]
    print(f"  {doc.metadata.get('listing_id')}: {predicted_label} (scores={scores})")
    print(f"    Premise: {school_sentence!r}")

print("\n=== Test 5: NLI cross-encoder with enriched (sentiment-adjective) text ===")
hypothesis = "This home is in a good school district."

for doc in candidates:
    school_sentence = extract_school_sentence(doc.page_content)
    enrichment = enrich_with_sentiment(school_sentence)
    combined = f"{school_sentence} {enrichment}"

    scores = nli_model.predict([(combined, hypothesis)])
    labels = ["contradiction", "entailment", "neutral"]
    predicted_label = labels[scores.argmax()]
    print(f"  {doc.metadata.get('listing_id')}: {predicted_label} (scores={scores})")
    print(f"    Premise: {combined!r}")