# %%
# --- Fix for FAISS/PyTorch OpenMP crash ---
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# %%
# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv("../.env")

# %%
# --- Import reusable search function ---
from search import search_listings

# %%
# --- Set up embeddings + load the saved FAISS index (nomic-embed-text) ---
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = FAISS.load_local(
    "../faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Loaded index with {vectorstore.index.ntotal} vectors")

# %%
# --- Rebuild `documents` list (needed for cross-model comparisons) ---
import re
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

with open("../data/listings_corpus.md") as f:
    text = f.read()

splitter = CharacterTextSplitter(separator="---", chunk_size=1000, chunk_overlap=0)
raw_chunks = splitter.split_text(text)

documents = []
for chunk in raw_chunks:
    listing_id_match = re.search(r"Property ID:\s*(\S+)", chunk)
    price_match = re.search(r"\*\*Price:\*\*\s*\$([\d,]+)", chunk)
    bed_match = re.search(r"\*\*Bedrooms:\*\*\s*(\d+)", chunk)
    metadata = {
        "listing_id": listing_id_match.group(1) if listing_id_match else None,
        "price": int(price_match.group(1).replace(",", "")) if price_match else None,
        "bedrooms": int(bed_match.group(1)) if bed_match else None,
    }
    documents.append(Document(page_content=chunk.strip(), metadata=metadata))

print(f"Rebuilt {len(documents)} documents")

# %%
# --- Helper functions used throughout ---
import numpy as np

def cosine_similarity(vec_a, vec_b):
    vec_a, vec_b = np.array(vec_a), np.array(vec_b)
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def get_school_line(doc_text):
    match = re.search(r"^.*[Ss]chool.*$", doc_text, re.MULTILINE)
    return match.group(0).strip() if match else "(no school mention found)"

# %%
# --- Approach A: pure semantic search, no metadata filtering ---
query = "3-bed homes under $450K near good schools"
results = vectorstore.similarity_search(query, k=3)

print("Approach A (no filtering) - top 3:\n")
for i, doc in enumerate(results, 1):
    print(f"#{i} {doc.metadata.get('listing_id')} — ${doc.metadata.get('price'):,} — {doc.metadata.get('bedrooms')} bed")

# %%
# --- Filter by hard constraints (price <= 450K, bedrooms == 3) ---
all_docs = list(vectorstore.docstore._dict.values())

candidates = [
    doc for doc in all_docs
    if doc.metadata.get("price") <= 450000
    and doc.metadata.get("bedrooms") == 3
]

print(f"{len(candidates)} candidates survived the filter:")
for doc in candidates:
    print(doc.metadata.get("listing_id"), doc.metadata.get("price"), doc.metadata.get("bedrooms"))

# %%
# --- Cosine similarity ranking on filtered candidates (query: "good schools") ---
query_vector = embeddings.embed_query("good schools")

scored = []
for doc in candidates:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(query_vector, doc_vector)
    scored.append((sim, doc))

scored.sort(key=lambda x: x[0], reverse=True)

print("nomic-embed-text ranking, query: 'good schools'\n")
for rank, (sim, doc) in enumerate(scored, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")
    print(f"    {get_school_line(doc.page_content)}")

# %%
# --- Full ranking, ALL 10 listings, full original query (no filtering) ---
full_query = "3-bed homes under $450K near good schools"
full_query_vector = embeddings.embed_query(full_query)

scored_all = []
for doc in all_docs:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(full_query_vector, doc_vector)
    scored_all.append((sim, doc))

scored_all.sort(key=lambda x: x[0], reverse=True)

print(f"Full ranking, all 10 listings, query: '{full_query}'\n")
for rank, (sim, doc) in enumerate(scored_all, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")

# %%
# --- Cross-encoder re-ranking (cross-encoder/ms-marco-MiniLM-L-6-v2) ---
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query_for_rerank = "3-bed home near good schools with a garage"
pairs = [[query_for_rerank, doc.page_content] for doc in candidates]
cross_scores = cross_encoder.predict(pairs)

reranked = sorted(zip(cross_scores, candidates), key=lambda x: x[0], reverse=True)

print(f"Cross-encoder re-ranking, query: '{query_for_rerank}'\n")
for rank, (score, doc) in enumerate(reranked, 1):
    listing_id = doc.metadata.get("listing_id")
    price = doc.metadata.get("price")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — cross_score={score:.4f} — ${price:,}{marker}")

# %%
# --- BGE-M3 comparison: rebuild index with a different embedding model ---
embeddings_bge = OllamaEmbeddings(model="bge-m3")

bge_vectorstore = FAISS.from_documents(documents, embeddings_bge)
print(f"BGE-M3 index built with {bge_vectorstore.index.ntotal} vectors")

# %%
all_docs_bge = list(bge_vectorstore.docstore._dict.values())

candidates_bge = [
    doc for doc in all_docs_bge
    if doc.metadata.get("price") <= 450000
    and doc.metadata.get("bedrooms") == 3
]

query_vector_bge = embeddings_bge.embed_query("good schools")

scored_bge = []
for doc in candidates_bge:
    doc_vector = embeddings_bge.embed_query(doc.page_content)
    sim = cosine_similarity(query_vector_bge, doc_vector)
    scored_bge.append((sim, doc))

scored_bge.sort(key=lambda x: x[0], reverse=True)

print("BGE-M3 ranking, query: 'good schools'\n")
for rank, (sim, doc) in enumerate(scored_bge, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")

# %%
# --- Test buyer profile → search_listings() ---
import json

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)

torres = buyer_profiles[0]
results = search_listings(torres["preferences"], vectorstore, embeddings, k=3)

print(f"Shortlist for {torres['name']}:\n")
for i, doc in enumerate(results, 1):
    print(f"#{i} {doc.metadata.get('listing_id')} — ${doc.metadata.get('price'):,} — {doc.metadata.get('bedrooms')} bed")
# %%
# Test different threshold values against the existing "good schools" ranking
print("Testing thresholds against 6 filtered candidates:\n")

for threshold in [0.40, 0.42, 0.43, 0.44, 0.45]:
    passing = [(sim, doc) for sim, doc in scored if sim >= threshold]
    passing_ids = [doc.metadata.get("listing_id") for sim, doc in passing]
    pine_included = "L_PINE_101" in passing_ids
    print(f"threshold={threshold} — {len(passing)} pass — {passing_ids} — L_PINE_101 included: {pine_included}")

# %%

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen3:8b")

def generate_hypothetical_document(query):
    prompt = f"""Write a short, realistic real-estate listing description 
(2-3 sentences) that would perfectly satisfy this buyer request: "{query}"
Focus on the specific attribute being asked about. Do not include a price or address."""
    response = llm.invoke(prompt)
    return response.content

hyde_text = generate_hypothetical_document("good schools")
print(hyde_text)

# %%
hyde_vector = embeddings.embed_query(hyde_text)

scored_hyde = []
for doc in candidates:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(hyde_vector, doc_vector)
    scored_hyde.append((sim, doc))

scored_hyde.sort(key=lambda x: x[0], reverse=True)

print("HyDE ranking (nomic-embed-text), original query: 'good schools'\n")
for rank, (sim, doc) in enumerate(scored_hyde, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")
    print(f"    {get_school_line(doc.page_content)}")


# %%
debug_hyde_text = """Nestled in a top-tier school district renowned for its academic excellence and robust extracurricular programs, this home offers proximity to award-winning schools with strong college acceptance rates and personalized learning opportunities. The district's commitment to safety, advanced STEM initiatives, and community engagement ensures a supportive environment for students of all ages."""

debug_query_vector = embeddings.embed_query(debug_hyde_text)

debug_candidates = [doc for doc in candidates if doc.metadata.get("listing_id") in 
                     ["L_PINE_101", "L_ELM_124", "L_MAPLE_105", "L_SPRUCE_106", "L_WALNUT_107"]]

scored_debug = []
for doc in debug_candidates:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(debug_query_vector, doc_vector)
    scored_debug.append((sim, doc))

scored_debug.sort(key=lambda x: x[0], reverse=True)

print("Debug ranking using the ACTUAL HyDE text from your app run:\n")
for rank, (sim, doc) in enumerate(scored_debug, 1):
    listing_id = doc.metadata.get("listing_id")
    print(f"#{rank} {listing_id} — sim={sim:.4f}")
    print(f"    {get_school_line(doc.page_content)}")
    print(f"    full text preview: {doc.page_content[:300]}")
    print()


    # %%
# %%
candidates_matching_app = [
    doc for doc in all_docs
    if doc.metadata.get("price") <= 450000
    and doc.metadata.get("bedrooms") >= 3
    and doc.metadata.get("garage_spaces") is not None
    and doc.metadata.get("garage_spaces") >= 1
]

print(f"Candidates matching app's real logic: {[d.metadata.get('listing_id') for d in candidates_matching_app]}")

debug_query_vector = embeddings.embed_query("""Nestled in a top-tier school district renowned for its academic excellence and robust extracurricular programs, this home offers proximity to award-winning schools with strong college acceptance rates and personalized learning opportunities. The district's commitment to safety, advanced STEM initiatives, and community engagement ensures a supportive environment for students of all ages.""")

scored_correct = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(debug_query_vector, doc_vector)
    scored_correct.append((sim, doc))

scored_correct.sort(key=lambda x: x[0], reverse=True)

print("\nCorrected ranking (matching real app candidates):\n")
for rank, (sim, doc) in enumerate(scored_correct, 1):
    print(f"#{rank} {doc.metadata.get('listing_id')} — sim={sim:.4f}")
# %%
# %%
bad_prompt = """Write a short, realistic real-estate listing description 
(2-3 sentences) describing a home zoned for a poorly-rated, underperforming 
school district. Do not include a price or address."""
bad_hyde_text = llm.invoke(bad_prompt).content
print(bad_hyde_text)

bad_query_vector = embeddings.embed_query(bad_hyde_text)

scored_contrastive = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    good_sim = cosine_similarity(debug_query_vector, doc_vector)
    bad_sim = cosine_similarity(bad_query_vector, doc_vector)
    delta = good_sim - bad_sim
    scored_contrastive.append((delta, good_sim, bad_sim, doc))

scored_contrastive.sort(key=lambda x: x[0], reverse=True)

print("\nContrastive ranking (good_sim - bad_sim):\n")
for rank, (delta, good_sim, bad_sim, doc) in enumerate(scored_contrastive, 1):
    print(f"#{rank} {doc.metadata.get('listing_id')} — delta={delta:.4f} (good={good_sim:.4f}, bad={bad_sim:.4f})")
# %%
bad_prompt = """Write a short, realistic real-estate listing description 
(2-3 sentences) that would describe a home in a poorly-rated school district 
(rated below 5 out of 10). Use the same tone and structure as a real estate 
listing — describe the home and neighborhood naturally, the way an agent 
would write it, mentioning the low school rating as one detail among others. 
Do not include a price or address. Do not include headers, bullet points, 
percentages, or any explanation of a rating system — write ONLY the listing 
description itself."""
bad_hyde_text = llm.invoke(bad_prompt).content
print(bad_hyde_text)

bad_query_vector = embeddings.embed_query(bad_hyde_text)

scored_contrastive = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    good_sim = cosine_similarity(debug_query_vector, doc_vector)
    bad_sim = cosine_similarity(bad_query_vector, doc_vector)
    delta = good_sim - bad_sim
    scored_contrastive.append((delta, good_sim, bad_sim, doc))

scored_contrastive.sort(key=lambda x: x[0], reverse=True)

print("\nContrastive ranking (good_sim - bad_sim):\n")
for rank, (delta, good_sim, bad_sim, doc) in enumerate(scored_contrastive, 1):
    print(f"#{rank} {doc.metadata.get('listing_id')} — delta={delta:.4f} (good={good_sim:.4f}, bad={bad_sim:.4f})")
# %%
# %%
query_for_rerank = "home in a highly-rated, good school district"

pairs = [[query_for_rerank, doc.page_content] for doc in candidates_matching_app]
cross_scores = cross_encoder.predict(pairs)

reranked_school = sorted(zip(cross_scores, candidates_matching_app), key=lambda x: x[0], reverse=True)

print(f"Cross-encoder ranking, query: '{query_for_rerank}'\n")
for rank, (score, doc) in enumerate(reranked_school, 1):
    listing_id = doc.metadata.get("listing_id")
    school_line = get_school_line(doc.page_content)
    print(f"#{rank} {listing_id} — cross_score={score:.4f}")
    print(f"    {school_line}")
# %%
# %%
query_short = "good school district"
pairs_short = [[query_short, doc.page_content] for doc in candidates_matching_app]
scores_short = cross_encoder.predict(pairs_short)

reranked_short = sorted(zip(scores_short, candidates_matching_app), key=lambda x: x[0], reverse=True)

print(f"Cross-encoder ranking, SHORT query: '{query_short}'\n")
for rank, (score, doc) in enumerate(reranked_short, 1):
    print(f"#{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
# %%
# %%
query_short = "good school district"
pairs_short = [[query_short, doc.page_content] for doc in candidates_matching_app]
scores_short = cross_encoder.predict(pairs_short)

reranked_short = sorted(zip(scores_short, candidates_matching_app), key=lambda x: x[0], reverse=True)

print(f"Cross-encoder ranking, SHORT query: '{query_short}'\n")
for rank, (score, doc) in enumerate(reranked_short, 1):
    print(f"#{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
# %%
# %%
# --- VARIANT 2: HyDE -> cosine -> cross-encoder (all stages use HyDE text) ---
hyde_prompt_v2 = """Write exactly 1-2 sentences describing a home in a highly-rated, 
excellent school district. Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
hyde_text_v2 = llm.invoke(hyde_prompt_v2).content
print(f"Variant 2 HyDE text: {hyde_text_v2}\n")

hyde_vector_v2 = embeddings.embed_query(hyde_text_v2)
v2_cosine_scored = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(hyde_vector_v2, doc_vector)
    v2_cosine_scored.append((sim, doc))
v2_cosine_scored.sort(key=lambda x: x[0], reverse=True)
v2_candidates = [doc for sim, doc in v2_cosine_scored]

pairs_v2 = [[hyde_text_v2, doc.page_content] for doc in v2_candidates]
cross_scores_v2 = cross_encoder.predict(pairs_v2)
v2_final = sorted(zip(cross_scores_v2, v2_candidates), key=lambda x: x[0], reverse=True)

print("Variant 2 final ranking (HyDE -> cosine -> cross-encoder):")
for rank, (score, doc) in enumerate(v2_final, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
    print(f"      {get_school_line(doc.page_content)}")

# %%
# --- VARIANT 3: plain-query cosine -> HyDE -> cross-encoder ---
plain_query = "good schools"
plain_query_vector = embeddings.embed_query(plain_query)
v3_cosine_scored = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(plain_query_vector, doc_vector)
    v3_cosine_scored.append((sim, doc))
v3_cosine_scored.sort(key=lambda x: x[0], reverse=True)
v3_candidates = [doc for sim, doc in v3_cosine_scored]

hyde_prompt_v3 = """Write exactly 1-2 sentences describing a home in a highly-rated, 
excellent school district. Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
hyde_text_v3 = llm.invoke(hyde_prompt_v3).content
print(f"Variant 3 HyDE text: {hyde_text_v3}\n")

pairs_v3 = [[hyde_text_v3, doc.page_content] for doc in v3_candidates]
cross_scores_v3 = cross_encoder.predict(pairs_v3)
v3_final = sorted(zip(cross_scores_v3, v3_candidates), key=lambda x: x[0], reverse=True)

print("Variant 3 final ranking (plain cosine -> HyDE -> cross-encoder):")
for rank, (score, doc) in enumerate(v3_final, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
    print(f"      {get_school_line(doc.page_content)}")

    # %%
# --- VARIANT 2: HyDE -> cosine -> cross-encoder (all stages use HyDE text) ---
hyde_prompt_v2 = """Write exactly 1-2 sentences describing a home in a highly-rated, 
excellent school district. Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
hyde_text_v2 = llm.invoke(hyde_prompt_v2).content
print(f"Variant 2 HyDE text: {hyde_text_v2}\n")

hyde_vector_v2 = embeddings.embed_query(hyde_text_v2)
v2_cosine_scored = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(hyde_vector_v2, doc_vector)
    v2_cosine_scored.append((sim, doc))
v2_cosine_scored.sort(key=lambda x: x[0], reverse=True)
v2_candidates = [doc for sim, doc in v2_cosine_scored]

pairs_v2 = [[hyde_text_v2, doc.page_content] for doc in v2_candidates]
cross_scores_v2 = cross_encoder.predict(pairs_v2)
v2_final = sorted(zip(cross_scores_v2, v2_candidates), key=lambda x: x[0], reverse=True)

print("Variant 2 final ranking (HyDE -> cosine -> cross-encoder):")
for rank, (score, doc) in enumerate(v2_final, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
    print(f"      {get_school_line(doc.page_content)}")

# %%
# --- VARIANT 3: plain-query cosine -> HyDE -> cross-encoder ---
plain_query = "good schools"
plain_query_vector = embeddings.embed_query(plain_query)
v3_cosine_scored = []
for doc in candidates_matching_app:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(plain_query_vector, doc_vector)
    v3_cosine_scored.append((sim, doc))
v3_cosine_scored.sort(key=lambda x: x[0], reverse=True)
v3_candidates = [doc for sim, doc in v3_cosine_scored]

hyde_prompt_v3 = """Write exactly 1-2 sentences describing a home in a highly-rated, 
excellent school district. Write it in natural real-estate listing style. 
No headers, no bullet points, no explanations."""
hyde_text_v3 = llm.invoke(hyde_prompt_v3).content
print(f"Variant 3 HyDE text: {hyde_text_v3}\n")

pairs_v3 = [[hyde_text_v3, doc.page_content] for doc in v3_candidates]
cross_scores_v3 = cross_encoder.predict(pairs_v3)
v3_final = sorted(zip(cross_scores_v3, v3_candidates), key=lambda x: x[0], reverse=True)

print("Variant 3 final ranking (plain cosine -> HyDE -> cross-encoder):")
for rank, (score, doc) in enumerate(v3_final, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id')} — cross_score={score:.4f}")
    print(f"      {get_school_line(doc.page_content)}")
    # %%
# --- Diagnostic: full pipeline trace for Torres, step by step ---
import importlib
import search
importlib.reload(search)  # pick up latest search.py changes

torres = buyer_profiles[0]
must_haves = torres["preferences"]["must_haves"]
max_budget = torres["preferences"]["max_budget"]
min_bedrooms = torres["preferences"]["min_bedrooms"]

# Step 1: hard filters (price, bedrooms, checkable attributes)
all_docs_diag = list(vectorstore.docstore._dict.values())
candidates_diag = [
    doc for doc in all_docs_diag
    if doc.metadata.get("price") <= max_budget
    and doc.metadata.get("bedrooms") >= min_bedrooms
]

fuzzy_must_haves_diag = []
for mh in must_haves:
    match = search._matches_checkable(mh)
    if match:
        key, check_fn = match
        candidates_diag = [doc for doc in candidates_diag if check_fn(doc.metadata.get(key))]
    else:
        fuzzy_must_haves_diag.append(mh)

print(f"Fuzzy must-haves (sent to HyDE): {fuzzy_must_haves_diag}")
print(f"Candidates after hard filters: {[d.metadata.get('listing_id') for d in candidates_diag]}\n")

# Step 2: generate HyDE text (same call the real app makes)
fuzzy_query_diag = " ".join(fuzzy_must_haves_diag)
hyde_text_diag = search._generate_hyde_text(fuzzy_query_diag)
print(f"HyDE text generated:\n{hyde_text_diag}\n")

# Step 3: cosine similarity, per candidate, against the HyDE text
hyde_vector_diag = embeddings.embed_query(hyde_text_diag)

print("--- Cosine similarity (bi-encoder) per candidate ---")
cosine_results_diag = []
for doc in candidates_diag:
    doc_vector = embeddings.embed_query(doc.page_content)
    sim = cosine_similarity(hyde_vector_diag, doc_vector)
    cosine_results_diag.append((sim, doc))
    print(f"  {doc.metadata.get('listing_id'):15} cosine_sim={sim:.4f}  {get_school_line(doc.page_content)}")

cosine_results_diag.sort(key=lambda x: x[0], reverse=True)
narrowed_diag = [doc for sim, doc in cosine_results_diag]

# Step 4: cross-encoder, per candidate, against the same HyDE text
print("\n--- Cross-encoder score (final ranking) per candidate ---")
pairs_diag = [[hyde_text_diag, doc.page_content] for doc in narrowed_diag]
cross_scores_diag = cross_encoder.predict(pairs_diag)

final_diag = sorted(zip(cross_scores_diag, narrowed_diag), key=lambda x: x[0], reverse=True)
for rank, (score, doc) in enumerate(final_diag, 1):
    print(f"  #{rank} {doc.metadata.get('listing_id'):15} cross_score={score:.4f}  {get_school_line(doc.page_content)}")

    # %%
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings_gemini = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

test_vec = embeddings_gemini.embed_query("good schools")
print(f"Vector length: {len(test_vec)}")
# %%# %%
query_vector_gemini = embeddings_gemini.embed_query("good schools")

scored_gemini = []
for doc in candidates:
    doc_vector = embeddings_gemini.embed_query(doc.page_content)
    sim = cosine_similarity(query_vector_gemini, doc_vector)
    scored_gemini.append((sim, doc))

scored_gemini.sort(key=lambda x: x[0], reverse=True)

print("Gemini (gemini-embedding-001) ranking, query: 'good schools'\n")
for rank, (sim, doc) in enumerate(scored_gemini, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")

# %%
# %%
full_query_gemini = "3-bed homes under $450K near good schools"
full_query_vector_gemini = embeddings_gemini.embed_query(full_query_gemini)

scored_all_gemini = []
for doc in all_docs:
    doc_vector = embeddings_gemini.embed_query(doc.page_content)
    sim = cosine_similarity(full_query_vector_gemini, doc_vector)
    scored_all_gemini.append((sim, doc))

scored_all_gemini.sort(key=lambda x: x[0], reverse=True)

print(f"Gemini full ranking, all 10 listings, query: '{full_query_gemini}'\n")
for rank, (sim, doc) in enumerate(scored_all_gemini, 1):
    listing_id = doc.metadata.get("listing_id")
    price = doc.metadata.get("price")
    bedrooms = doc.metadata.get("bedrooms")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f} — ${price:,} — {bedrooms} bed{marker}")
# %%
# %%
garage_query_gemini = "3-bed home near good schools with a garage"
garage_query_vector_gemini = embeddings_gemini.embed_query(garage_query_gemini)

scored_garage_gemini = []
for doc in candidates:
    doc_vector = embeddings_gemini.embed_query(doc.page_content)
    sim = cosine_similarity(garage_query_vector_gemini, doc_vector)
    scored_garage_gemini.append((sim, doc))

scored_garage_gemini.sort(key=lambda x: x[0], reverse=True)

print(f"Gemini ranking, query: '{garage_query_gemini}'\n")
for rank, (sim, doc) in enumerate(scored_garage_gemini, 1):
    listing_id = doc.metadata.get("listing_id")
    garage = doc.metadata.get("garage_spaces")
    marker = "  <-- L_BIRCH_110 (NO GARAGE)" if listing_id == "L_BIRCH_110" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f} — garage_spaces={garage}{marker}")
# %%
# %%
# Gemini alone, on the REAL post-garage-filter candidate pool (Torres scenario)
query_schools_only = "good school district"
query_vector_real = embeddings_gemini.embed_query(query_schools_only)

scored_real = []
for doc in candidates_matching_app:  # already filtered: price, bedrooms>=3, garage>=1
    doc_vector = embeddings_gemini.embed_query(doc.page_content)
    sim = cosine_similarity(query_vector_real, doc_vector)
    scored_real.append((sim, doc))

scored_real.sort(key=lambda x: x[0], reverse=True)

print("Gemini alone, real Torres candidate pool (garage already filtered):\n")
for rank, (sim, doc) in enumerate(scored_real, 1):
    listing_id = doc.metadata.get("listing_id")
    marker = "  <-- L_PINE_101" if listing_id == "L_PINE_101" else ""
    print(f"#{rank} {listing_id} — sim={sim:.4f}{marker}")
# %%
