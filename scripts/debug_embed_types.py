import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import numpy as np

embeddings_gemini = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings_gemini, allow_dangerous_deserialization=True)
all_docs = list(vectorstore.docstore._dict.values())

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
llm = ChatOllama(model="qwen3:8b")

henderson_candidates = [
    doc for doc in all_docs
    if doc.metadata.get("price") <= 460000
    and doc.metadata.get("bedrooms") >= 3
]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Stage 1: HyDE
hyde_prompt = """Write exactly 1-2 sentences describing a home that matches 
this buyer request: "top-tier schools". Write it in natural real-estate 
listing style. No headers, no bullet points, no explanations."""
hyde_text = llm.invoke(hyde_prompt).content
print(f"HyDE text: {hyde_text}\n")

# Stage 2: Gemini cosine narrowing using HyDE text
hyde_vector = embeddings_gemini.embed_query(hyde_text)
scored = []
for doc in henderson_candidates:
    doc_vector = embeddings_gemini.embed_documents([doc.page_content])[0]
    sim = cosine_similarity(hyde_vector, doc_vector)
    scored.append((sim, doc))
scored.sort(key=lambda x: x[0], reverse=True)
narrowed = [doc for sim, doc in scored]

# Stage 3: cross-encoder final re-rank, same HyDE text
pairs = [[hyde_text, doc.page_content] for doc in narrowed]
scores = cross_encoder.predict(pairs)
final = sorted(zip(scores, narrowed), key=lambda x: x[0], reverse=True)

print("Final ranking (Gemini + HyDE + cross-encoder), Henderson scenario:\n")
for rank, (score, doc) in enumerate(final[:5], 1):
    listing_id = doc.metadata.get("listing_id")
    print(f"#{rank} {listing_id} — cross_score={score:.4f}")