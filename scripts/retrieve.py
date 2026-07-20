# %%
# --- Load environment variables (same as ingest.py) ---
from dotenv import load_dotenv
load_dotenv("../.env")

# %%
# --- Recreate the embeddings object (must match what built the index) ---
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# %%
# --- Load the saved FAISS index back from disk ---
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.load_local(
    "../faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print(f"Loaded index with {vectorstore.index.ntotal} vectors")
# %%
# %%
# --- Approach A: pure semantic search, no metadata filtering ---
query = "3-bed homes under $450K near good schools"

results = vectorstore.similarity_search(query, k=3)

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"listing_id: {doc.metadata.get('listing_id')}")
    print(f"price: {doc.metadata.get('price')}")
    print(f"bedrooms: {doc.metadata.get('bedrooms')}")
    print(doc.page_content[:150])
    print()
# %%
# %%
# --- Step 1: Filter by hard constraints ---
all_docs = list(vectorstore.docstore._dict.values())  # get all stored Documents back out

candidates = [
    doc for doc in all_docs
    if  doc.metadata.get("price") <= 450000  # price constraint here
    and doc.metadata.get("bedrooms") == 3 
]

print(f"{len(candidates)} candidates survived the filter:")
for doc in candidates:
    print(doc.metadata.get("listing_id"), doc.metadata.get("price"), doc.metadata.get("bedrooms"))
# %%
# %%
# --- Step 2: Semantic search, restricted to filtered candidates ---
filtered_vectorstore = FAISS.from_documents(candidates, embeddings)

results = filtered_vectorstore.similarity_search("good school", k=3)

for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(doc.metadata.get("listing_id"), doc.metadata.get("price"), doc.metadata.get("bedrooms"))
    print(doc.page_content[:150])
    print()
# %%
