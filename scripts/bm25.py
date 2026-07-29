# test_bm25.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dotenv import load_dotenv
load_dotenv("../.env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from search import search_listings, _last_full_analysis

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

buyer_preferences = {
    "max_budget": 500000,
    "min_bedrooms": 2,
    "must_haves": ["Pinecrest Elementary"],
}

results = search_listings(buyer_preferences, vectorstore, embeddings, k=5)

print("\n=== RESULTS ===")
for doc in results:
    print(doc.metadata.get("listing_id"))