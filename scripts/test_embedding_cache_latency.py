# test_embedding_cache_latency.py
import time
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dotenv import load_dotenv
load_dotenv("../.env")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from search import search_listings, _doc_embedding_cache

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

buyer_preferences = {
    "max_budget": 10000000,
    "min_bedrooms": 0,
    "must_haves": ["top-rated schools"],
}

# Clear cache to force a cold run
_doc_embedding_cache.clear()

start = time.perf_counter()
search_listings(buyer_preferences, vectorstore, embeddings, k=5)
cold_time = time.perf_counter() - start

# Run again immediately - cache should now be warm
start = time.perf_counter()
search_listings(buyer_preferences, vectorstore, embeddings, k=5)
warm_time = time.perf_counter() - start

print(f"\nCold (empty cache):  {cold_time:.3f} sec")
print(f"Warm (cache full):   {warm_time:.3f} sec")
print(f"Speedup: {cold_time/warm_time:.1f}x")