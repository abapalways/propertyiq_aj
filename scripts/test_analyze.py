import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from search import analyze_listings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)

torres = buyer_profiles[0]
results = analyze_listings(torres["preferences"], vectorstore, embeddings)

print(f"Analysis for {torres['name']}:\n")
for r in sorted(results, key=lambda x: {"matched": 0, "passed_but_not_selected": 1, "hard_filter_rejected": 2}[x["tier"]]):
    print(f"[{r['tier']:25}] {r['listing_id']:15} — {r['reason']}")