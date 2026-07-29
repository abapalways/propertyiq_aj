import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv('../.env')

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import search

embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
vectorstore = FAISS.load_local('../faiss_index', embeddings, allow_dangerous_deserialization=True)

buyer_preferences = {
    'min_bedrooms': 3,
    'max_budget': 450000,
    'must_haves': ['garage', 'good school district'],
    'preferred_city': 'Austin',
}

all_docs = list(vectorstore.docstore._dict.values())
all_docs = [d for d in all_docs if d.metadata.get('listing_id') not in ['L_WALNUT_107']]

candidates = [d for d in all_docs if d.metadata.get('price') <= 450000 and d.metadata.get('bedrooms') >= 3]
candidates = [d for d in candidates if d.metadata.get('city') == 'Austin']

hard_filter_applied = False
fuzzy_must_haves = []
for mh in buyer_preferences['must_haves']:
    match = search._matches_checkable(mh)
    print(f'must_have={mh!r} matches_checkable={match}')
    if match:
        key, check_fn = match
        candidates = [d for d in candidates if check_fn(d.metadata.get(key))]
        hard_filter_applied = True
    else:
        fuzzy_must_haves.append(mh)

print(f'hard_filter_applied={hard_filter_applied}')
print(f'fuzzy_must_haves={fuzzy_must_haves}')
print(f'candidates after all hard filters: {[d.metadata.get("listing_id") for d in candidates]}')

fuzzy_query = ' '.join(fuzzy_must_haves)
print(f'fuzzy_query={fuzzy_query!r}')

from langchain_community.vectorstores import FAISS

filtered_vectorstore = FAISS.from_documents(candidates, embeddings)
results = filtered_vectorstore.similarity_search(fuzzy_query, k=3)
print(f'{len(results)} final results:')
for doc in results:
    print(doc.metadata.get('listing_id'))