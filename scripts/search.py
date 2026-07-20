"""
Core search logic: takes buyer preferences + a vector store, 
returns a ranked shortlist of matching listings.
"""

from langchain_community.vectorstores import FAISS


def search_listings(buyer_preferences, vectorstore, embeddings, k=3):
    """
    buyer_preferences: dict like {"min_bedrooms": 3, "max_budget": 450000, 
                                    "must_haves": ["good school district", "garage"]}
    Returns: list of Document objects, ranked
    """
    all_docs = list(vectorstore.docstore._dict.values())
    max_budget = buyer_preferences["max_budget"]
    min_bedrooms = buyer_preferences["min_bedrooms"]

    candidates = [
        doc for doc in all_docs
        if doc.metadata.get("price") <= max_budget
        and doc.metadata.get("bedrooms") >= min_bedrooms
    ]

    if not candidates:
        return []

    fuzzy_query = " ".join(buyer_preferences["must_haves"])

    filtered_vectorstore = FAISS.from_documents(candidates, embeddings)
    results = filtered_vectorstore.similarity_search(fuzzy_query, k=k)

    return results