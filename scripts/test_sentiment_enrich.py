import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from dotenv import load_dotenv
load_dotenv('../.env')

import numpy as np
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings

llm = ChatGroq(model="openai/gpt-oss-120b")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def enrich_with_sentiment(school_text):
    """One-time LLM call: read school-related text, produce a short, 
    sentiment-clear phrase to append before embedding."""
    prompt = f"""Read this real estate listing's school district description and 
summarize its quality in ONE short sentence using clear positive or negative 
language (e.g. "excellent, top-rated" or "poor, underperforming"). 
Be direct and unambiguous about whether it's good or bad.

Text: {school_text}

One-sentence summary:"""
    return llm.invoke(prompt).content.strip()


# Real excerpts from your corpus
pine_school_text = "Situated 2 blocks away from the top-rated Pinecrest Elementary School (ranked #1 in the district)."
maple_school_text = "Located within the Northside School District (currently rated 4/10 on state performance indexes). Perfect for buyers prioritizing raw space, square footage, and home offices over school performance indicators."

pine_enrichment = enrich_with_sentiment(pine_school_text)
maple_enrichment = enrich_with_sentiment(maple_school_text)

print(f"L_PINE_101 enrichment: {pine_enrichment}")
print(f"L_MAPLE_105 enrichment: {maple_enrichment}\n")

# Now compare: original text alone vs. text + enrichment, against "good school district"
query = "good school district"
query_vec = embeddings.embed_query(query)

for label, text in [
    ("L_PINE_101 (original only)", pine_school_text),
    ("L_PINE_101 (with enrichment)", pine_school_text + " " + pine_enrichment),
    ("L_MAPLE_105 (original only)", maple_school_text),
    ("L_MAPLE_105 (with enrichment)", maple_school_text + " " + maple_enrichment),
]:
    doc_vec = embeddings.embed_documents([text])[0]
    sim = cosine_similarity(query_vec, doc_vec)
    print(f"{label}: sim={sim:.4f}")

print("\n--- Full enriched text (what would actually be embedded) ---\n")
print("L_PINE_101 enriched:")
print(pine_school_text + " " + pine_enrichment)
print("\nL_MAPLE_105 enriched:")
print(maple_school_text + " " + maple_enrichment)