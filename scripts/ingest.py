# %%
# --- Load environment variables (LangSmith tracing config) ---
from dotenv import load_dotenv
load_dotenv("../.env")  # .env lives at project root, this script runs from scripts/

# %%
# --- Sanity check: confirm working directory and env vars loaded ---
import os
print(f"cwd: {os.getcwd()}")
print(f"TRACING_V2: {os.environ.get('LANGCHAIN_TRACING_V2')}")
print(f"PROJECT: {os.environ.get('LANGCHAIN_PROJECT')}")

# %%
# --- Load the raw corpus text ---
with open("../data/listings_corpus.md") as f:
    text = f.read()

print(f"Loaded {len(text)} characters")
text[:500]

# %%
# --- Split into chunks + build Document objects with metadata ---
# %%
import re
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

with open("../data/listings_corpus.md") as f:
    text = f.read()

splitter = CharacterTextSplitter(separator="---", chunk_size=1000, chunk_overlap=0)
raw_chunks = splitter.split_text(text)

import re
from langchain_groq import ChatGroq

_enrichment_llm = ChatGroq(model="openai/gpt-oss-120b")


def extract_school_text(chunk):
    """Pull out just the sentence(s) mentioning school district/rating."""
    match = re.search(r"^.*[Ss]chool.*$", chunk, re.MULTILINE)
    return match.group(0).strip() if match else None


def enrich_with_sentiment(school_text):
    """One-time LLM call: read school-related text, produce a short, 
    sentiment-clear phrase to append before embedding."""
    if not school_text:
        return ""
    prompt = f"""Read this real estate listing's school district description and 
summarize its quality in ONE short sentence using clear positive or negative 
language (e.g. "excellent, top-rated" or "poor, underperforming"). 
Be direct and unambiguous about whether it's good or bad.

Text: {school_text}

One-sentence summary:"""
    return _enrichment_llm.invoke(prompt).content.strip()

def extract_garage_spaces(chunk):
    if re.search(r"(does not have|no|without)\s+(a\s+)?garage", chunk, re.IGNORECASE):
        return 0
    match = re.search(r"(\d+)-car\b.*?\bgarage", chunk, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.search(r"garage", chunk, re.IGNORECASE):
        return 1
    return None

def extract_has_hoa(chunk):
    return bool(re.search(r"\bHOA\b", chunk, re.IGNORECASE))

def extract_city(chunk):
    match = re.search(r"Address:\*\*\s*.+?,\s*([A-Za-z\s]+),\s*TX", chunk)
    return match.group(1).strip() if match else None

def extract_has_private_yard(chunk):
    """True if a private/fenced yard is described positively, False if explicitly denied, None if unmentioned."""
    if re.search(r"no\s+private\s+yard", chunk, re.IGNORECASE):
        return False
    if re.search(r"(private|fenced|expansive|large)\s+(backyard|yard)", chunk, re.IGNORECASE):
        return True
    return None

documents = []
for chunk in raw_chunks:
    listing_id_match = re.search(r"Property ID:\s*(\S+)", chunk)
    price_match = re.search(r"\*\*Price:\*\*\s*\$([\d,]+)", chunk)
    bed_match = re.search(r"\*\*Bedrooms:\*\*\s*(\d+)", chunk)
    bath_match = re.search(r"\*\*Bathrooms:\*\*\s*([\d.]+)", chunk)
    sqft_match = re.search(r"\*\*Square Footage:\*\*\s*([\d,]+)", chunk)

    metadata = {
        "listing_id": listing_id_match.group(1) if listing_id_match else None,
        "price": int(price_match.group(1).replace(",", "")) if price_match else None,
        "bedrooms": int(bed_match.group(1)) if bed_match else None,
        "bathrooms": float(bath_match.group(1)) if bath_match else None,
        "square_footage": int(sqft_match.group(1).replace(",", "")) if sqft_match else None,
        "city": extract_city(chunk),
        "garage_spaces": extract_garage_spaces(chunk),
        "has_hoa": extract_has_hoa(chunk),
        "has_private_yard": extract_has_private_yard(chunk),
    }
    school_text = extract_school_text(chunk)
    sentiment_addition = enrich_with_sentiment(school_text)

    enriched_content = chunk.strip()
    if sentiment_addition:
        enriched_content += " " + sentiment_addition

    documents.append(Document(page_content=enriched_content, metadata=metadata))

print(f"Built {len(documents)} documents\n")
for doc in documents:
    print(doc.metadata)
# %%
# --- Embed a test document, confirm model + LangSmith tracing work ---
# from langchain_ollama import OllamaEmbeddings

# Old:
# embeddings = OllamaEmbeddings(model="nomic-embed-text")

# New:
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

test_vector = embeddings.embed_query(documents[0].page_content)
print(f"Vector length: {len(test_vector)}")
print(f"First 5 values: {test_vector[:5]}")

# --- Embed all documents and build the FAISS vector store ---
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(documents, embeddings)

print(f"FAISS index built with {vectorstore.index.ntotal} vectors")

# %%
# --- Persist the vector store to disk ---
vectorstore.save_local("../faiss_index")
print("Saved FAISS index to ../faiss_index")

# %%
