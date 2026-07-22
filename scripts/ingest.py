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
    documents.append(Document(page_content=chunk.strip(), metadata=metadata))

print(f"Built {len(documents)} documents\n")
for doc in documents:
    print(doc.metadata)
# %%
# --- Embed a test document, confirm model + LangSmith tracing work ---
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

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
