# %%
"""
Separate ingestion script for the EVAL corpus (data/eval_corpus.md).

This is intentionally a standalone copy of ingest.py's logic, not a shared
import, so that eval runs can NEVER accidentally touch or overwrite the real
faiss_index/ built from data/listings_corpus.md. Same extraction/enrichment
logic, different input file, different output folder.

Run from the scripts/ directory, same as ingest.py:
    python ingest_eval.py
"""
from dotenv import load_dotenv
load_dotenv("../.env")

import os
print(f"cwd: {os.getcwd()}")

from dimensions_config import FUZZY_DIMENSIONS

# --- Load the EVAL corpus (not the real one) ---
EVAL_CORPUS_PATH = "../data/eval_corpus.md"
EVAL_INDEX_PATH = "../eval_faiss_index"

with open(EVAL_CORPUS_PATH) as f:
    text = f.read()

print(f"Loaded {len(text)} characters from {EVAL_CORPUS_PATH}")

# %%
import re
from langchain_core.documents import Document
from langchain_groq import ChatGroq

_enrichment_llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
# NOTE: deliberately NOT using CharacterTextSplitter here. Its chunk_size
# parameter doesn't just split on the separator - it also MERGES adjacent
# split chunks back together if their combined size stays under chunk_size.
# Two short listings in a row (as happened with E_OVERBUDGET_208 and
# E_COMPACT_209 here) get silently merged into one chunk, and re.search()
# then only picks up the FIRST listing's metadata, silently discarding the
# second. Splitting directly on the literal separator avoids this entirely -
# each "---"-delimited block is already exactly one full listing.
raw_chunks = [chunk.strip() for chunk in text.split("---") if chunk.strip()]


def extract_dimension_text(chunk, dimension_config):
    matched_lines = []
    seen = set()
    for keyword in dimension_config["keywords"]:
        matches = re.findall(rf"^.*{keyword}.*$", chunk, re.MULTILINE | re.IGNORECASE)
        for line in matches:
            line = line.strip()
            if line not in seen:
                seen.add(line)
                matched_lines.append(line)
    if not matched_lines:
        return None
    return "\n".join(matched_lines)


def enrich_dimension(text, dimension_name):
    if not text:
        return ""
    prompt = f"""State the quality of this {dimension_name.replace('_', ' ')} 
in ONE short sentence using an EXPLICIT quality word (excellent, poor, 
outdated, top-rated, move-in-ready, needs-work, etc.) - be direct.

Text: {text}

One-sentence quality statement:"""
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

    dimension_enrichments = {}
    for dim_name, dim_config in FUZZY_DIMENSIONS.items():
        dim_text = extract_dimension_text(chunk, dim_config)
        dimension_enrichments[dim_name] = {
            "raw_text": dim_text,
            "enrichment": enrich_dimension(dim_text, dim_name) if dim_text else "",
        }
    metadata["dimension_enrichments"] = dimension_enrichments

    documents.append(Document(page_content=chunk.strip(), metadata=metadata))

print(f"\nBuilt {len(documents)} eval documents\n")
for doc in documents:
    print(doc.metadata)

# %%
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

test_vector = embeddings.embed_query(documents[0].page_content)
print(f"\nVector length: {len(test_vector)}")

vectorstore = FAISS.from_documents(documents, embeddings)
print(f"EVAL FAISS index built with {vectorstore.index.ntotal} vectors")

# %%
vectorstore.save_local(EVAL_INDEX_PATH)
print(f"Saved EVAL FAISS index to {EVAL_INDEX_PATH}")
print("\nReal listings_corpus.md and faiss_index/ were never touched by this script.")