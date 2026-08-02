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
from dimensions_config import FUZZY_DIMENSIONS

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

_enrichment_llm = ChatGroq(model="openai/gpt-oss-120b", temperature = 0)

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
def enrich_listing_sentiment(full_listing_text):
    """Read the entire listing and add explicit, unambiguous sentiment 
    clarifications for any subjective/qualitative claims (school quality, 
    condition, neighborhood character, etc.) - not just schools. Returns 
    the clarifying sentences to append, or empty string if nothing needs it."""
    prompt = f"""Read this real estate listing. Identify any SUBJECTIVE or 
QUALITATIVE claims that could be ambiguous to a search system - things like 
school ratings, home condition, renovation needs, or neighborhood character. 
Do NOT flag purely factual/objective data (price, square footage, bedroom 
count, address) - those need no clarification.

For each subjective claim you find, add ONE short sentence using an EXPLICIT 
quality adjective ("poor", "excellent", "underperforming", "outdated", 
"move-in-ready", etc.) - do not describe who the home is "ideal for" or 
imply quality indirectly. State the quality directly, e.g. "This school 
district is poor and underperforming" NOT "This home suits buyers who value 
other things over schools."
...

Output ONLY the clarifying sentences, one per claim, nothing else. If there 
is nothing genuinely subjective/ambiguous to clarify, output nothing.

Listing:
{full_listing_text}

Clarifying sentences:"""
    response = _enrichment_llm.invoke(prompt).content.strip()
    return response

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
    dimension_enrichments = {}
    for dim_name, dim_config in FUZZY_DIMENSIONS.items():
        dim_text = extract_dimension_text(chunk, dim_config)
        dimension_enrichments[dim_name] = {
            "raw_text": dim_text,
            "enrichment": enrich_dimension(dim_text, dim_name) if dim_text else "",
        }
    metadata["dimension_enrichments"] = dimension_enrichments
    documents.append(Document(page_content=chunk.strip(), metadata=metadata))

# OLD:
# school_text = extract_school_text(chunk)
# sentiment_addition = enrich_with_sentiment(school_text)

# NEW:

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

