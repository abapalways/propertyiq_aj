import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from search import search_listings
import gradio as gr

load_dotenv("../.env")

# --- One-time setup: load everything the app needs, before the UI starts ---
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)

buyer_names = [buyer.get("name") for buyer in buyer_profiles]


def get_shortlist(buyer_name):
    """Given a buyer's name (selected from dropdown), return a formatted shortlist string."""
    buyer = [b for b in buyer_profiles if b.get("name") == buyer_name][0]
    results = search_listings(buyer["preferences"], vectorstore, embeddings, k=3)

    if not results:
        return "No listings matched this buyer's criteria."

    output = f"Shortlist for {buyer_name}:\n\n"
    for doc in results:
        output += f"- {doc.metadata.get('listing_id')} — ${doc.metadata.get('price'):,} — {doc.metadata.get('bedrooms')} bed\n"
    return output


demo = gr.Interface(
    fn=get_shortlist,
    inputs=gr.Dropdown(choices=buyer_names, label="Select Buyer Profile"),
    outputs=gr.Textbox(label="Shortlist"),
    title="PropertyIQ — Prototype Shortlist Generator"
)

demo.launch(share=True)