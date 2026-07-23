import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from search import search_listings, analyze_listings
import gradio as gr

load_dotenv("../.env")

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)

buyer_names = [buyer.get("name") for buyer in buyer_profiles]


def render_analysis_html(buyer_name):
    buyer = [b for b in buyer_profiles if b.get("name") == buyer_name][0]
    results = analyze_listings(buyer["preferences"], vectorstore, embeddings)

    tier_order = {"matched": 0, "passed_but_not_selected": 1, "hard_filter_rejected": 2}
    tier_colors = {
        "matched": "#d4edda",
        "passed_but_not_selected": "#fff3cd",
        "hard_filter_rejected": "#f8d7da",
    }
    tier_labels = {
        "matched": "✅ MATCHED",
        "passed_but_not_selected": "🟡 PASSED FILTERS, NOT SELECTED",
        "hard_filter_rejected": "❌ REJECTED",
    }

    sorted_results = sorted(results, key=lambda x: tier_order[x["tier"]])

    html = f"<h3>Analysis for {buyer_name}</h3>"
    for r in sorted_results:
        color = tier_colors[r["tier"]]
        label = tier_labels[r["tier"]]
        doc = r["doc"]
        price = doc.metadata.get("price")
        bedrooms = doc.metadata.get("bedrooms")

        html += f"""
        <div style="background-color: {color}; padding: 12px; margin: 8px 0; border-radius: 6px; border: 1px solid #ccc;">
            <strong>{label} — {r['listing_id']}</strong><br>
            Price: ${price:,} | Bedrooms: {bedrooms}<br>
            Reason: {r['reason']}
        </div>
        """
    return html


def get_shortlist(buyer_name):
    buyer = [b for b in buyer_profiles if b.get("name") == buyer_name][0]
    results = search_listings(buyer["preferences"], vectorstore, embeddings, k=3)

    if not results:
        return "No listings matched this buyer's criteria."

    output = f"Shortlist for {buyer_name}:\n\n"
    for doc in results:
        output += f"- {doc.metadata.get('listing_id')} — ${doc.metadata.get('price'):,} — {doc.metadata.get('bedrooms')} bed\n"
    return output


with gr.Blocks(title="PropertyIQ — Prototype Shortlist Generator") as demo:
    gr.Markdown("# PropertyIQ — Prototype Shortlist Generator")

    with gr.Tabs():
        with gr.Tab("Shortlist"):
            buyer_dropdown = gr.Dropdown(choices=buyer_names, label="Select Buyer Profile")
            output_box = gr.Textbox(label="Shortlist")
            submit_btn = gr.Button("Submit")
            submit_btn.click(fn=get_shortlist, inputs=buyer_dropdown, outputs=output_box)

        with gr.Tab("Debug: Full Analysis (temporary)"):
            debug_dropdown = gr.Dropdown(choices=buyer_names, label="Select Buyer Profile")
            debug_output = gr.HTML()
            debug_dropdown.change(fn=render_analysis_html, inputs=debug_dropdown, outputs=debug_output)

    demo.load(fn=render_analysis_html, inputs=debug_dropdown, outputs=debug_output)

demo.launch(share=True)