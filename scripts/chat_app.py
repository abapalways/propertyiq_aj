import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from memory import extract_semantic_memory
import json
import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from chat_agent import llm_with_tools, build_system_prompt, buyer_profiles, TOOL_MAP, vectorstore, embeddings
from search import analyze_listings

buyer_names = [b["name"] for b in buyer_profiles] + ["Guest"]

EXAMPLE_PROMPTS = [
    "Show me listings that match my preferences.",
    "What would my monthly mortgage payment be on the first listing, with 20% down at 6.5% interest?",
    "Can you find comparable sold homes for the first listing?",
    "I don't want the second listing you showed me, it's too small.",
    "Find me a 3-bedroom home in Austin under $400,000 with a garage.",
]


def fill_prompt(prompt_text):
    return prompt_text


def get_buyer(buyer_name):
    if buyer_name == "Guest":
        return None
    return next((b for b in buyer_profiles if b["name"] == buyer_name), None)


def _run_turn(messages_state):
    """Invoke the LLM, execute any tool calls, loop until it gives a final text answer."""
    response = llm_with_tools.invoke(messages_state)
    messages_state.append(response)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_fn = TOOL_MAP[tool_call["name"]]
            result = tool_fn(**tool_call["args"])
            messages_state.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call["id"]))
        response = llm_with_tools.invoke(messages_state)
        messages_state.append(response)

    return response, messages_state

from search import _cosine_similarity

def build_trace_html(messages_state):
    if not messages_state:
        return "<p>No activity yet.</p>"

    html = "<h4 style='color: #1a1a1a;'>Agent Trace</h4>"
    tool_call_names = {}
    tool_call_args = {}

    for msg in messages_state:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_call_names[tc["id"]] = tc["name"]
                tool_call_args[tc["id"]] = tc["args"]
                args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                html += f"""
                <div style="background-color: #e7f0ff; color: #1a1a1a; padding: 8px; margin: 4px 0; border-radius: 4px; border-left: 4px solid #4285f4;">
                    🔧 <strong>Called:</strong> {tc['name']}({args_str})
                </div>
                """

        elif msg_type == "ToolMessage":
            tool_name = tool_call_names.get(msg.tool_call_id, "unknown tool")
            content_preview = str(msg.content)[:200]
            html += f"""
            <div style="background-color: #f0f0f0; color: #1a1a1a; padding: 8px; margin: 4px 0 12px 20px; border-radius: 4px; border-left: 4px solid #888;">
                📤 <strong>Result from {tool_name}:</strong> {content_preview}...
            </div>
            """

            # If this was a search, show similarity scores too
            if tool_name == "_search_listings_tool":
                args = tool_call_args.get(msg.tool_call_id, {})
                must_haves = args.get("must_haves") or []
                fuzzy_terms = [mh for mh in must_haves if "garage" not in mh.lower() and "yard" not in mh.lower()]
                if fuzzy_terms:
                    fuzzy_query = " ".join(fuzzy_terms)
                    try:
                        results = json.loads(msg.content)
                        query_vec = embeddings.embed_query(fuzzy_query)
                        html += f"""<div style="margin: 4px 0 12px 20px; font-size: 0.9em; color: #555;">
                        <em>Similarity scores for '{fuzzy_query}':</em><ul>"""
                        for r in results:
                            listing_id = r.get("listing_id")
                            doc_text = r.get("description", "")
                            if doc_text:
                                doc_vec = embeddings.embed_documents([doc_text])[0]
                                sim = _cosine_similarity(query_vec, doc_vec)
                                html += f"<li>{listing_id}: {sim:.4f}</li>"
                        html += "</ul></div>"
                    except Exception as e:
                        html += f"<div style='margin-left:20px; color:#888;'>(could not compute similarity: {e})</div>"

    return html


def start_session(buyer_name):
    """Called when a buyer is selected: resets the conversation with a fresh system prompt."""
    buyer = get_buyer(buyer_name)
    system_prompt = build_system_prompt(buyer)
    messages_state = [SystemMessage(system_prompt)]
    display_history = []
    return display_history, messages_state, build_trace_html(messages_state)


def respond(user_message, display_history, messages_state):
    messages_state.append(HumanMessage(user_message))
    response, messages_state = _run_turn(messages_state)
    display_history.append({"role": "user", "content": user_message})
    display_history.append({"role": "assistant", "content": response.content})
    return "", display_history, messages_state, build_trace_html(messages_state)

def end_session(buyer_name, messages_state):
    buyer = get_buyer(buyer_name)
    if buyer is None:
        return "Guest sessions have nothing to save."
    result = extract_semantic_memory(buyer["buyer_id"], messages_state)
    if result["applied"]:
        return "Saved: " + "; ".join(result["applied"])
    return "Nothing new to save from this session."


_analysis_cache = {}


def get_analysis(buyer_name):
    if buyer_name == "Guest":
        return []
    if buyer_name not in _analysis_cache:
        buyer = get_buyer(buyer_name)
        rejected_ids = [r["listing_id"] for r in buyer["session_history"]["rejected_listings"]]
        _analysis_cache[buyer_name] = analyze_listings(
            buyer["preferences"], vectorstore, embeddings, rejected_listing_ids=rejected_ids
        )
    return _analysis_cache[buyer_name]


def render_analysis_html(buyer_name):
    results = get_analysis(buyer_name)
    if not results:
        return "<p>Select a buyer profile to see analysis (not available for Guest).</p>"

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

    html = f"<h3 style='color: #1a1a1a;'>Analysis for {buyer_name}</h3>"
    for r in sorted_results:
        color = tier_colors[r["tier"]]
        label = tier_labels[r["tier"]]
        doc = r["doc"]
        price = doc.metadata.get("price")
        bedrooms = doc.metadata.get("bedrooms")

        html += f"""
        <div style="background-color: {color}; color: #1a1a1a; padding: 12px; margin: 8px 0; border-radius: 6px; border: 1px solid #ccc;">
            <strong style="color: #1a1a1a;">{label} — {r['listing_id']}</strong><br>
            <span style="color: #1a1a1a;">Price: ${price:,} | Bedrooms: {bedrooms}</span><br>
            <span style="color: #1a1a1a;">Reason: {r['reason']}</span>
        </div>
        """
    return html


with gr.Blocks(title="PropertyIQ — Chat Agent") as demo:
    gr.Markdown("# PropertyIQ — Chat Agent")

    with gr.Tabs():
        with gr.Tab("Chat"):
            buyer_dropdown = gr.Dropdown(choices=buyer_names, label="Shopping as", value="Guest")

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(label="Conversation")
                with gr.Column(scale=1):
                    with gr.Accordion("🔍 Agent Trace (tools & memory used)", open=False):
                        trace_output = gr.HTML()

            gr.Markdown("**Try one of these, or type your own:**")
            with gr.Row():
                example_buttons = [gr.Button(p, size="sm") for p in EXAMPLE_PROMPTS]

            msg_box = gr.Textbox(label="Message", placeholder="Type a message...")
            messages_state = gr.State([])

            buyer_dropdown.change(fn=start_session, inputs=buyer_dropdown, outputs=[chatbot, messages_state, trace_output])
            demo.load(fn=start_session, inputs=buyer_dropdown, outputs=[chatbot, messages_state, trace_output])

            for btn in example_buttons:
                btn.click(fn=fill_prompt, inputs=btn, outputs=msg_box)

            msg_box.submit(fn=respond, inputs=[msg_box, chatbot, messages_state], outputs=[msg_box, chatbot, messages_state, trace_output])

            with gr.Row():
                end_session_btn = gr.Button("💾 End Session (Save Memory)")
                end_session_output = gr.Textbox(label="Memory extraction result", interactive=False)

            end_session_btn.click(fn=end_session, inputs=[buyer_dropdown, messages_state], outputs=end_session_output)

            with gr.Tab("Debug: Full Analysis (temporary)"):
                debug_output = gr.HTML()
                debug_refresh_btn = gr.Button("🔄 Refresh Analysis for Current Chat Buyer")
                debug_refresh_btn.click(fn=render_analysis_html, inputs=buyer_dropdown, outputs=debug_output)



if __name__ == "__main__":
    demo.launch(share=True)