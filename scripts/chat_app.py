import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from memory import extract_semantic_memory
import json
import gradio as gr
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import search
import chat_agent  # needed to read the live-updated module variable, not a stale import snapshot

from chat_agent import llm_with_tools, build_system_prompt, buyer_profiles, TOOL_MAP, vectorstore, embeddings

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #1E2A28;
    --paper: #EEF0EA;
    --paper-raised: #F7F8F4;
    --rule: #C3CDC2;
    --pine: #2F6F5E;
    --pine-tint: #DEE9E2;
    --rust: #B5482F;
    --rust-tint: #F3DFD8;
    --brass: #9C7A1E;
    --brass-tint: #F0E6C8;
}

.gradio-container {
    max-width: 1360px !important;
    margin: auto !important;
    background: var(--paper) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 17px !important;
    color: var(--ink) !important;
}

h1, h2, h3, .prose h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
    color: var(--ink) !important;
}

.tab-nav { border-bottom: 1px solid var(--rule) !important; gap: 4px !important; }
.tab-nav button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: var(--ink) !important;
    opacity: 0.55;
    border: none !important;
    background: transparent !important;
}
.tab-nav button.selected {
    opacity: 1;
    color: var(--pine) !important;
    border-bottom: 2px solid var(--pine) !important;
}

.block, .form {
    background: var(--paper-raised) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 6px !important;
}

button.primary, button[variant="primary"] {
    background: var(--pine) !important;
    border: 1px solid var(--pine) !important;
    color: var(--paper) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    font-size: 14px !important;
    border-radius: 4px !important;
}
button.primary:hover { background: #255c4e !important; }

button.secondary, button:not(.primary) {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 14px !important;
    border-radius: 4px !important;
    border: 1px solid var(--rule) !important;
    color: var(--ink) !important;
}

label, .label-wrap span {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    letter-spacing: 0.03em !important;
    text-transform: uppercase !important;
    color: var(--ink) !important;
    opacity: 0.75;
}

input, textarea {
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 17px !important;
    border: 1px solid var(--rule) !important;
    border-radius: 4px !important;
}

/* Chatbot message bubbles - Gradio nests these deeply, target broadly */
.message, .message-wrap, .message-content, [data-testid="bot"], [data-testid="user"] {
    font-size: 18px !important;
    line-height: 1.6 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
.message p, .message li, .message td, .message th {
    font-size: 18px !important;
}

table {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 15px !important;
    border-collapse: collapse !important;
}
table th, table td {
    padding: 10px 12px !important;
    border-bottom: 1px solid var(--rule) !important;
}
table th {
    font-size: 12px !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    opacity: 0.65;
    background: transparent !important;
}
"""

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
    """Invoke the LLM, execute any tool calls, loop until it gives a final text answer.
    Deduplicates repeated _search_listings_tool calls within a single turn -
    if the LLM tries to call it again with the exact same args it already
    used earlier this turn, skip re-executing and reuse the prior result."""
    response = llm_with_tools.invoke(messages_state)
    messages_state.append(response)

    search_calls_this_turn = {}

    while response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call["name"] == "_search_listings_tool":
                call_key = json.dumps(tool_call["args"], sort_keys=True, default=str)
                if call_key in search_calls_this_turn:
                    print(f"SKIPPED duplicate _search_listings_tool call this turn: {tool_call['args']}")
                    result = search_calls_this_turn[call_key]
                else:
                    tool_fn = TOOL_MAP[tool_call["name"]]
                    result = tool_fn(**tool_call["args"])
                    search_calls_this_turn[call_key] = result
            else:
                tool_fn = TOOL_MAP[tool_call["name"]]
                result = tool_fn(**tool_call["args"])

            messages_state.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call["id"]))
        response = llm_with_tools.invoke(messages_state)
        messages_state.append(response)

    return response, messages_state


def build_trace_html(messages_state):
    if not messages_state:
        return "<p>No activity yet.</p>"

    html = "<h4 style='font-family: Fraunces, serif; color: #1E2A28; font-weight:600;'>Agent Trace</h4>"

    stamp_base = ("display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:11px; "
                  "letter-spacing:0.05em; text-transform:uppercase; padding:5px 12px; margin-right:10px; "
                  "border-radius:3px; transform:rotate(-1deg);")

    if _last_guardrail_status["blocked"]:
        html += f"""<div style="{stamp_base} border:1.5px solid #B5482F; color:#B5482F; background:#F3DFD8;">
        ⛔ Blocked</div>"""
    else:
        html += f"""<div style="{stamp_base} border:1.5px solid #2F6F5E; color:#2F6F5E; background:#DEE9E2;">
        ✓ Passed</div>"""

    stats = search._last_cache_stats
    total = stats["hits"] + stats["misses"]
    if total > 0:
        html += f"""<div style="{stamp_base} border:1.5px solid #9C7A1E; color:#9C7A1E; background:#F0E6C8; transform:rotate(1deg);">
        {stats['hits']} hits / {stats['misses']} misses</div><br><br>"""
    else:
        html += "<br><br>"

    tool_call_names = {}

    for msg in messages_state:
        msg_type = type(msg).__name__

        if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                tool_call_names[tc["id"]] = tc["name"]
                args_str = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                html += f"""
                <div style="background-color: #DEE9E2; color: #1E2A28; padding: 8px 12px; margin: 4px 0; border-radius: 4px; border-left: 3px solid #2F6F5E; font-family:'IBM Plex Mono',monospace; font-size:13px;">
                    → <strong>{tc['name']}</strong>({args_str})
                </div>
                """

        elif msg_type == "ToolMessage":
            tool_name = tool_call_names.get(msg.tool_call_id, "unknown tool")
            content_preview = str(msg.content)[:200]
            html += f"""
            <div style="background-color: #F7F8F4; color: #1E2A28; padding: 8px 12px; margin: 4px 0 12px 20px; border-radius: 4px; border-left: 3px solid #C3CDC2; font-family:'IBM Plex Mono',monospace; font-size:12px;">
                ← <strong>{tool_name}</strong>: {content_preview}...
            </div>
            """

            if tool_name == "_search_listings_tool":
                if search._last_full_analysis:
                    matched = sorted(
                        [r for r in search._last_full_analysis if r["tier"] == "matched"],
                        key=lambda r: r.get("_rrf_score", 0),
                        reverse=True,
                    )
                    html += """<div style="margin: 4px 0 12px 20px; font-size: 13px; font-family:'IBM Plex Mono',monospace; color: #1E2A28; opacity:0.75;">
                    <em>Search results (from the actual search):</em><ul>"""
                    for r in matched:
                        html += f"<li>{r['listing_id']}: {r['reason']}</li>"
                    html += "</ul></div>"

    return html


def start_session(buyer_name):
    """Called when a buyer is selected: resets the conversation with a fresh system prompt."""
    buyer = get_buyer(buyer_name)
    system_prompt = build_system_prompt(buyer)
    messages_state = [SystemMessage(system_prompt)]
    display_history = []
    return display_history, messages_state, build_trace_html(messages_state)


from guardrails import Guard
from guardrails.errors import ValidationError
from guardrails_check import FairHousingCheck, OnTopicCheck

_fair_housing_guard = Guard().use(FairHousingCheck())
_on_topic_guard = Guard().use(OnTopicCheck())


def _render_results_html(results, label):
    tier_order = {"matched": 0, "passed_but_not_selected": 1, "hard_filter_rejected": 2}
    tier_colors = {"matched": "#DEE9E2", "passed_but_not_selected": "#F0E6C8", "hard_filter_rejected": "#F3DFD8"}
    tier_border = {"matched": "#2F6F5E", "passed_but_not_selected": "#9C7A1E", "hard_filter_rejected": "#B5482F"}
    tier_labels = {"matched": "MATCHED", "passed_but_not_selected": "PASSED, NOT SELECTED", "hard_filter_rejected": "REJECTED"}

    sorted_results = sorted(results, key=lambda x: (tier_order[x["tier"]], -x.get("_rrf_score", 0)))
    html = f"<h3 style='font-family: Fraunces, serif; color: #1E2A28; font-weight:600;'>Analysis for {label}</h3>"
    for r in sorted_results:
        color = tier_colors[r["tier"]]
        border = tier_border[r["tier"]]
        label_text = tier_labels[r["tier"]]
        doc = r["doc"]
        price = doc.metadata.get("price")
        bedrooms = doc.metadata.get("bedrooms")
        html += f"""
        <div style="background-color: {color}; color: #1E2A28; padding: 14px 16px; margin: 8px 0; border-radius: 6px; border-left: 3px solid {border};">
            <strong style="font-family:'IBM Plex Mono',monospace; font-size:12px; letter-spacing:0.04em; text-transform:uppercase;">{label_text}</strong>
            <strong style="font-family:'IBM Plex Mono',monospace;"> — {r['listing_id']}</strong><br>
            <span style="font-family:'IBM Plex Mono',monospace; font-size:13px;">${price:,} · {bedrooms} bed</span><br>
            <span style="font-size:14px;">{r['reason']}</span>
        """
        if r.get("nli_checks"):
            html += "<div style='margin-top:6px; font-size:12px; font-family:\"IBM Plex Mono\",monospace; color:#1E2A28; opacity:0.7;'><em>NLI trail:</em><ul>"
            for check in r["nli_checks"]:
                if check["outcome"] == "checked":
                    html += (
                        f"<li>'{check['criterion']}' → dimension={check['classified_dimension']} "
                        f"(conf={check['classification_score']}) → NLI={check['nli_label']} "
                        f"scores={check['nli_scores']}</li>"
                    )
                else:
                    html += f"<li>'{check['criterion']}' → {check['outcome']}</li>"
            html += "</ul></div>"
        html += "</div>"
    return html


def render_current_query_analysis():
    if not search._last_full_analysis:
        return "<p>No search has been run yet.</p>"
    return _render_results_html(search._last_full_analysis, "most recent search")


_last_guardrail_status = {"blocked": False, "message": None}
_guardrail_log = []


def respond(user_message, display_history, messages_state):
    global _last_guardrail_status, _guardrail_log
    try:
        _fair_housing_guard.validate(user_message)
        _on_topic_guard.validate(user_message)
        _last_guardrail_status = {"blocked": False, "message": None}
        _guardrail_log.append({"blocked": False, "message": user_message[:80]})
    except ValidationError as e:
        error_text = str(e)
        if "errors:" in error_text:
            error_text = error_text.split("errors:", 1)[1].strip()
        _last_guardrail_status = {"blocked": True, "message": error_text}
        _guardrail_log.append({"blocked": True, "message": user_message[:80]})
        display_history.append({"role": "user", "content": user_message})
        display_history.append({"role": "assistant", "content": error_text})
        return "", display_history, messages_state, build_trace_html(messages_state)

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


def render_dashboard():
    if not search._search_log and not _guardrail_log:
        return "<p>No activity yet this session.</p>"

    total_guardrail_checks = len(_guardrail_log)
    blocked_count = sum(1 for entry in _guardrail_log if entry["blocked"])
    block_rate = (blocked_count / total_guardrail_checks * 100) if total_guardrail_checks else 0

    total_searches = len(search._search_log)
    zero_result_searches = sum(1 for entry in search._search_log if entry["num_results"] == 0)
    avg_results = (sum(entry["num_results"] for entry in search._search_log) / total_searches) if total_searches else 0

    html = "<h3 style='font-family: Fraunces, serif; color: #1E2A28; font-weight:600;'>Session Dashboard</h3>"

    html += f"""
    <div style="display:flex; gap:16px; margin-bottom:24px;">
        <div style="background:#F7F8F4; padding:20px 22px; border-radius:6px; border-left:3px solid #2F6F5E; flex:1;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.05em; text-transform:uppercase; opacity:0.65;">Guardrail trigger rate</div>
            <div style="font-family:Fraunces,serif; font-size:26px; font-weight:600; margin-top:6px;">{blocked_count} / {total_guardrail_checks}</div>
            <div style="font-size:13px; opacity:0.7; margin-top:2px;">blocked ({block_rate:.0f}%)</div>
        </div>
        <div style="background:#F7F8F4; padding:20px 22px; border-radius:6px; border-left:3px solid #9C7A1E; flex:1;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:11px; letter-spacing:0.05em; text-transform:uppercase; opacity:0.65;">Match rate</div>
            <div style="font-family:Fraunces,serif; font-size:26px; font-weight:600; margin-top:6px;">{total_searches - zero_result_searches} / {total_searches}</div>
            <div style="font-size:13px; opacity:0.7; margin-top:2px;">avg {avg_results:.1f} results per search</div>
        </div>
    </div>
    """

    html += "<h4 style='font-family: Fraunces, serif; color: #1E2A28; font-weight:600;'>Search-Filter Compliance Log</h4>"
    html += "<table style='width:100%;'>"
    html += "<tr><th style='text-align:left;'>Time</th><th style='text-align:left;'>Budget</th><th style='text-align:left;'>Min Beds</th><th style='text-align:left;'>City</th><th style='text-align:left;'>Must-Haves</th><th style='text-align:left;'>Results</th></tr>"
    for entry in reversed(search._search_log[-20:]):
        f = entry["filters"]
        html += f"""<tr>
            <td>{entry['timestamp'][11:19]}</td>
            <td>${f['max_budget']:,}</td>
            <td>{f['min_bedrooms']}</td>
            <td>{f['preferred_city'] or '—'}</td>
            <td>{', '.join(f['must_haves']) if f['must_haves'] else '—'}</td>
            <td>{entry['num_results']}</td>
        </tr>"""
    html += "</table>"

    html += "<h4 style='font-family: Fraunces, serif; color: #1E2A28; font-weight:600; margin-top:24px;'>Guardrail Log</h4>"
    html += "<table style='width:100%;'>"
    html += "<tr><th style='text-align:left;'>Status</th><th style='text-align:left;'>Message</th></tr>"
    for entry in reversed(_guardrail_log[-20:]):
        if entry["blocked"]:
            status = "<span style='color:#B5482F; font-weight:600;'>⛔ Blocked</span>"
        else:
            status = "<span style='color:#2F6F5E; font-weight:600;'>✓ Passed</span>"
        html += f"""<tr>
            <td>{status}</td>
            <td style="font-family:'IBM Plex Sans',sans-serif;">{entry['message']}</td>
        </tr>"""
    html += "</table>"

    return html


with gr.Blocks(title="PropertyIQ — Chat Agent", css=CUSTOM_CSS) as demo:
    gr.Markdown("# PropertyIQ — Chat Agent")

    with gr.Tabs():
        trace_output = gr.HTML(render=False)  # created now, placed visually in its own tab below

        with gr.Tab("Chat"):
            buyer_dropdown = gr.Dropdown(choices=buyer_names, label="Shopping as", value="Guest")

            chatbot = gr.Chatbot(label="Conversation", height=600)

            with gr.Row():
                msg_box = gr.Textbox(label="Message", placeholder="Type a message...", scale=5)
                send_btn = gr.Button("Send", variant="primary", scale=1)

            messages_state = gr.State([])

            with gr.Accordion("Try one of these", open=False):
                with gr.Row():
                    example_buttons = [gr.Button(p, size="sm") for p in EXAMPLE_PROMPTS]

            with gr.Accordion("Session", open=False):
                with gr.Row():
                    end_session_btn = gr.Button("💾 End Session (Save Memory)")
                    end_session_output = gr.Textbox(label="Memory extraction result", interactive=False)

            buyer_dropdown.change(fn=start_session, inputs=buyer_dropdown, outputs=[chatbot, messages_state, trace_output])
            demo.load(fn=start_session, inputs=buyer_dropdown, outputs=[chatbot, messages_state, trace_output])

            for btn in example_buttons:
                btn.click(fn=fill_prompt, inputs=btn, outputs=msg_box)

            msg_box.submit(fn=respond, inputs=[msg_box, chatbot, messages_state], outputs=[msg_box, chatbot, messages_state, trace_output])
            send_btn.click(fn=respond, inputs=[msg_box, chatbot, messages_state], outputs=[msg_box, chatbot, messages_state, trace_output])

            end_session_btn.click(fn=end_session, inputs=[buyer_dropdown, messages_state], outputs=end_session_output)

        with gr.Tab("🔍 Agent Trace"):
            trace_output.render()

        with gr.Tab("Debug: Full Analysis (temporary)"):
            debug_output = gr.HTML()
            debug_refresh_btn = gr.Button("🔄 Refresh Analysis for Current Chat Buyer")
            debug_refresh_btn.click(fn=render_current_query_analysis, inputs=[], outputs=debug_output)

        with gr.Tab("📊 Dashboard"):
            dashboard_output = gr.HTML()
            dashboard_refresh_btn = gr.Button("🔄 Refresh Dashboard")
            dashboard_refresh_btn.click(fn=render_dashboard, inputs=[], outputs=dashboard_output)


if __name__ == "__main__":
    demo.launch(share=True)