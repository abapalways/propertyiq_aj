import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

import json
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from tools import calc_mortgage, get_comps, reject_listing
from search import search_listings

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)


def _search_listings_tool(min_bedrooms: int, max_budget: float, must_haves: list, k: int = 3, rejected_listing_ids: list = None) -> list:
    """Search for property listings matching a buyer's criteria.

    Args:
        min_bedrooms: Minimum number of bedrooms required.
        max_budget: Maximum price the buyer will pay.
        must_haves: List of short phrases describing required features, e.g. ["garage", "good school district"].
        k: Number of results to return, defaults to 3.
        rejected_listing_ids: List of listing IDs to exclude, defaults to none.
    """
    buyer_preferences = {"min_bedrooms": min_bedrooms, "max_budget": max_budget, "must_haves": must_haves}
    results = search_listings(buyer_preferences, vectorstore, embeddings, k=k, rejected_listing_ids=rejected_listing_ids or [])
    return [
        {"listing_id": doc.metadata.get("listing_id"), "price": doc.metadata.get("price"),
         "bedrooms": doc.metadata.get("bedrooms"), "garage_spaces": doc.metadata.get("garage_spaces")}
        for doc in results
    ]


llm = ChatOllama(model="qwen3:8b")
llm_with_tools = llm.bind_tools([calc_mortgage, get_comps, reject_listing, _search_listings_tool])

TOOL_MAP = {
    "calc_mortgage": calc_mortgage,
    "get_comps": get_comps,
    "reject_listing": reject_listing,
    "_search_listings_tool": _search_listings_tool,
}


def build_system_prompt(buyer):
    if buyer is None:
        return "You are a helpful real estate assistant. The user is browsing as a guest (no saved profile, no memory of past interactions)."

    prefs = buyer["preferences"]
    rejected = buyer["session_history"]["rejected_listings"]
    rejected_ids = [r["listing_id"] for r in rejected]

    return f"""You are a helpful real estate assistant for {buyer['name']} (buyer_id: {buyer['buyer_id']}).
Their preferences: min_bedrooms={prefs['min_bedrooms']}, max_budget={prefs['max_budget']}, 
must_haves={prefs['must_haves']}, preferred_city={prefs['preferred_city']}.
They have already rejected these listing IDs: {rejected_ids} — never suggest these again.
When searching, always pass rejected_listing_ids={rejected_ids} to search_listings.
If the buyer says they don't want a specific listing, call reject_listing with their buyer_id, 
the listing_id, and a brief reason based on what they said."""


def run_conversation(buyer_name):
    buyer = next((b for b in buyer_profiles if b["name"] == buyer_name), None) if buyer_name != "Guest" else None
    system_prompt = build_system_prompt(buyer)

    messages = [SystemMessage(system_prompt)]
    print(f"\n{'='*60}\nChatting as: {buyer_name}\n{'='*60}")
    print("Type 'quit' to exit.\n")

    # Kick off with an initial search if logged in
    if buyer:
        messages.append(HumanMessage("Please show me some listings that match my preferences."))

    while True:
        if len(messages) > 1 or buyer is None:
            if buyer is None and len(messages) == 1:
                user_input = input("You: ")
                if user_input.lower() == "quit":
                    break
                messages.append(HumanMessage(user_input))

        response = llm_with_tools.invoke(messages)
        messages.append(response)

        while response.tool_calls:
            for tool_call in response.tool_calls:
                tool_fn = TOOL_MAP[tool_call["name"]]
                result = tool_fn(**tool_call["args"])
                messages.append(ToolMessage(content=json.dumps(result, default=str), tool_call_id=tool_call["id"]))
            response = llm_with_tools.invoke(messages)
            messages.append(response)

        print(f"\nAssistant: {response.content}\n")

        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        messages.append(HumanMessage(user_input))


if __name__ == "__main__":
    print("Available buyers:", [b["name"] for b in buyer_profiles] + ["Guest"])
    buyer_name = input("Who are you shopping as? ")
    run_conversation(buyer_name)