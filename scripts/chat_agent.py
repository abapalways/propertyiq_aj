import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import re
from dotenv import load_dotenv
load_dotenv("../.env")
from langchain_groq import ChatGroq



import json
from langchain_ollama import ChatOllama
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from search import search_listings, SearchUnavailableError

from tools import calc_mortgage, get_comps
from memory import reject_listing, undo_last_rejection, reset_rejected_listings


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = FAISS.load_local("../faiss_index", embeddings, allow_dangerous_deserialization=True)

with open("../data/buyer_profiles.json") as f:
    buyer_profiles = json.load(f)


from typing import Optional
_last_full_analysis = None
def _search_listings_tool(min_bedrooms: int, max_budget: float, must_haves: Optional[list] = None, preferred_city: Optional[str] = None, k: int = 3, rejected_listing_ids: Optional[list] = None) -> list:
    """Search for property listings matching a buyer's criteria.

        Args:
            min_bedrooms: Minimum number of bedrooms required.
            max_budget: Maximum price the buyer will pay.
            must_haves: List of short phrases describing required features, e.g. ["garage", "good school district"]. Optional.
            preferred_city: The city the buyer wants to live in, e.g. "Austin". Optional.
            k: Number of results to return, defaults to 3.
            rejected_listing_ids: List of listing IDs to exclude, defaults to none.
        """
    buyer_preferences = {
        "min_bedrooms": min_bedrooms, "max_budget": max_budget,
        "must_haves": must_haves or [], "preferred_city": preferred_city,
    }
    try:
        results = search_listings(buyer_preferences, vectorstore, embeddings, k=k, rejected_listing_ids=rejected_listing_ids or [])
    except SearchUnavailableError:
        return {"error": "Search is temporarily unavailable right now (the search service returned an error). Please try again in a moment."}

    output = []
    for i, doc in enumerate(results, 1):
        address_match = re.search(r"\*\*Address:\*\*\s*(.+)", doc.page_content)
        address = address_match.group(1).strip() if address_match else None

        # Surface the VERIFIED dimension enrichments directly, rather than
        # relying on the truncated description[:400] to happen to include
        # the school/condition text - which it often doesn't (the Key
        # Features section, where this data lives, frequently falls after
        # the 400-char cutoff). This is the same enrichment data already
        # used internally for NLI contradiction-checking - now also
        # surfaced to the LLM so it can report it accurately instead of
        # inventing plausible-sounding details for information it never saw.
        enrichments = doc.metadata.get("dimension_enrichments", {})
        school_info = enrichments.get("school_quality", {}).get("raw_text")
        condition_info = enrichments.get("condition", {}).get("raw_text")

        output.append({
            "rank": i,
            "listing_id": doc.metadata.get("listing_id"),
            "price": doc.metadata.get("price"),
            "bedrooms": doc.metadata.get("bedrooms"),
            "garage_spaces": doc.metadata.get("garage_spaces"),
            "square_footage": doc.metadata.get("square_footage"),
            "address": address,
            "description": doc.page_content[:400],
            "school_district_info": school_info,      # None if not available - LLM must say so, not guess
            "condition_info": condition_info,          # None if not available - LLM must say so, not guess
        })
    return output

#llm = ChatGroq(model="openai/gpt-oss-120b")
llm = ChatGroq(model="llama-3.3-70b-versatile")


llm_with_tools = llm.bind_tools([
    calc_mortgage, get_comps,
    reject_listing, undo_last_rejection, reset_rejected_listings,
    _search_listings_tool,
])

TOOL_MAP = {
    "calc_mortgage": calc_mortgage,
    "get_comps": get_comps,
    "reject_listing": reject_listing,
    "undo_last_rejection": undo_last_rejection,
    "reset_rejected_listings": reset_rejected_listings,
    "_search_listings_tool": _search_listings_tool,
}

def build_system_prompt(buyer):
    if buyer is None:
        return """You are a helpful real estate assistant. The user is browsing as a guest — 
there is no saved profile and nothing is remembered between sessions. Do not ask about 
previously rejected listings, since none are tracked for guests. If the guest wants to 
exclude a specific listing from just this search, you may do so for this one search only, 
but make clear it will not be remembered next time.

CRITICAL: When describing search results, you MUST only state facts that appear 
literally in the "description", "school_district_info", or "condition_info" 
fields returned by the search tool. Do NOT infer, guess, or generate 
plausible-sounding details about schools, neighborhoods, letter grades, or 
any other attribute. If "school_district_info" or "condition_info" is null, 
say the information isn't available for that listing rather than inventing 
a plausible answer. Quote or closely paraphrase only what is actually 
written in these fields.
When presenting search results, ALWAYS format them as a markdown table with exactly 
    4 columns: "Rank", "Listing ID", "Price", and "Key Details". Rank reflects how well 
    each listing matches the buyer's fuzzy/qualitative criteria, best match first - 
    always preserve the order given by the search tool, do not reorder results yourself."""

    prefs = buyer.get("preferences", {})
    rejected = buyer.get("session_history", {}).get("rejected_listings", [])
    rejected_ids = [r["listing_id"] for r in rejected]

    known_prefs = []
    if "min_bedrooms" in prefs:
        known_prefs.append(f"min_bedrooms={prefs['min_bedrooms']}")
    if "max_budget" in prefs:
        known_prefs.append(f"max_budget={prefs['max_budget']}")
    if "must_haves" in prefs and prefs["must_haves"]:
        known_prefs.append(f"must_haves={prefs['must_haves']}")
    if "preferred_city" in prefs:
        known_prefs.append(f"preferred_city={prefs['preferred_city']}")
    if "notes" in prefs and prefs["notes"]:
        known_prefs.append(f"other preferences={prefs['notes']}")

    prefs_line = (
        f"Their known preferences: {', '.join(known_prefs)}."
        if known_prefs
        else "No preferences are on file yet for this buyer — ask them what they're looking for, or use whatever they mention in conversation, before searching."
    )

    rejected_line = (
            f"They have already rejected these listing IDs: {rejected_ids} — never suggest these again."
            if rejected_ids
            else "They have no rejected listings on file yet."
        )

    return f"""You are a helpful real estate assistant for {buyer['name']} (buyer_id: {buyer['buyer_id']}).
    {prefs_line}
    {rejected_line}
    When searching, always pass rejected_listing_ids={rejected_ids} to search_listings.
    If the buyer says they don't want a specific listing, call reject_listing with their buyer_id, 
    the listing_id, and a brief reason based on what they said.
    If the buyer wants to reconsider their most recent rejection (e.g. "actually show me that last 
    one again" or "I changed my mind"), call undo_last_rejection with their buyer_id.
    If the buyer wants to clear their entire rejection history and start over, call 
    reset_rejected_listings with their buyer_id.
    If the buyer states preferences during conversation, simply acknowledge them naturally - 
    these will be saved automatically when the session ends.
    If the buyer states specific criteria directly in their message (e.g. "3-bedroom home", 
    "under $400,000", "move-in ready"), use ONLY what is stated in that message for that search - 
    do not merge in stored preferences the buyer did not mention this turn, even if some fields 
    are on file. Only fall back to their stored preferences when the buyer does not state specific 
    criteria themselves (e.g. "show me listings that match my preferences").
    CRITICAL: When describing search results, you MUST only state facts that appear 
    literally in the "description", "school_district_info", or "condition_info" 
    fields returned by the search tool. Do NOT infer, guess, or generate 
    plausible-sounding details about schools, neighborhoods, letter grades, or 
    any other attribute. If "school_district_info" or "condition_info" is null, 
    say the information isn't available for that listing rather than inventing 
    a plausible answer. Quote or closely paraphrase only what is actually 
    written in these fields.
    When presenting search results, ALWAYS format them as a markdown table with exactly 
    4 columns: "Rank", "Listing ID", "Price", and "Key Details". Rank reflects how well 
    each listing matches the buyer's fuzzy/qualitative criteria, best match first - 
    always preserve the order given by the search tool, do not reorder results yourself."""



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