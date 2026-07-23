import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

from mcp.server.fastmcp import FastMCP
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from tools import calc_mortgage, get_comps, reject_listing
from search import search_listings

mcp = FastMCP("PropertyIQ")

# Load once at server startup, reused across all tool calls
_embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
_vectorstore = FAISS.load_local("../faiss_index", _embeddings, allow_dangerous_deserialization=True)


@mcp.tool()
def calc_mortgage_tool(price: float, down_payment_pct: float, interest_rate: float, loan_term_years: int = 30) -> dict:
    """Calculate the estimated monthly mortgage payment for a home purchase.

    Args:
        price: Home price in dollars, must be positive.
        down_payment_pct: Down payment as a decimal fraction, e.g. 0.20 for 20%.
        interest_rate: Annual interest rate as a decimal, e.g. 0.065 for 6.5%.
        loan_term_years: Loan term in years, defaults to 30.
    """
    return calc_mortgage(price, down_payment_pct, interest_rate, loan_term_years)


@mcp.tool()
def get_comps_tool(listing_id: str, target_listing_metadata: dict) -> dict:
    """Find comparable recently-sold properties for a given active listing.

    Args:
        listing_id: The target listing's ID.
        target_listing_metadata: Dict with the target's city, bedrooms, and square_footage.
    """
    return get_comps(listing_id, target_listing_metadata)


@mcp.tool()
def search_listings_tool(min_bedrooms: int, max_budget: float, must_haves: list, k: int = 3, rejected_listing_ids: list = None) -> list:
    """Search for property listings matching a buyer's criteria.

    Args:
        min_bedrooms: Minimum number of bedrooms required.
        max_budget: Maximum price the buyer will pay.
        must_haves: List of short phrases describing required features, e.g. ["garage", "good school district"].
        k: Number of results to return, defaults to 3.
        rejected_listing_ids: List of listing IDs to exclude (already rejected by this buyer), defaults to none.
    """
    buyer_preferences = {"min_bedrooms": min_bedrooms, "max_budget": max_budget, "must_haves": must_haves}
    results = search_listings(buyer_preferences, _vectorstore, _embeddings, k=k, rejected_listing_ids=rejected_listing_ids or [])
    return [
        {"listing_id": doc.metadata.get("listing_id"), "price": doc.metadata.get("price"),
         "bedrooms": doc.metadata.get("bedrooms"), "description_preview": doc.page_content[:200]}
        for doc in results
    ]


@mcp.tool()
def reject_listing_tool(buyer_id: str, listing_id: str, reason: str) -> dict:
    """Record that a buyer has rejected a specific listing, so it will not 
    be suggested to them again.

    Args:
        buyer_id: The buyer's ID, e.g. "B001".
        listing_id: The listing's ID being rejected.
        reason: A short explanation of why the buyer rejected it.
    """
    return reject_listing(buyer_id, listing_id, reason)


if __name__ == "__main__":
    mcp.run(transport="stdio")