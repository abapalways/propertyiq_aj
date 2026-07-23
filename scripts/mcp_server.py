from mcp.server.fastmcp import FastMCP
from tools import calc_mortgage, get_comps

mcp = FastMCP("PropertyIQ")


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
        listing_id: The target listing's ID, used only for error messages.
        target_listing_metadata: Dict with the target's city, bedrooms, and square_footage.
    """
    return get_comps(listing_id, target_listing_metadata)


if __name__ == "__main__":
    mcp.run(transport="stdio")