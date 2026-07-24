"""
Deterministic financial tools for PropertyIQ: mortgage calculation and 
comparable listings lookup. No LLM/embeddings involved — pure Python logic.
"""


def calc_mortgage(price: float, down_payment_pct: float, interest_rate: float, loan_term_years: int = 30) -> dict:
    """Calculate the estimated monthly mortgage payment for a home purchase.

    Args:
        price: Home price in dollars, must be positive.
        down_payment_pct: Down payment as a decimal fraction, e.g. 0.20 for 20%.
        interest_rate: Annual interest rate as a decimal, e.g. 0.065 for 6.5%.
        loan_term_years: Loan term in years, defaults to 30.

    Returns:
        A dict with monthly_payment, loan_amount, and total_interest_paid.
    """
    if price is None or down_payment_pct is None or interest_rate is None:
        raise TypeError("price, down_payment_pct, and interest_rate are required")

    if price <= 0:
        raise ValueError("price must be positive")

    if not (0 <= down_payment_pct <= 1):
        raise ValueError("down_payment_pct must be between 0 and 1")

    if not (0 <= interest_rate <= 0.30):
        raise ValueError("interest_rate must be between 0 and 0.30")

    if loan_term_years <= 0:
        raise ValueError("loan_term_years must be positive")

    loan_amount = price - (price * down_payment_pct)
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12

    if monthly_rate == 0:
        monthly_payment = loan_amount / num_payments
    else:
        monthly_payment = loan_amount * (
            monthly_rate * (1 + monthly_rate) ** num_payments
        ) / (
            (1 + monthly_rate) ** num_payments - 1
        )

    total_paid = monthly_payment * num_payments
    total_interest_paid = total_paid - loan_amount

    return {
        "monthly_payment": round(monthly_payment, 2),
        "loan_amount": round(loan_amount, 2),
        "total_interest_paid": round(total_interest_paid, 2),
    }


def get_comps(listing_id: str, target_listing_metadata: dict) -> dict:
    """Find comparable recently-sold properties for a given active listing, 
    to help judge whether its asking price is reasonable relative to the market.

    Args:
        listing_id: The target listing's ID, used only for error messages.
        target_listing_metadata: Dict with the target's city, bedrooms, 
            and square_footage.

    Returns:
        A dict with comps (list of matching sold records) and 
        average_comp_price, or an empty comps list with a message if 
        none are found.
    """
    import json
    import os

    sold_comps_path = os.path.join(os.path.dirname(__file__), "..", "data", "sold_comps.json")
    with open(sold_comps_path) as f:
        sold_comps = json.load(f)

    if target_listing_metadata is None:
        return {"comps": [], "message": f"No listing found with id {listing_id}"}

    target_city = target_listing_metadata.get("city")
    target_bedrooms = target_listing_metadata.get("bedrooms")
    target_sqft = target_listing_metadata.get("square_footage")

    if target_city is None or target_bedrooms is None or target_sqft is None:
        return {"comps": [], "message": f"Listing {listing_id} is missing required fields (city, bedrooms, or square_footage)"}

    sqft_min = target_sqft * 0.85
    sqft_max = target_sqft * 1.15

    matches = [
        comp for comp in sold_comps
        if comp["city"] == target_city
        and sqft_min <= comp["square_footage"] <= sqft_max
        and abs(comp["bedrooms"] - target_bedrooms) <= 1
    ]

    if not matches:
        return {"comps": [], "message": f"No comparable sold listings found for {listing_id}"}

    average_comp_price = sum(comp["sold_price"] for comp in matches) / len(matches)

    return {
        "comps": matches,
        "average_comp_price": round(average_comp_price, 2),
    }


def reject_listing(buyer_id: str, listing_id: str, reason: str) -> dict:
    """Record that a buyer has rejected a specific listing, so it will not 
    be suggested to them again in future searches. Persists to buyer_profiles.json.

    Args:
        buyer_id: The buyer's ID, e.g. "B001".
        listing_id: The listing's ID being rejected, e.g. "L_ELM_124".
        reason: A short explanation of why the buyer rejected it.

    Returns:
        A dict confirming the rejection was recorded, or an error message.
    """
    import json
    import os
    from datetime import datetime, timezone

    profiles_path = os.path.join(os.path.dirname(__file__), "..", "data", "buyer_profiles.json")

    with open(profiles_path) as f:
        buyer_profiles = json.load(f)

    buyer = next((b for b in buyer_profiles if b["buyer_id"] == buyer_id), None)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    already_rejected = any(
        r["listing_id"] == listing_id
        for r in buyer["session_history"]["rejected_listings"]
    )
    if already_rejected:
        return {"success": True, "message": f"{listing_id} was already rejected for {buyer_id}"}

    buyer["session_history"]["rejected_listings"].append({
        "listing_id": listing_id,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    with open(profiles_path, "w") as f:
        json.dump(buyer_profiles, f, indent=2)

    return {"success": True, "message": f"Recorded rejection of {listing_id} for {buyer_id}"}


def reset_rejected_listings(buyer_id: str) -> dict:
    """Clear a buyer's entire rejection history, so previously rejected 
    listings can be suggested again. Use only when the buyer explicitly 
    asks to start over or reconsider past rejections.

    Args:
        buyer_id: The buyer's ID, e.g. "B001".

    Returns:
        A dict confirming the reset, or an error message.
    """
    import json
    import os

    profiles_path = os.path.join(os.path.dirname(__file__), "..", "data", "buyer_profiles.json")

    with open(profiles_path) as f:
        buyer_profiles = json.load(f)

    buyer = next((b for b in buyer_profiles if b["buyer_id"] == buyer_id), None)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    count = len(buyer["session_history"]["rejected_listings"])
    buyer["session_history"]["rejected_listings"] = []

    with open(profiles_path, "w") as f:
        json.dump(buyer_profiles, f, indent=2)

    return {"success": True, "message": f"Cleared {count} rejected listing(s) for {buyer_id}"}


def undo_last_rejection(buyer_id: str) -> dict:
    """Remove only the most recently rejected listing, so it can be 
    suggested again. Use when the buyer says something like 'actually, 
    show me that last one again' or 'I changed my mind.'

    Args:
        buyer_id: The buyer's ID, e.g. "B001".

    Returns:
        A dict confirming which listing was restored, or an error message.
    """
    import json
    import os

    profiles_path = os.path.join(os.path.dirname(__file__), "..", "data", "buyer_profiles.json")

    with open(profiles_path) as f:
        buyer_profiles = json.load(f)

    buyer = next((b for b in buyer_profiles if b["buyer_id"] == buyer_id), None)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    rejected = buyer["session_history"]["rejected_listings"]
    if not rejected:
        return {"success": True, "message": f"{buyer_id} has no rejections to undo"}

    removed = rejected.pop()  # remove the most recently appended entry

    with open(profiles_path, "w") as f:
        json.dump(buyer_profiles, f, indent=2)

    return {"success": True, "message": f"Restored {removed['listing_id']} (removed from rejection list)"}

if __name__ == "__main__":
    result = calc_mortgage(price=425000, down_payment_pct=0.20, interest_rate=0.065)
    print(result)

    import json
    with open("../data/sold_comps.json") as f:
        sold_comps = json.load(f)
    pine_101_metadata = {"city": "Austin", "bedrooms": 3, "square_footage": 1850}
    print(get_comps("L_PINE_101", pine_101_metadata))

    print("\n--- Adding test rejections for B001 ---")
    print(reject_listing("B001", "L_PINE_101", "Test rejection 1 - too far from downtown"))
    print(reject_listing("B001", "L_SPRUCE_106", "Test rejection 2 - needs too much work"))

    print("\n--- Testing undo_last_rejection ---")
    print(undo_last_rejection("B001"))

    print("\n--- Testing reset_rejected_listings ---")
    print(reset_rejected_listings("B001"))