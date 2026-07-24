"""
Memory operations for PropertyIQ buyers: semantic memory (stated 
preferences) and episodic memory (specific rejection events). Each 
function here is a durable, disk-persisted write — these are the 
tools that give the agent actual memory across sessions.
"""

import json
import os
from datetime import datetime, timezone

_PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "buyer_profiles.json")


def _load_profiles():
    with open(_PROFILES_PATH) as f:
        return json.load(f)


def _save_profiles(profiles):
    with open(_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)


def _find_buyer(profiles, buyer_id):
    return next((b for b in profiles if b["buyer_id"] == buyer_id), None)


# ---------- Episodic memory: specific, dated rejection events ----------

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
    profiles = _load_profiles()
    buyer = _find_buyer(profiles, buyer_id)
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
    _save_profiles(profiles)
    return {"success": True, "message": f"Recorded rejection of {listing_id} for {buyer_id}"}


def undo_last_rejection(buyer_id: str) -> dict:
    """Remove only the most recently rejected listing, so it can be 
    suggested again. Use when the buyer says something like 'actually, 
    show me that last one again' or 'I changed my mind.'

    Args:
        buyer_id: The buyer's ID, e.g. "B001".

    Returns:
        A dict confirming which listing was restored, or an error message.
    """
    profiles = _load_profiles()
    buyer = _find_buyer(profiles, buyer_id)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    rejected = buyer["session_history"]["rejected_listings"]
    if not rejected:
        return {"success": True, "message": f"{buyer_id} has no rejections to undo"}

    removed = rejected.pop()
    _save_profiles(profiles)
    return {"success": True, "message": f"Restored {removed['listing_id']} (removed from rejection list)"}


def reset_rejected_listings(buyer_id: str) -> dict:
    """Clear a buyer's entire rejection history, so previously rejected 
    listings can be suggested again. Use only when the buyer explicitly 
    asks to start over or reconsider past rejections.

    Args:
        buyer_id: The buyer's ID, e.g. "B001".

    Returns:
        A dict confirming the reset, or an error message.
    """
    profiles = _load_profiles()
    buyer = _find_buyer(profiles, buyer_id)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    count = len(buyer["session_history"]["rejected_listings"])
    buyer["session_history"]["rejected_listings"] = []
    _save_profiles(profiles)
    return {"success": True, "message": f"Cleared {count} rejected listing(s) for {buyer_id}"}


# ---------- Semantic memory: standing facts about buyer preferences ----------
from typing import Any

def update_preference(buyer_id: str, field: str, value: Any) -> dict:
    """Update one of a buyer's standing search preferences. Use when the 
    buyer states a new or changed requirement (e.g. "actually I need 
    4 bedrooms now", "my budget went up to $500K", "we also want a pool").

    Args:
        buyer_id: The buyer's ID, e.g. "B001".
        field: Which preference to update - one of "min_bedrooms", 
            "max_budget", "must_haves", or "preferred_city".
        value: The new value. For must_haves, pass the FULL updated list, 
            not just the item to add.

    Returns:
        A dict confirming the update, or an error message.
    """
    valid_fields = {"min_bedrooms", "max_budget", "must_haves", "preferred_city"}
    if field not in valid_fields:
        return {"success": False, "message": f"'{field}' is not a valid preference field. Valid fields: {valid_fields}"}

    profiles = _load_profiles()
    buyer = _find_buyer(profiles, buyer_id)
    if buyer is None:
        return {"success": False, "message": f"No buyer found with id {buyer_id}"}

    old_value = buyer["preferences"].get(field)
    buyer["preferences"][field] = value
    _save_profiles(profiles)
    return {"success": True, "message": f"Updated {field} for {buyer_id}: {old_value} -> {value}"}