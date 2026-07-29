# dimensions_config.py
# Single source of truth for fuzzy dimensions - imported by both
# ingest.py (enrichment at ingestion time) and search.py (query-time
# classification), so the two can never drift out of sync.

FUZZY_DIMENSIONS = {
    "school_quality": {
        "keywords": ["school"],
        "canonical_phrases": ["good school district", "top-tier schools", "excellent schools"],
    },
    "condition": {
        "keywords": ["renovat", "cosmetic", "as-is", "fixer", "move-in", "diamond in the rough"],
        "canonical_phrases": ["move-in ready", "good condition", "no renovations needed"],
    },
}