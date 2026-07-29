# scripts/test_broad_enrichment.py
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from dotenv import load_dotenv
load_dotenv("../.env")

from ingest import enrich_listing_sentiment

maple_text = """# Property ID: L_MAPLE_105
- **School District:** Located within the Northside School District (currently rated 4/10 on state performance indexes).
- **Value Focus:** Perfect for buyers prioritizing raw space, square footage, and home offices over school performance indicators."""
print("--- Maple ---")
print(enrich_listing_sentiment(maple_text))

spruce_text = """# Property ID: L_SPRUCE_106
### Description
A true diamond in the rough. This classic property has solid bones but requires significant cosmetic updates, a kitchen overhaul, and immediate roof repairs. Sold entirely as-is."""
print("\n--- Spruce ---")
print(enrich_listing_sentiment(spruce_text))

# %%
if __name__ == "__main__":
    maple_text = """# Property ID: L_MAPLE_105
...
- **School District:** Located within the Northside School District (currently rated 4/10 on state performance indexes).
- **Value Focus:** Perfect for buyers prioritizing raw space, square footage, and home offices over school performance indicators."""
    print(enrich_listing_sentiment(maple_text))

    spruce_text = """# Property ID: L_SPRUCE_106
### Description
A true diamond in the rough. This classic property has solid bones but requires significant cosmetic updates, a kitchen overhaul, and immediate roof repairs. Sold entirely as-is."""
    print(enrich_listing_sentiment(spruce_text))