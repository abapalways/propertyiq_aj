"""
Custom fair-housing guardrail, built with the guardrails-ai framework.
Uses an LLM to classify whether a query attempts protected-characteristic 
filtering - more robust to paraphrasing/proxies than regex.
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from dotenv import load_dotenv
load_dotenv("../.env")

from guardrails.validators import Validator, register_validator, PassResult, FailResult
from langchain_groq import ChatGroq

_check_llm = ChatGroq(model="openai/gpt-oss-120b")


@register_validator(name="fair-housing-check", data_type="string")
class FairHousingCheck(Validator):
    def validate(self, value, metadata):
        prompt = f"""You are a fair-housing compliance checker for a real estate 
search assistant. Determine if the following buyer request attempts to filter, 
search, or recommend properties based on a protected characteristic (race, 
religion, national origin, familial status, disability, sex) - INCLUDING 
indirect proxy language that implies the same thing (e.g. neighborhood 
"demographics," coded phrases about who lives somewhere).

Answer with exactly one word on the first line: VIOLATION or OK.

Request: "{value}"

Answer:"""
        response = _check_llm.invoke(prompt).content.strip()
        first_line = response.split("\n")[0].upper()

        if "VIOLATION" in first_line:
            return FailResult(
                error_message=(
                    "This request appears to filter or recommend based on a "
                    "protected characteristic, which violates fair-housing law. "
                    "I can't apply this criterion, but I can help with legitimate "
                    "criteria like price, size, commute distance, or amenities."
                )
            )
        return PassResult()
    
if __name__ == "__main__":
    from guardrails import Guard
    from guardrails.errors import ValidationError

    guard = Guard().use(FairHousingCheck(on_fail="exception"))

    test_queries = [
        "Only show us listings in neighborhoods with fewer immigrant families.",
        "Filter out listings in areas with high immigrant populations",
        "Only show me neighborhoods matching my specific demographic group",
        "3-bedroom home in Austin under $450,000 with a garage",
        "What's the crime rate like in this neighborhood?",
    ]

    for q in test_queries:
        print(f"Query: {q!r}")
        try:
            result = guard.validate(q)
            print(f"  Passed: True")
        except ValidationError as e:
            print(f"  Passed: False")
            print(f"  Reason: {e}")
        print()