"""
Custom fair-housing guardrail, built with the guardrails-ai framework.
Uses an LLM to classify whether a query attempts protected-characteristic 
filtering - more robust to paraphrasing/proxies than regex.
"""
from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult
from guardrails.hub import DetectPII
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

    IMPORTANT: Asking about school QUALITY, ratings, rankings, or academic 
    performance (e.g. "good schools," "top-rated schools," "top-tier schools," 
    "excellent school district") is LEGITIMATE and must be marked OK. This is 
    about educational quality, NOT about who attends the school. Only mark 
    VIOLATION if the request explicitly or implicitly references the demographic, 
    racial, religious, or national-origin composition of a school, neighborhood, 
    or population - not academic quality or rankings.

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
    
@register_validator(name="on-topic-check", data_type="string")
class OnTopicCheck(Validator):
    def validate(self, value, metadata):
        prompt = f"""You are a scope checker for a real estate property-search 
assistant. This assistant helps buyers search listings, calculate mortgages, 
find comparable sales, and manage their preferences/rejections.

Determine if the following message is relevant to real estate / home buying 
in any way (including casual follow-ups like "tell me more" or "thanks").

Answer with exactly one word on the first line: RELEVANT or OFF_TOPIC.

Message: "{value}"

Answer:"""
        response = _check_llm.invoke(prompt).content.strip()
        first_line = response.split("\n")[0].upper()

        if "OFF_TOPIC" in first_line:
            return FailResult(
                error_message=(
                    "I'm a real estate search assistant, so I can only help with "
                    "property search, mortgage calculations, comps, and related "
                    "questions. Let me know if you'd like help with any of those!"
                )
            )
        return PassResult()

from guardrails.hub import DetectPII


def build_pii_guard():
    """PII guard using Guardrails Hub's pre-built DetectPII validator, 
    not custom - detects emails, phone numbers, SSNs, etc. in buyer input."""
    return Guard().use(
        DetectPII(
            pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD"],
            on_fail="fix",  # redacts PII rather than hard-blocking the message
        )
    )
if __name__ == "__main__":
    pii_guard = build_pii_guard()

    test_inputs = [
        "My social security number is 123-45-6789",
        "SSN: 123-45-6789",
        "123-45-6789",
    ]
    for text in test_inputs:
        result = pii_guard.validate(text)
        print(f"Original: {text!r}")
        print(f"Redacted: {result.validated_output!r}")
        print()