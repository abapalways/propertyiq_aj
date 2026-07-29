# test_nli_wording.py
from sentence_transformers import CrossEncoder

nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
NLI_LABELS = ["contradiction", "entailment", "neutral"]

needs_work_text = ("A true diamond in the rough. This classic property has solid bones "
                    "but requires significant cosmetic updates, a kitchen overhaul, and "
                    "immediate roof repairs. Sold entirely as-is. The property is in a needs-work condition.")

move_in_ready_text = ("Beautiful, move-in-ready single-family home featuring a spacious "
                       "open-concept living area, updated kitchen with stainless steel "
                       "appliances, and a private, fenced backyard. The home is in excellent, "
                       "move-in-ready condition.")

hypotheses = {
    "current template":      "This home satisfies the requirement: needs renovation.",
    "rephrased positive":    "This home needs renovation.",
    "rephrased alt":         "This home requires significant repair work.",
    "explicit contrast":     "This home is NOT move-in ready and requires renovation.",
}

for label, hyp in hypotheses.items():
    for text_label, text in [("needs_work_listing", needs_work_text), ("move_in_ready_listing", move_in_ready_text)]:
        scores = nli_model.predict([(text, hyp)])
        result = NLI_LABELS[scores.argmax()]
        print(f"[{label}] vs {text_label}: {result}  (scores={scores[0].round(3)})")