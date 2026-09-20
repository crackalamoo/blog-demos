"""The ten judgments, one per request."""

from __future__ import annotations

from typing import Any, Dict, List

from typesafe_sdk import Choice, Noul, Score

PREAMBLE = (
    "The state is the extracted text of a company's landing page, exactly "
    "as the page presents it. Judge only what this page says."
)


def _noul(key: str, label: str, instructions: str, true: str,
          false: str) -> Dict[str, Any]:
    return {
        "key": key,
        "kind": "noul",
        "label": label,
        "question": lambda: Noul(
            instructions=f"{PREAMBLE}\n\n{instructions}",
            criteria={"true": true, "false": false},
        ),
    }


def _score(key: str, label: str, instructions: str,
           rungs: List[str], criteria: List[str]) -> Dict[str, Any]:
    return {
        "key": key,
        "kind": "score",
        "label": label,
        "rungs": rungs,
        "question": lambda: Score(
            instructions=f"{PREAMBLE}\n\n{instructions}",
            criteria=criteria,
        ),
    }


def _choice(key: str, label: str, instructions: str,
            criteria: Dict[str, str]) -> Dict[str, Any]:
    return {
        "key": key,
        "kind": "choice",
        "label": label,
        "options": list(criteria),
        "question": lambda: Choice(
            instructions=f"{PREAMBLE}\n\n{instructions}",
            criteria=criteria,
        ),
    }


QUESTIONS: List[Dict[str, Any]] = [
    _noul(
        "usable_today",
        "Can you use this today?",
        "Can a reader start using this product today, or is the only "
        "thing on offer a waitlist, an invite request, or a demo "
        "booking?",
        "The product can be used, bought, downloaded, or signed up for "
        "today.",
        "There is no way in; the page offers only a waitlist, an early "
        "access request, a sales conversation, or nothing at all.",
    ),
    _noul(
        "has_price",
        "Is there a price on the page?",
        "Is an actual price stated anywhere on this page?",
        "The page states a price: a currency figure, a per-unit rate, or "
        "an explicit statement that it is free.",
        "No price appears. Pointers to sales, custom quotes, or vague "
        "affordability claims do not count.",
    ),
    _noul(
        "names_customer",
        "Does it name a real customer?",
        "Does the page name a specific, identifiable customer or user of "
        "the product?",
        "At least one customer or user is named specifically enough to "
        "be looked up: a company name, or a named person with their "
        "affiliation.",
        "Customers are referred to anonymously, by first name only, by "
        "category, or not at all.",
    ),
    _noul(
        "says_what_it_does",
        "Does it say what it does?",
        "Does the page ever state what the product actually does, "
        "concretely enough that a reader could describe it to someone "
        "else?",
        "The page plainly states the product's function.",
        "The page describes only benefits, values, or ambitions; a "
        "reader finishes it unable to say what the thing does.",
    ),
    _score(
        "readiness",
        "How ready is it, really?",
        "How far along is the thing this page is selling, judging by "
        "what the page itself reveals rather than what it claims?",
        ["idea", "prototype", "private beta", "shipping", "mature"],
        [
            "An idea. Nothing exists yet that anyone outside could use.",
            "A prototype. Something runs, but only as a demonstration.",
            "A private beta. Real users, but by invitation and with "
            "known gaps.",
            "Shipping. Generally available and usable by anyone who "
            "wants it.",
            "Mature. Long in the field, with scale, versions, or a "
            "track record behind it.",
        ],
    ),
    _score(
        "shipped_vs_roadmap",
        "Shipped, or roadmap?",
        "Of what this page describes, how much exists now versus how "
        "much is promised for later?",
        ["all roadmap", "mostly roadmap", "half", "mostly shipped",
         "all shipped"],
        [
            "Everything described is future. Nothing on this page "
            "exists yet.",
            "Mostly future. A little exists; the page is about what is "
            "coming.",
            "An even split between what is built and what is promised.",
            "Mostly built. A few clearly-marked things are still ahead.",
            "Everything described already exists. Nothing is promised "
            "for later.",
        ],
    ),
    _score(
        "concreteness",
        "How concrete are the claims?",
        "How concrete is the language of this page?",
        ["adjectives", "vague", "mixed", "specific", "numbers"],
        [
            "Adjectives with no nouns. Nothing here could be checked.",
            "Vague. General descriptions, no particulars.",
            "Mixed. Some particulars among the adjectives.",
            "Specific. Named features, named things, checkable claims.",
            "Numbers. Prices, measurements, versions, quantities "
            "throughout.",
        ],
    ),
    _choice(
        "category",
        "What is this, actually?",
        "What kind of thing is this page selling?",
        {
            "dev tool": "Software sold to engineers to build with: an "
                        "API, library, platform, or developer service.",
            "consumer app": "An application sold to individuals for "
                            "their own use.",
            "marketplace": "A place where two sides meet to transact.",
            "consultancy": "Human services: advice, implementation, or "
                           "staffing, sold as engagements.",
            "research project": "Research output rather than a product: "
                                "a lab, a paper, a prototype.",
            "other": "None of the above fits this page.",
        },
    ),
    _choice(
        "omission",
        "What is most conspicuously missing?",
        "Reading this page, what is the most conspicuous thing it does "
        "not say?",
        {
            "price": "What it costs is nowhere on the page.",
            "what it does": "The page never says what the product "
                            "actually is or does.",
            "who built it": "No company, team, or person stands behind "
                            "the page.",
            "proof it works": "Nothing on the page supports its claims: "
                              "no customers, numbers, or evidence.",
            "nothing - the page is complete": "Nothing conspicuous is "
                "missing. The page says what it is, what it costs, who "
                "built it, and why to believe it.",
        },
    ),
    _score(
        "coyness",
        "How coy is the page?",
        "How much is this page working to avoid saying something a "
        "reader plainly wants to know?",
        ["candid", "reticent", "evasive", "coy", "hiding"],
        [
            "Candid. It volunteers the awkward parts.",
            "Reticent. It answers what is asked, no more.",
            "Evasive. Obvious questions go around, not through.",
            "Coy. The page clearly knows what it is not saying.",
            "Hiding. The whole page is built around an omission.",
        ],
    ),
]

BY_KEY = {q["key"]: q for q in QUESTIONS}
