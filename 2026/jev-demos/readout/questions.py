from __future__ import annotations

from typesafe_sdk import (
    Choice, Noul, Score, SystemOneResponse, TypeSafeClient)


def read(client: TypeSafeClient, page: str) -> SystemOneResponse:
    return client.system_one(
        state={
            "about": (
                "The extracted text of a company's landing page, exactly "
                "as the page presents it. Judge only what this page says."),
            "landing_page": page,
        },
        questions={
            "usable_today": Noul(
                instructions=(
                    "Can a reader start using this product today, or is the "
                    "only thing on offer a waitlist, an invite request, or a "
                    "demo booking?"),
                criteria={
                    "true": (
                        "The product can be used, bought, downloaded, or "
                        "signed up for today."),
                    "false": (
                        "There is no way in; the page offers only a waitlist, "
                        "an early access request, a sales conversation, or "
                        "nothing at all."),
                },
            ),
            "has_price": Noul(
                instructions=(
                    "Is an actual price stated anywhere on this page?"),
                criteria={
                    "true": (
                        "The page states a price: a currency figure, a "
                        "per-unit rate, or an explicit statement that it is "
                        "free."),
                    "false": (
                        "No price appears. Pointers to sales, custom quotes, "
                        "or vague affordability claims do not count."),
                },
            ),
            "names_customer": Noul(
                instructions=(
                    "Does the page name a specific, identifiable customer or "
                    "user of the product?"),
                criteria={
                    "true": (
                        "At least one customer or user is named specifically "
                        "enough to be looked up: a company name, or a named "
                        "person with their affiliation."),
                    "false": (
                        "Customers are referred to anonymously, by first name "
                        "only, by category, or not at all."),
                },
            ),
            "says_what_it_does": Noul(
                instructions=(
                    "Does the page ever state what the product actually does, "
                    "concretely enough that a reader could describe it to "
                    "someone else?"),
                criteria={
                    "true": (
                        "The page plainly states the product's function."),
                    "false": (
                        "The page describes only benefits, values, or "
                        "ambitions; a reader finishes it unable to say what "
                        "the thing does."),
                },
            ),
            "readiness": Score(
                instructions=(
                    "How far along is the thing this page is selling, judging "
                    "by what the page itself reveals rather than what it "
                    "claims?"),
                criteria=[
                    "An idea. Nothing exists yet that anyone outside could "
                    "use.",
                    "A prototype. Something runs, but only as a "
                    "demonstration.",
                    "A private beta. Real users, but by invitation and with "
                    "known gaps.",
                    "Shipping. Generally available and usable by anyone who "
                    "wants it.",
                    "Mature. Long in the field, with scale, versions, or a "
                    "track record behind it.",
                ],
            ),
            "shipped_vs_roadmap": Score(
                instructions=(
                    "Of what this page describes, how much exists now versus "
                    "how much is promised for later?"),
                criteria=[
                    "Everything described is future. Nothing on this page "
                    "exists yet.",
                    "Mostly future. A little exists; the page is about what "
                    "is coming.",
                    "An even split between what is built and what is "
                    "promised.",
                    "Mostly built. A few clearly-marked things are still "
                    "ahead.",
                    "Everything described already exists. Nothing is promised "
                    "for later.",
                ],
            ),
            "concreteness": Score(
                instructions=(
                    "How concrete is the language of this page?"),
                criteria=[
                    "Adjectives with no nouns. Nothing here could be checked.",
                    "Vague. General descriptions, no particulars.",
                    "Mixed. Some particulars among the adjectives.",
                    "Specific. Named features, named things, checkable "
                    "claims.",
                    "Numbers. Prices, measurements, versions, quantities "
                    "throughout.",
                ],
            ),
            "category": Choice(
                instructions=(
                    "What kind of thing is this page selling?"),
                criteria={
                    "dev tool": (
                        "Software sold to engineers to build with: an API, "
                        "library, platform, or developer service."),
                    "consumer app": (
                        "An application sold to individuals for their own "
                        "use."),
                    "marketplace": (
                        "A place where two sides meet to transact."),
                    "consultancy": (
                        "Human services: advice, implementation, or staffing, "
                        "sold as engagements."),
                    "research project": (
                        "Research output rather than a product: a lab, a "
                        "paper, a prototype."),
                    "other": (
                        "None of the above fits this page."),
                },
            ),
            "omission": Choice(
                instructions=(
                    "Reading this page, what is the most conspicuous thing it "
                    "does not say?"),
                criteria={
                    "price": (
                        "What it costs is nowhere on the page."),
                    "what it does": (
                        "The page never says what the product actually is or "
                        "does."),
                    "who built it": (
                        "No company, team, or person stands behind the page."),
                    "proof it works": (
                        "Nothing on the page supports its claims: no "
                        "customers, numbers, or evidence."),
                    "nothing - the page is complete": (
                        "Nothing conspicuous is missing. The page says what "
                        "it is, what it costs, who built it, and why to "
                        "believe it."),
                },
            ),
            "coyness": Score(
                instructions=(
                    "How much is this page working to avoid saying something "
                    "a reader plainly wants to know?"),
                criteria=[
                    "Candid. It volunteers the awkward parts.",
                    "Reticent. It answers what is asked, no more.",
                    "Evasive. Obvious questions go around, not through.",
                    "Coy. The page clearly knows what it is not saying.",
                    "Hiding. The whole page is built around an omission.",
                ],
            ),
        },
        model="jev-latest",
    )
