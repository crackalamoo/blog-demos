"""The three questions asked of every in-scope Wikipedia edit."""

from __future__ import annotations

PREAMBLE = (
    "The state is a single edit to an English Wikipedia article: the "
    "article title, the editor's name, the edit summary they wrote, the "
    "change in article size in bytes, and the wikitext that the edit "
    "added and removed. Judge the edit itself, from the added and "
    "removed text. The edit summary is the editor's own claim about the "
    "edit and may be wrong, empty, or misleading."
)

ACTION_LABELS = [
    "vandalism",
    "reverting damage",
    "adding content",
    "citation work",
    "copyedit",
    "metadata",
    "other",
]

ACTION_INSTRUCTIONS = (
    f"{PREAMBLE}\n\n"
    "What is this edit doing? Choose the single label that best "
    "describes the edit's dominant effect on the article."
)

ACTION_CRITERIA = {
    "vandalism": (
        "The edit damages the article on purpose: obscenity, insults, "
        "nonsense, blanking without reason, joke claims, or replacing "
        "real content with junk."
    ),
    "reverting damage": (
        "The edit undoes someone else's damage or unwanted change, "
        "restoring a previous version of the text."
    ),
    "adding content": (
        "The edit adds substantive new prose or facts to the article "
        "beyond what was there."
    ),
    "citation work": (
        "The edit is mainly about sources: adding, repairing, "
        "reformatting, or removing references and citation templates."
    ),
    "copyedit": (
        "The edit changes wording, grammar, spelling, punctuation, or "
        "phrasing without changing what the article claims."
    ),
    "metadata": (
        "The edit works on the article's apparatus rather than its "
        "prose: categories, infobox fields, links, templates, "
        "short descriptions, or formatting."
    ),
    "other": (
        "The edit fits none of the above labels."
    ),
}

MISLEADING_INSTRUCTIONS = (
    f"{PREAMBLE}\n\n"
    "If this edit stood and a reader read the article afterwards, how "
    "misleading would the article be to them because of this edit? Rate "
    "the effect on the reader, not the editor's intent, and not whether "
    "the edit was an improvement in style."
)

MISLEADING_RUNGS = [
    "no effect on the reader",
    "cosmetic only",
    "slightly misleading",
    "materially wrong",
    "outright false",
]

MISLEADING_CRITERIA = [
    "No effect on the reader. What the article tells a reader is "
    "unchanged, or the edit repairs the article.",
    "Cosmetic only. Wording, formatting, or apparatus changed; every "
    "claim the reader takes away is the same as before.",
    "Slightly misleading. A reader comes away with a shaded or "
    "imprecise impression: emphasis, hedging, or a detail lost.",
    "Materially wrong. A reader comes away believing something the "
    "sources do not support: a wrong figure, date, name, or claim.",
    "Outright false. A reader is told something flatly untrue, or the "
    "article's substance is destroyed.",
]

SUMMARY_INSTRUCTIONS = (
    f"{PREAMBLE}\n\n"
    "Does the edit summary honestly describe what the edit actually "
    "changed? Compare the summary against the added and removed text. "
    "An empty summary is not dishonest; a summary that describes a "
    "different change than the one made is."
)

SUMMARY_CRITERIA = {
    "true": (
        "The summary is an honest account of the edit: it describes the "
        "change that was actually made, or it is empty, or it is a "
        "section marker with nothing claimed."
    ),
    "false": (
        "The summary misrepresents the edit: it claims a change that "
        "was not made, understates or disguises what was changed, or "
        "describes the edit as routine when it is not."
    ),
}
