"""Map free-form model output to yes / no / unparsed.

Strict mode accepts only a leading yes/no (what we asked for). Lenient mode also
accepts a yes/no anywhere in the first sentence; the gap between the two is the
"format compliance" number we report.
"""

import math
import re

_LEAD = re.compile(r"^[\W_]*(yes|no)\b", re.IGNORECASE)
_ANY = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> tuple[str, str]:
    """Return (answer, mode) with answer in {"yes", "no", "unparsed"}."""
    t = text.strip()
    m = _LEAD.match(t)
    if m:
        return m.group(1).lower(), "strict"
    first_sentence = re.split(r"[.\n]", t, maxsplit=1)[0]
    hits = {h.lower() for h in _ANY.findall(first_sentence)}
    if len(hits) == 1:
        return hits.pop(), "lenient"
    return "unparsed", "none"


def p_yes_from_top_logprobs(top: dict[str, float]) -> float | None:
    """P(yes) renormalised over yes/no variants of the first generated token.

    `top` maps token string -> logprob. Returns None if neither appears.
    """
    yes = no = 0.0
    for tok, lp in top.items():
        w = tok.strip().lower()
        if w == "yes":
            yes += math.exp(lp)
        elif w == "no":
            no += math.exp(lp)
    if yes + no == 0:
        return None
    return yes / (yes + no)
