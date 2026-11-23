"""Make inconsistent medicine names line up.

"Paracetamol 500mg Tabs", "paracetamol 500 mg tablets" and "PARACETAMOL 500MG TAB."
all become one product. Strengths must match exactly, so "Amoxicillin 250mg" and
"Amoxicillin 500mg" are never merged even though the names look alike.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

import pandas as pd

_FORMS = [
    (r"\b(tabs?|tablets?|tbl?s?)\b", "tablets"),
    (r"\b(caps?|capsules?)\b", "capsules"),
    (r"\b(syr|syrups?)\b", "syrup"),
    (r"\b(susp|suspensions?)\b", "suspension"),
    (r"\b(inj|injections?|injectables?)\b", "injection"),
    (r"\b(oint|ointments?)\b", "ointment"),
    (r"\b(soln|solutions?)\b", "solution"),
    (r"\b(supp|suppositor(y|ies))\b", "suppository"),
    (r"\b(gtts|drops?)\b", "drops"),
]
_UNITS = r"(mg|mcg|µg|ug|g|kg|ml|l|iu|%)"
# Common shorthand on pharmacy tills. Spelling only: brands are never merged
# with generics, because they are different products with different prices.
ABBREVIATIONS = {
    "vit": "vitamin", "pcm": "paracetamol", "para": "paracetamol", "amox": "amoxicillin",
    "cipro": "ciprofloxacin", "metro": "metronidazole", "ors": "oral rehydration salts",
    "hctz": "hydrochlorothiazide", "fesso4": "ferrous sulphate", "feso4": "ferrous sulphate",
    "sulfate": "sulphate", "multivit": "multivitamin", "amlo": "amlodipine",
}
MERGE_THRESHOLD = 0.92


def normalise_name(name: object) -> str:
    text = unicodedata.normalize("NFKC", str(name)).lower().strip()
    text = text.replace("&", " and ")
    text = re.sub(rf"(\d+(?:\.\d+)?)\s*{_UNITS}\b", r"\1\2", text)
    text = text.replace("µg", "mcg").replace("ug", "mcg")
    text = re.sub(r"[^a-z0-9%./+ ]+", " ", text)
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)
    for pattern, replacement in _FORMS:
        text = re.sub(pattern, replacement, text)
    words = [ABBREVIATIONS.get(w, w) for w in text.split()]
    return " ".join(" ".join(words).split())


def _numbers(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"\d+(?:\.\d+)?", text))


def group_names(names: pd.Series, aliases: dict[str, str] | None = None) -> tuple[dict[str, str], list[dict]]:
    """Return (raw name -> canonical key) and a list of merges for the user to review.

    The canonical key is the most frequent normalised spelling in each group.
    ``aliases`` maps a name as written to the name it should count as
    (for example a brand to its generic, when the pharmacy wants that).
    """
    counts = Counter(names.dropna().astype(str))
    alias_norm = {normalise_name(k): normalise_name(v) for k, v in (aliases or {}).items()}
    normalised: dict[str, str] = {}
    for raw in counts:
        norm = normalise_name(raw)
        normalised[raw] = alias_norm.get(norm, norm)
    norm_counts: Counter[str] = Counter()
    for raw, n in counts.items():
        norm_counts[normalised[raw]] += n

    # Fuzzy-merge normalised names, most frequent first so they become the anchor.
    anchors: list[str] = []
    norm_to_anchor: dict[str, str] = {}
    for norm, _ in norm_counts.most_common():
        target = None
        for anchor in anchors:
            if _numbers(anchor) != _numbers(norm):
                continue
            if abs(len(anchor) - len(norm)) > 6:
                continue
            if SequenceMatcher(None, anchor, norm).ratio() >= MERGE_THRESHOLD:
                target = anchor
                break
        if target is None:
            anchors.append(norm)
            target = norm
        norm_to_anchor[norm] = target

    raw_to_key = {raw: norm_to_anchor[normalised[raw]] for raw in counts}

    merges: list[dict] = []
    groups: dict[str, list[str]] = {}
    for raw, key in raw_to_key.items():
        groups.setdefault(key, []).append(raw)
    for key, variants in groups.items():
        if len(variants) > 1:
            variants.sort(key=lambda v: -counts[v])
            merges.append({
                "key": key,
                "kept_as": variants[0],
                "variants": [{"name": v, "rows": counts[v]} for v in variants],
                "fuzzy": any(norm_to_anchor[normalised[v]] != normalised[v] for v in variants),
            })
    merges.sort(key=lambda m: -sum(v["rows"] for v in m["variants"]))
    return raw_to_key, merges


def display_names(raw: pd.Series, keys: pd.Series) -> dict[str, str]:
    """Most frequent original spelling for each key, for showing to people."""
    frame = pd.DataFrame({"raw": raw.astype(str), "key": keys})
    return frame.groupby("key")["raw"].agg(lambda s: s.value_counts().index[0]).to_dict()


def possible_duplicates(keys: list[str], limit: int = 20) -> list[dict]:
    """Pairs that look alike but were NOT merged (for example a brand and a generic,
    or a spelling the rules did not catch). Shown to the user, never merged automatically."""
    pairs = []
    unique = sorted(set(keys))
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            if _numbers(a) != _numbers(b) or not _numbers(a):
                continue
            score = SequenceMatcher(None, a, b).ratio()
            words_a, words_b = set(a.split()), set(b.split())
            shared = len(words_a & words_b) / max(1, min(len(words_a), len(words_b)))
            if score >= 0.7 or shared >= 0.6:
                pairs.append({"a": a, "b": b, "similarity": round(max(score, shared), 2)})
    pairs.sort(key=lambda p: -p["similarity"])
    return pairs[:limit]
