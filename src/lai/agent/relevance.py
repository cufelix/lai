"""Which of many things are worth putting in front of the model.

A prompt that describes everything the agent could possibly do costs the same
on every turn of every run, and most of it is irrelevant to any given task. The
answer is not a smaller catalogue — it is showing the part that matches what
was asked and naming the rest.

The scoring is deliberately a keyword match rather than anything cleverer: the
corpora here are a few hundred short descriptions, and a ranking that can be
explained to the person wondering why their tool was skipped beats one that
cannot.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeVar

COMMON_TERM_FRACTION = 0.25
"""A task word appearing in more than this share of the corpus is noise."""

MIN_COMMON_HITS = 2
"""...but it takes at least this many hits, or a small corpus dismisses real matches."""

COMMON_TERM_CEILING = 12
"""...and above this many hits it is noise however large the corpus is.

A share alone stops working once the corpus is big and repetitive. Two hundred
tool names contain "page" twenty times, which is well under a quarter of them
and still tells you nothing about which twenty.
"""

NAME_WEIGHT = 6
"""A hit in the name is worth this many hits in the description."""

T = TypeVar("T")


def stem(term: str) -> str:
    """Crude singular of a task word — enough for matching, not for grammar."""
    if term.endswith("es") and len(term) > 5:
        return term[:-2]
    if term.endswith("s") and not term.endswith("ss") and len(term) > 4:
        return term[:-1]
    return term


def terms_of(task: str) -> set[str]:
    """The words in a task worth matching on. Short words carry no signal."""
    return {t for t in re.split(r"\W+", (task or "").lower()) if len(t) > 3}


def rank(
    items: Sequence[T],
    task: str,
    *,
    name_of: Callable[[T], str],
    text_of: Callable[[T], str],
) -> tuple[list[T], int]:
    """(items best-first, how many actually matched the task).

    A count of zero means nothing here is relevant — which is information, and
    the caller should act on it rather than showing an arbitrary six.
    """
    scored = rank_scored(items, task, name_of=name_of, text_of=text_of)
    return [item for _, item in scored], sum(1 for score, _ in scored if score > 0)


def rank_scored(
    items: Sequence[T],
    task: str,
    *,
    name_of: Callable[[T], str],
    text_of: Callable[[T], str],
) -> list[tuple[int, T]]:
    """The same ranking, with the scores, for callers that want a threshold.

    "Matched at all" is a weak test on a large corpus: one shared word in a
    description scores 1 and is almost always coincidence, while a hit in the
    name is usually the thing you meant.
    """
    terms = terms_of(task)
    if not terms:
        return [(0, item) for item in items]

    haystacks = [(name_of(item).lower(), (text_of(item) or "").lower()) for item in items]
    # Whole words, not substrings: "open the calculator" was matching every
    # entry whose description contains "workflow", on the strength of "work".
    # Plurals are stemmed on both sides, so a task saying "videos" still finds
    # an entry about "video".
    patterns = {term: re.compile(rf"\b{re.escape(stem(term))}(?:e?s)?\b") for term in terms}
    # A word that appears in half the corpus tells us nothing about which half
    # to pick: "open the calculator" matched a dozen entries on "open" alone.
    # Rarity is the signal, and computing it beats maintaining a stopword list.
    # The floor matters: in a corpus of two, "a quarter of them" is half an
    # entry, so a single genuine match would be dismissed as noise.
    noise_threshold = min(
        max(MIN_COMMON_HITS, len(items) * COMMON_TERM_FRACTION), COMMON_TERM_CEILING
    )
    common = {
        term for term, pattern in patterns.items()
        if sum(bool(pattern.search(f"{name} {body}")) for name, body in haystacks) > noise_threshold
    }
    terms -= common
    if not terms:
        return [(0, item) for item in items]

    scored = []
    for index, ((name, description), item) in enumerate(zip(haystacks, items, strict=True)):
        score = sum(NAME_WEIGHT for term in terms if patterns[term].search(name))
        score += sum(1 for term in terms if patterns[term].search(description))
        # index keeps the original order stable among equals
        scored.append((-score, index, item))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    return [(-negative, item) for negative, _, item in scored]
