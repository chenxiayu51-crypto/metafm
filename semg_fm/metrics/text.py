"""Text-sequence metrics with no assumption of character-level timestamps."""

from __future__ import annotations


def levenshtein_distance(reference: str, hypothesis: str) -> int:
    """Return Unicode-codepoint edit distance using O(min(n,m)) memory."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("reference and hypothesis must be strings")
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, reference_char in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Return edit distance normalized by reference length, with empty safety."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


__all__ = ["character_error_rate", "levenshtein_distance"]
