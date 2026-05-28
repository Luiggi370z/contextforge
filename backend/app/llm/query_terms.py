"""Query tokenization for retrieval grading and contextual expansion."""

from __future__ import annotations

import re

from app.core.constants import GRADE_MIN_QUERY_TERM_OVERLAP

_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "with",
        "from",
        "that",
        "this",
        "what",
        "when",
        "where",
        "which",
        "does",
        "have",
        "need",
        "for",
        "the",
        "and",
        "are",
        "you",
        "your",
        "can",
        "per",
        "our",
        "not",
        "but",
        "will",
        "was",
        "were",
        "use",
        "always",
    }
)


def significant_query_terms(query: str) -> set[str]:
    """Tokenize the query into terms used for overlap grading (min length 3)."""
    return {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) >= 3 and term not in _QUERY_STOPWORDS
    }


def grading_terms(query: str, retrieval_query: str | None = None) -> set[str]:
    """Terms for overlap grading; include prior-turn terms when retrieval was expanded."""
    terms = significant_query_terms(query)
    if retrieval_query and retrieval_query.strip() != query.strip():
        terms |= significant_query_terms(retrieval_query)
    return terms


def term_matches_content(term: str, content_lower: str) -> bool:
    """Match exact token or simple singular/plural (e.g. ptos ↔ pto)."""
    if term in content_lower:
        return True
    singular = term.rstrip("s")
    return len(singular) >= 3 and singular in content_lower


def term_overlap_ratio(terms: set[str], content: str) -> float:
    if not terms:
        return 1.0
    content_lower = content.lower()
    hits = sum(1 for term in terms if term_matches_content(term, content_lower))
    return hits / len(terms)


def query_term_overlap_ratio(
    query: str,
    content: str,
    *,
    retrieval_query: str | None = None,
) -> float:
    return term_overlap_ratio(grading_terms(query, retrieval_query), content)


def discriminative_terms(terms: set[str], contexts: list[str]) -> set[str]:
    """Drop terms that appear in every snippet (e.g. shared words like approval, systems)."""
    if not terms or not contexts:
        return terms
    if len(contexts) == 1:
        return terms
    chunk_count = len(contexts)
    focused = {
        term
        for term in terms
        if sum(1 for content in contexts if term in content.lower()) < chunk_count
    }
    return focused or terms


def chunk_has_viable_overlap(
    query: str,
    content: str,
    *,
    retrieval_query: str | None = None,
) -> bool:
    return (
        query_term_overlap_ratio(query, content, retrieval_query=retrieval_query)
        >= GRADE_MIN_QUERY_TERM_OVERLAP
    )
