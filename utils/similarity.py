import re
from difflib import SequenceMatcher


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def title_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b),
    ).ratio()


def normalize_author(name: str) -> str:
    return normalize_text(name)


def author_overlap(authors_a, authors_b, threshold=0.5) -> bool:
    """
    Check whether main authors overlap sufficiently.
    `authors_*` is a list of author names (strings).
    """
    if not authors_a or not authors_b:
        return False

    set_a = {normalize_author(a) for a in authors_a[:3]}  # main authors
    set_b = {normalize_author(b) for b in authors_b[:3]}

    overlap = set_a & set_b
    return len(overlap) / max(len(set_a), 1) >= threshold


def compute_confidence(openalex_paper, s2_paper) -> dict:
    """
    Returns:
    {
        "same_paper": bool,
        "confidence": float,
        "match_type": "id" | "metadata" | "none"
    }
    """
    # ---------- STEP 1: Global ID matching ----------
    # DOI
    if openalex_paper.doi.lower() == s2_paper["doi"].lower():
        return {
            "same_paper": True,
            "confidence": 0.98,
            "match_type": "doi",
        }

    # PMID
    if openalex_paper.pmid == s2_paper["pmid"]:
        return {
            "same_paper": True,
            "confidence": 0.97,
            "match_type": "pmid",
        }

    # ---------- STEP 2: Metadata matching ----------
    title_score = title_similarity(
        openalex_paper.title,
        s2_paper.get("title"),
    )

    same_year = (
        openalex_paper.get("year")
        and s2_paper.get("year")
        and openalex_paper["year"] == s2_paper["year"]
    )

    same_authors = author_overlap(
        openalex_paper.get("authors", []),
        [a.get("name") for a in s2_paper.get("authors", [])],
    )

    if title_score >= 0.95 and same_year and same_authors:
        return {
            "same_paper": True,
            "confidence": round(0.85 + 0.1 * title_score, 3),
            "match_type": "metadata",
        }

    # ---------- NO MATCH ----------
    return {
        "same_paper": False,
        "confidence": round(title_score * 0.5, 3),
        "match_type": "none",
    }
