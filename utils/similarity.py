import re
from difflib import SequenceMatcher
from typing import List, Dict


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_product / (norm1 * norm2)


def calculate_retrieval_metrics(ranks: List[int]) -> Dict[str, float]:
    """Calculate standard retrieval metrics from ranks."""
    if not ranks:
        return {'mrr': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0}

    # MRR (Mean Reciprocal Rank)
    mrr = sum(1.0 / rank for rank in ranks) / len(ranks)

    # Recall@k
    recall_at_1 = sum(1 for rank in ranks if rank <= 1) / len(ranks)
    recall_at_5 = sum(1 for rank in ranks if rank <= 5) / len(ranks)
    recall_at_10 = sum(1 for rank in ranks if rank <= 10) / len(ranks)

    return {
        'mrr': mrr,
        'recall_at_1': recall_at_1,
        'recall_at_5': recall_at_5,
        'recall_at_10': recall_at_10
    }


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


def compute_confidence(paper_data, s2_paper) -> float:
    """
    Compute confidence score between OpenAlex Paper object and Semantic Scholar dict.
    Returns: float in [0, 1]
    """

    openalex_paper = paper_data["paper"]

    # ---------- STEP 1: Global ID matching ----------
    # DOI
    if (
            openalex_paper.doi
            and s2_paper.get("doi")
            and openalex_paper.doi.lower() == s2_paper["doi"].lower()
    ):
        return 0.98

    # PMID
    if (
            openalex_paper.pmid
            and s2_paper.get("pmid")
            and openalex_paper.pmid == s2_paper["pmid"]
    ):
        return 0.97

    # ---------- STEP 2: Metadata matching ----------
    title_score = title_similarity(
        openalex_paper.title.lower(),
        s2_paper.get("title").lower(),
    )

    openalex_year = getattr(openalex_paper, "year", None)
    s2_year = s2_paper.get("year")

    same_year = (
            isinstance(openalex_year, int)
            and isinstance(s2_year, int)
            and openalex_year == s2_year
    )

    same_authors = author_overlap(
        paper_data.get("authors", []),
        [a.get("name") for a in s2_paper.get("authors", [])],
    )

    if title_score >= 0.95 and same_year and same_authors:
        return round(0.85 + 0.1 * title_score, 3)

    # ---------- NO STRONG MATCH ----------
    return round(title_score * 0.5, 3)
