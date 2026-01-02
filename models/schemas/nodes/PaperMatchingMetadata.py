"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

from dataclasses import dataclass
from typing import Optional
from models.schemas.nodes.Paper import Paper

@dataclass
class PaperMatchingMetadata:
    """Metadata describing how two papers were matched or aligned."""

    src_paper: Paper
    dest_paper: Paper

    openalex_id: Optional[str] = None
    semantic_scholar_id: Optional[str] = None
    doi: Optional[str] = None

    confidence: float = 1.0
