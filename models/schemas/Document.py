"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid
from Author import Author


# class DocumentType(Enum):
#     """Types of documents in the knowledge fabric."""
#     PAPER = "paper"
#     ARTICLE = "article"
#     PREPRINT = "preprint"
#     BOOK_CHAPTER = "book_chapter"
#     CONFERENCE_PAPER = "conference_paper"

@dataclass
class Document:
    """Core document representation in the knowledge fabric."""
    id: str
    title: str
    abstract: str
    content: Optional[str] = None
    # document_type: DocumentType = DocumentType.PAPER

    # Authors and venue
    authors: List[Author] = field(default_factory=list)
    # venue: Optional[Venue] = None

    # Publication details
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None

    # Citations
    # citations: List[Citation] = field(default_factory=list)
    # references: List[str] = field(default_factory=list)

    # Multi-modal content
    # figures: List[Figure] = field(default_factory=list)
    # tables: List[Table] = field(default_factory=list)

    # Metadata
    # keywords: List[str] = field(default_factory=list)
    # subjects: List[str] = field(default_factory=list)
    # language: str = "en"

    # Vector embeddings
    title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None
    # content_embedding: Optional[List[float]] = None

    # Provenance
    source: Optional[str] = None  # OpenAlex, Semantic Scholar, etc.
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
