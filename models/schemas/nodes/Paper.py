"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import uuid

from models.schemas.nodes import PaperMetadata


# class DocumentType(Enum):
#     """Types of documents in the knowledge fabric."""
#     PAPER = "paper"
#     ARTICLE = "article"
#     PREPRINT = "preprint"
#     BOOK_CHAPTER = "book_chapter"
#     CONFERENCE_PAPER = "conference_paper"

@dataclass
class Paper:
    """Core document representation in the knowledge fabric."""
    id: str
    title: str
    abstract: str

    # Publication details
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    metadata: PaperMetadata
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
