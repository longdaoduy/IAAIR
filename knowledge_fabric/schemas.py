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


class DocumentType(Enum):
    """Types of documents in the knowledge fabric."""
    PAPER = "paper"
    ARTICLE = "article"
    PREPRINT = "preprint"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"


class VenueType(Enum):
    """Types of publication venues."""
    JOURNAL = "journal"
    CONFERENCE = "conference"
    WORKSHOP = "workshop"
    ARXIV = "arxiv"
    BOOK = "book"


@dataclass
class Author:
    """Represents an author in the knowledge fabric."""
    id: str
    name: str
    orcid: Optional[str] = None
    affiliation: Optional[str] = None
    email: Optional[str] = None
    h_index: Optional[int] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Venue:
    """Represents a publication venue."""
    id: str
    name: str
    venue_type: VenueType
    issn: Optional[str] = None
    impact_factor: Optional[float] = None
    publisher: Optional[str] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Citation:
    """Represents a citation relationship between documents."""
    citing_paper_id: str
    cited_paper_id: str
    context: Optional[str] = None
    intent: Optional[str] = None  # background, method, result, etc.
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Figure:
    """Represents a figure or image in a document."""
    id: str
    document_id: str
    caption: str
    image_path: Optional[str] = None
    figure_type: Optional[str] = None  # chart, diagram, photo, etc.
    extracted_text: Optional[str] = None
    page_number: Optional[int] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Table:
    """Represents a table in a document."""
    id: str
    document_id: str
    caption: str
    content: Dict[str, Any]  # Structured table content
    extracted_text: Optional[str] = None
    page_number: Optional[int] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class Document:
    """Core document representation in the knowledge fabric."""
    id: str
    title: str
    abstract: str
    content: Optional[str] = None
    document_type: DocumentType = DocumentType.PAPER
    
    # Authors and venue
    authors: List[Author] = field(default_factory=list)
    venue: Optional[Venue] = None
    
    # Publication details
    publication_date: Optional[datetime] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pmid: Optional[str] = None
    url: Optional[str] = None
    
    # Citations
    citations: List[Citation] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # Multi-modal content
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    
    # Metadata
    keywords: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    language: str = "en"
    
    # Vector embeddings
    title_embedding: Optional[List[float]] = None
    abstract_embedding: Optional[List[float]] = None
    content_embedding: Optional[List[float]] = None
    
    # Provenance
    source: Optional[str] = None  # OpenAlex, Semantic Scholar, etc.
    source_id: Optional[str] = None
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class AttributionSpan:
    """Represents an attribution to a source document."""
    document_id: str
    start_char: int
    end_char: int
    text: str
    confidence: float
    page_number: Optional[int] = None


@dataclass
class ProvenanceRecord:
    """Records the provenance of data transformations."""
    id: str
    entity_id: str  # ID of the entity this provenance refers to
    operation: str  # ingestions, embedding, extraction, etc.
    source: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())


@dataclass
class EvidenceBundle:
    """Structured evidence package for query responses."""
    query: str
    sources: List[Document]
    attributions: List[AttributionSpan]
    confidence: float
    reasoning_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Multi-modal evidence
    relevant_figures: List[Figure] = field(default_factory=list)
    relevant_tables: List[Table] = field(default_factory=list)


@dataclass
class SearchResult:
    """Represents a search result with score and attribution."""
    document: Document
    score: float
    query: str
    attribution_spans: List[AttributionSpan] = field(default_factory=list)
    explanation: Optional[str] = None
    retrieval_method: str = "hybrid"  # vector_store, graph_store, hybrid


@dataclass
class QueryPlan:
    """Represents a query execution plan for hybrid retrieval."""
    query: str
    use_vector_search: bool = True
    use_graph_search: bool = True
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    filters: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 10