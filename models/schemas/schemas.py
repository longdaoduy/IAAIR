"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

# from dataclasses import dataclass, field
# from datetime import datetime
# from typing import List, Dict, Optional, Any, Union
# from enum import Enum
# import uuid
#
# @dataclass
# class Figure:
#     """Represents a figure or image in a document."""
#     id: str
#     document_id: str
#     caption: str
#     image_path: Optional[str] = None
#     figure_type: Optional[str] = None  # chart, diagram, photo, etc.
#     extracted_text: Optional[str] = None
#     page_number: Optional[int] = None
#
#     def __post_init__(self):
#         if not self.id:
#             self.id = str(uuid.uuid4())
#
#
# @dataclass
# class Table:
#     """Represents a table in a document."""
#     id: str
#     document_id: str
#     caption: str
#     content: Dict[str, Any]  # Structured table content
#     extracted_text: Optional[str] = None
#     page_number: Optional[int] = None
#
#     def __post_init__(self):
#         if not self.id:
#             self.id = str(uuid.uuid4())

# @dataclass
# class AttributionSpan:
#     """Represents an attribution to a source document."""
#     document_id: str
#     start_char: int
#     end_char: int
#     text: str
#     confidence: float
#     page_number: Optional[int] = None
#
#
# @dataclass
# class ProvenanceRecord:
#     """Records the provenance of data transformations."""
#     id: str
#     entity_id: str  # ID of the entity this provenance refers to
#     operation: str  # ingestions, embedding, extraction, etc.
#     source: str
#     timestamp: datetime
#     metadata: Dict[str, Any] = field(default_factory=dict)
#
#     def __post_init__(self):
#         if not self.id:
#             self.id = str(uuid.uuid4())
#
#
# @dataclass
# class EvidenceBundle:
#     """Structured evidence package for query responses."""
#     query: str
#     sources: List[Document]
#     attributions: List[AttributionSpan]
#     confidence: float
#     reasoning_trace: List[str] = field(default_factory=list)
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     created_at: datetime = field(default_factory=datetime.now)
#
#     # Multi-modal evidence
#     relevant_figures: List[Figure] = field(default_factory=list)
#     relevant_tables: List[Table] = field(default_factory=list)
#
#
# @dataclass
# class SearchResult:
#     """Represents a search result with score and attribution."""
#     document: Document
#     score: float
#     query: str
#     attribution_spans: List[AttributionSpan] = field(default_factory=list)
#     explanation: Optional[str] = None
#     retrieval_method: str = "hybrid"  # vector_store, graph_store, hybrid
#
#
# @dataclass
# class QueryPlan:
#     """Represents a query execution plan for hybrid retrieval."""
#     query: str
#     use_vector_search: bool = True
#     use_graph_search: bool = True
#     vector_weight: float = 0.5
#     graph_weight: float = 0.5
#     filters: Dict[str, Any] = field(default_factory=dict)
#     max_results: int = 10