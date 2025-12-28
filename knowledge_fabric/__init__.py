"""
Knowledge Fabric - Multi-modal Scientific Literature Retrieval System

A hybrid graph_store+vector_store knowledge fabric for scientific literature with
complete provenance tracking and attribution capabilities.
"""

__version__ = "0.1.0"
__author__ = "IAAIR Team"

from .core import KnowledgeFabric
from .schemas import Document, Author, Venue, Citation, EvidenceBundle

__all__ = [
    "KnowledgeFabric",
    "Document", 
    "Author",
    "Venue", 
    "Citation",
    "EvidenceBundle"
]