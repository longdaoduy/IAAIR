"""
Vector Store implementation for semantic search over scientific literature.

Supports multiple vector_store database backends (Milvus, Weaviate) with
optimized indexing for SciBERT embeddings and scientific text.
"""

import logging
from typing import List
from abc import ABC, abstractmethod

from models.schemas.schemas import Document, SearchResult

logger = logging.getLogger(__name__)


class VectorClient(ABC):
    """Abstract base class for vector_store store implementations."""

    @abstractmethod
    async def connect(self):
        """Establish connection to vector_store database."""
        pass

    @abstractmethod
    async def close(self):
        """Close vector_store database connection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check."""
        pass

    @abstractmethod
    async def store_document(self, document: Document):
        """Store document embeddings in vector_store database."""
        pass

    @abstractmethod
    async def search(
            self,
            query_embedding: List[float],
            limit: int = 10
    ) -> List[SearchResult]:
        """Perform vector_store similarity search."""
        pass
