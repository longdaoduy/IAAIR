"""
Vector Store implementation for semantic search over scientific literature.

Supports multiple vector_store database backends (Milvus, Weaviate) with
optimized indexing for SciBERT embeddings and scientific text.
"""

import logging
from typing import List, Dict, Any, Optional
from .VectorClient import VectorClient

from knowledge_fabric.schemas import Document, SearchResult
from configurators.Settings import VectorDBConfig
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility, connections

logger = logging.getLogger(__name__)


class MilvusClient(VectorClient):
    """Milvus implementation of vector_store store."""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None

    async def connect(self):
        """Connect to Milvus server."""
        try:
            from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )

            # Create collection schema if it doesn't exist
            await self._ensure_collection_exists()
            logger.info("Connected to Milvus successfully")

        except ImportError:
            logger.error("pymilvus not installed. Run: pip install pymilvus")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    async def _ensure_collection_exists(self):
        """Create collection schema if it doesn't exist."""

        if utility.has_collection(self.config.collection_name):
            self.collection = Collection(self.config.collection_name)
            return

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="title_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="abstract_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(fields, "Scientific papers vector_store store")
        self.collection = Collection(self.config.collection_name, schema)

        # Create index
        index_params = {
            "metric_type": self.config.metric_type,
            "index_type": self.config.index_type,
            "params": {"nlist": self.config.nlist}
        }

        self.collection.create_index("title_embedding", index_params)
        self.collection.create_index("abstract_embedding", index_params)

        logger.info(f"Created Milvus collection: {self.config.collection_name}")

    async def close(self):
        """Close Milvus connection."""
        connections.disconnect("default")
        logger.info("Milvus connection closed")

    async def health_check(self) -> bool:
        """Check Milvus connection health."""
        try:
            return connections.has_connection("default")
        except Exception as e:
            logger.error(f"Milvus health check failed: {e}")
            return False

    async def store_document(self, document: Document):
        """Store document embeddings in Milvus."""
        if not document.title_embedding or not document.abstract_embedding:
            logger.warning(f"Document {document.id} missing embeddings, skipping")
            return

        data = [{
            "id": document.id,
            "title": document.title,
            "abstract": document.abstract,
            "title_embedding": document.title_embedding,
            "abstract_embedding": document.abstract_embedding,
            "metadata": {
                "document_type": document.document_type.value if document.document_type else None,
                "publication_date": document.publication_date.isoformat() if document.publication_date else None,
                "doi": document.doi,
                "authors": [author.name for author in document.authors],
                "venue": document.venue.name if document.venue else None,
                "keywords": document.keywords,
                "subjects": document.subjects
            }
        }]

        self.collection.insert(data)
        logger.debug(f"Stored document {document.id} in Milvus")

    async def search(self, query_embedding: List[float], limit: int = 10, search_field: str = "abstract_embedding") -> \
    List[SearchResult]:
        """Perform vector_store similarity search in Milvus."""
        search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": 10}}

        self.collection.load()
        results = self.collection.search(
            data=[query_embedding],
            anns_field=search_field,
            param=search_params,
            limit=limit,
            output_fields=["id", "title", "abstract", "metadata"]
        )

        search_results = []
        for hits in results:
            for hit in hits:
                # Convert to SearchResult format
                doc_data = hit.entity
                document = Document(
                    id=doc_data["id"],
                    title=doc_data["title"],
                    abstract=doc_data["abstract"]
                )

                search_result = SearchResult(
                    document=document,
                    score=float(hit.score),
                    query="",  # Will be set by caller
                    retrieval_method="vector_store"
                )
                search_results.append(search_result)

        return search_results