"""
Vector Store implementation for semantic search over scientific literature.

Supports multiple vector_store database backends (Milvus, Weaviate) with
optimized indexing for SciBERT embeddings and scientific text.
"""

import logging
from typing import List
from .VectorClient import VectorClient

from models.schemas.schemas import Document, SearchResult
from models.configurators import VectorDBConfig
import weaviate

logger = logging.getLogger(__name__)

class WeaviateClient(VectorClient):
    """Weaviate implementation of vector_store store."""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None

    async def connect(self):
        """Connect to Weaviate server."""
        try:
            self.client = weaviate.Client(
                url=f"{self.config.scheme}://{self.config.host}:{self.config.port}"
            )

            # Create schema if it doesn't exist
            await self._ensure_schema_exists()
            logger.info("Connected to Weaviate successfully")

        except ImportError:
            logger.error("weaviate-client not installed. Run: pip install weaviate-client")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    async def _ensure_schema_exists(self):
        """Create Weaviate schema if it doesn't exist."""
        schema = {
            "class": "ScientificPaper",
            "description": "Scientific literature documents",
            "vectorizer": "none",
            "properties": [
                {"name": "document_id", "dataType": ["string"]},
                {"name": "title", "dataType": ["text"]},
                {"name": "abstract", "dataType": ["text"]},
                {"name": "document_type", "dataType": ["string"]},
                {"name": "doi", "dataType": ["string"]},
                {"name": "authors", "dataType": ["string[]"]},
                {"name": "venue", "dataType": ["string"]},
                {"name": "publication_date", "dataType": ["date"]},
                {"name": "keywords", "dataType": ["string[]"]},
                {"name": "subjects", "dataType": ["string[]"]}
            ]
        }

        if not self.client.schema.exists("ScientificPaper"):
            self.client.schema.create_class(schema)
            logger.info("Created Weaviate schema for ScientificPaper")

    async def close(self):
        """Close Weaviate connection."""
        # Weaviate client doesn't require explicit closure
        logger.info("Weaviate connection closed")

    async def health_check(self) -> bool:
        """Check Weaviate connection health."""
        try:
            return self.client.is_ready()
        except Exception as e:
            logger.error(f"Weaviate health check failed: {e}")
            return False

    async def store_document(self, document: Document):
        """Store document in Weaviate."""
        if not document.abstract_embedding:
            logger.warning(f"Paper {document.id} missing embedding, skipping")
            return

        data_object = {
            "document_id": document.id,
            "title": document.title,
            "abstract": document.abstract,
            "document_type": document.document_type.value if document.document_type else None,
            "doi": document.doi,
            "authors": [author.name for author in document.authors],
            "venue": document.venue.name if document.venue else None,
            "publication_date": document.publication_date.isoformat() if document.publication_date else None,
            "keywords": document.keywords,
            "subjects": document.subjects
        }

        self.client.data_object.create(
            data_object=data_object,
            class_name="ScientificPaper",
            vector=document.abstract_embedding
        )

        logger.debug(f"Stored document {document.id} in Weaviate")

    async def search(
            self,
            query_embedding: List[float],
            limit: int = 10
    ) -> List[SearchResult]:
        """Perform vector_store similarity search in Weaviate."""
        result = (
            self.client.query
            .get("ScientificPaper", ["document_id", "title", "abstract", "authors", "venue"])
            .with_near_vector({"vector_store": query_embedding})
            .with_limit(limit)
            .with_additional(["certainty"])
            .do()
        )

        search_results = []
        for item in result["data"]["Get"]["ScientificPaper"]:
            document = Document(
                id=item["document_id"],
                title=item["title"],
                abstract=item["abstract"]
            )

            search_result = SearchResult(
                document=document,
                score=float(item["_additional"]["certainty"]),
                query="",  # Will be set by caller
                retrieval_method="vector_store"
            )
            search_results.append(search_result)

        return search_results
