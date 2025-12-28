"""
Data ingestions pipeline for scientific literature.

Handles ingestions from OpenAlex, Semantic Scholar, and other scholarly APIs
with support for citation graph_store construction and entity linking.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from knowledge_fabric.schemas import Document, Author, Venue, Citation
from clients.graph_store import Neo4jClient
from clients.vector_store import MilvusClient
from knowledge_fabric.provenance.provenance_ledger import ProvenanceLedger
from .extractors.openalex_extractor import OpenAlexExtractor
from .extractors.semantic_scholar_extractor import SemanticScholarExtractor
from .processors.entity_linker import EntityLinker
from .processors.citation_extractor import CitationExtractor
from .processors.multimodal_extractor import MultiModalExtractor

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Orchestrates the complete data ingestions pipeline.
    
    Handles extraction from multiple sources, entity linking,
    citation graph_store construction, and storage in both graph_store
    and vector_store stores with complete provenance tracking.
    """
    
    def __init__(
        self,
        graph_store: Neo4jClient,
        vector_store: MilvusClient,
        provenance_ledger: ProvenanceLedger
    ):
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.provenance_ledger = provenance_ledger
        
        # Initialize extractors
        self.openalex_extractor = OpenAlexExtractor()
        self.semantic_scholar_extractor = SemanticScholarExtractor()
        
        # Initialize processors
        self.entity_linker = EntityLinker()
        self.citation_extractor = CitationExtractor()
        self.multimodal_extractor = MultiModalExtractor()
        
        logger.info("Ingestion pipeline initialized")
    
    async def process_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Process a batch of documents through the complete pipeline.
        
        Args:
            documents: List of documents to process
            batch_size: Number of documents to process in parallel
            
        Returns:
            Processing statistics and results
        """
        start_time = datetime.now()
        stats = {
            "total_documents": len(documents),
            "processed": 0,
            "failed": 0,
            "errors": [],
            "processing_time": 0
        }
        
        logger.info(f"Processing {len(documents)} documents in batches of {batch_size}")
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_results = await self._process_batch(batch)
            
            stats["processed"] += batch_results["processed"]
            stats["failed"] += batch_results["failed"]
            stats["errors"].extend(batch_results["errors"])
            
            logger.info(f"Batch {i//batch_size + 1} completed: "
                       f"{batch_results['processed']}/{len(batch)} successful")
        
        stats["processing_time"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Ingestion completed: {stats}")
        
        return stats
    
    async def _process_batch(self, documents: List[Document]) -> Dict[str, Any]:
        """Process a single batch of documents."""
        batch_stats = {"processed": 0, "failed": 0, "errors": []}
        
        # Create tasks for parallel processing
        tasks = [self._process_single_document(doc) for doc in documents]
        
        # Execute tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect statistics
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_stats["failed"] += 1
                batch_stats["errors"].append({
                    "document_id": documents[i].id,
                    "error": str(result)
                })
                logger.error(f"Failed to process document {documents[i].id}: {result}")
            else:
                batch_stats["processed"] += 1
        
        return batch_stats
    
    async def _process_single_document(self, document: Document) -> bool:
        """Process a single document through the complete pipeline."""
        try:
            # Record ingestions start
            await self.provenance_ledger.record_ingestion_start(document)
            
            # 1. Entity linking (authors, venues, concepts)
            document = await self.entity_linker.process(document)
            
            # 2. Citation extraction and linking
            document = await self.citation_extractor.process(document)
            
            # 3. Multi-modal content extraction (figures, tables)
            document = await self.multimodal_extractor.process(document)
            
            # 4. Generate embeddings
            document = await self._generate_embeddings(document)
            
            # 5. Store in graph_store database
            await self.graph_store.store_document(document)
            
            # 6. Store in vector_store database
            await self.vector_store.store_document(document)
            
            # 7. Record successful ingestions
            await self.provenance_ledger.record_ingestion_complete(document)
            
            return True
            
        except Exception as e:
            await self.provenance_ledger.record_ingestion_error(document, str(e))
            raise e
    
    async def _generate_embeddings(self, document: Document) -> Document:
        """Generate embeddings for document text content."""
        # This would integrate with the embedding service
        # For now, we'll mark where embeddings would be generated
        
        # Title embedding
        if document.title and not document.title_embedding:
            # document.title_embedding = await embedding_service.embed(document.title)
            pass
        
        # Abstract embedding
        if document.abstract and not document.abstract_embedding:
            # document.abstract_embedding = await embedding_service.embed(document.abstract)
            pass
        
        # Content embedding (if available)
        if document.content and not document.content_embedding:
            # document.content_embedding = await embedding_service.embed(document.content)
            pass
        
        return document
    
    async def ingest_from_openalex(
        self,
        query: str,
        max_documents: int = 1000
    ) -> Dict[str, Any]:
        """
        Ingest documents from OpenAlex API.
        
        Args:
            query: Search query for OpenAlex
            max_documents: Maximum number of documents to fetch
            
        Returns:
            Ingestion results
        """
        logger.info(f"Starting OpenAlex ingestions: query='{query}', max={max_documents}")
        
        # Extract documents from OpenAlex
        documents = await self.openalex_extractor.extract(query, max_documents)
        
        # Process through pipeline
        results = await self.process_documents(documents)
        results["source"] = "OpenAlex"
        results["query"] = query
        
        return results
    
    async def ingest_from_semantic_scholar(
        self,
        query: str,
        max_documents: int = 1000
    ) -> Dict[str, Any]:
        """
        Ingest documents from Semantic Scholar API.
        
        Args:
            query: Search query for Semantic Scholar
            max_documents: Maximum number of documents to fetch
            
        Returns:
            Ingestion results
        """
        logger.info(f"Starting Semantic Scholar ingestions: query='{query}', max={max_documents}")
        
        # Extract documents from Semantic Scholar
        documents = await self.semantic_scholar_extractor.extract(query, max_documents)
        
        # Process through pipeline
        results = await self.process_documents(documents)
        results["source"] = "Semantic Scholar"
        results["query"] = query
        
        return results