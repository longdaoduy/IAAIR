"""
Core Knowledge Fabric system that orchestrates all components.

This is the main entry point for the hybrid graph_store+vector_store retrieval system
with provenance tracking and attribution capabilities.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any

from models.schemas.schemas import Document, SearchResult, EvidenceBundle, QueryPlan
from models.pipelines import IngestionPipeline
from ..clients.graph_store import Neo4jClient
from ..clients.vector_store import MilvusClient
from .fusion.hybrid_retriever import HybridRetriever
from .attribution.attribution_tracker import AttributionTracker
from .provenance.provenance_ledger import ProvenanceLedger
from .integrations.snomed_ct import SnomedCTIntegrator
from models.configurators.Settings import Settings

logger = logging.getLogger(__name__)


class KnowledgeFabric:
    """
    Main Knowledge Fabric system that coordinates all components.
    
    Provides a unified interface for scientific literature ingestions,
    hybrid retrieval, attribution tracking, and provenance management.
    """
    
    def __init__(self, config: Optional[Settings] = None):
        """Initialize the Knowledge Fabric system."""
        self.config = config or Settings()
        self._setup_logging()
        
        # Initialize core components
        self.graph_store = Neo4jClient(self.config.neo4j)
        self.vector_store = MilvusClient(self.config.vector_db)
        self.attribution_tracker = AttributionTracker()
        self.provenance_ledger = ProvenanceLedger(self.graph_store)
        
        # Initialize biomedical ontology integrations
        self.snomed_integrator = None
        if hasattr(self.config, 'bioportal_api_key') and self.config.bioportal_api_key:
            self.snomed_integrator = SnomedCTIntegrator(api_key=self.config.bioportal_api_key)
            logger.info("SNOMED CT integration enabled")
        
        # Initialize ingestions and retrieval
        self.ingestion_pipeline = IngestionPipeline(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            provenance_ledger=self.provenance_ledger
        )
        
        self.hybrid_retriever = HybridRetriever(
            graph_store=self.graph_store,
            vector_store=self.vector_store,
            attribution_tracker=self.attribution_tracker
        )
        
        logger.info("Knowledge Fabric system initialized")
    
    def _setup_logging(self):
        """Configure logging for the system."""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def ingest_documents(
        self, 
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Ingest a batch of documents into the knowledge fabric.
        
        Args:
            documents: List of documents to ingest
            batch_size: Number of documents to process in each batch
            
        Returns:
            Ingestion statistics and results
        """
        logger.info(f"Starting ingestions of {len(documents)} documents")
        
        results = await self.ingestion_pipeline.process_documents(
            documents, batch_size=batch_size
        )
        
        logger.info(f"Ingestion completed: {results}")
        return results
    
    async def search(
        self,
        query: str,
        max_results: int = 10,
        query_plan: Optional[QueryPlan] = None,
        include_attribution: bool = True
    ) -> List[SearchResult]:
        """
        Perform hybrid search across the knowledge fabric.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            query_plan: Optional query execution plan
            include_attribution: Whether to include source attribution
            
        Returns:
            List of search results with scores and attribution
        """
        logger.info(f"Searching for: '{query}'")
        
        if not query_plan:
            query_plan = self._generate_query_plan(query, max_results)
        
        results = await self.hybrid_retriever.search(
            query=query,
            query_plan=query_plan,
            include_attribution=include_attribution
        )
        
        logger.info(f"Search completed: {len(results)} results")
        return results
    
    async def generate_evidence_bundle(
        self,
        query: str,
        max_sources: int = 5
    ) -> EvidenceBundle:
        """
        Generate a comprehensive evidence bundle for a query.
        
        Args:
            query: The research query or claim to investigate
            max_sources: Maximum number of source documents to include
            
        Returns:
            Complete evidence bundle with attribution and confidence
        """
        logger.info(f"Generating evidence bundle for: '{query}'")
        
        # Perform comprehensive search
        search_results = await self.search(
            query=query,
            max_results=max_sources * 2  # Get more to select best
        )
        
        # Select top sources and build evidence bundle
        evidence_bundle = await self.attribution_tracker.build_evidence_bundle(
            query=query,
            search_results=search_results[:max_sources]
        )
        
        # Record provenance
        await self.provenance_ledger.record_evidence_generation(
            evidence_bundle=evidence_bundle
        )
        
        logger.info("Evidence bundle generated successfully")
        return evidence_bundle
    
    def _generate_query_plan(self, query: str, max_results: int) -> QueryPlan:
        """Generate an optimal query execution plan."""
        # Simple heuristic-based planning (can be enhanced with ML)
        return QueryPlan(
            query=query,
            use_vector_search=True,
            use_graph_search=True,
            vector_weight=0.6,  # Slightly favor vector_store search
            graph_weight=0.4,
            max_results=max_results
        )
    
    async def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """Retrieve a specific document by ID."""
        return await self.vector_store.get_document(document_id)
    
    async def get_citation_network(
        self, 
        document_id: str, 
        depth: int = 2
    ) -> Dict[str, Any]:
        """Get the citation network around a specific document."""
        return await self.graph_store.get_citation_subgraph(document_id, depth)
    
    async def get_provenance(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get complete provenance trail for an entity."""
        return await self.provenance_ledger.get_lineage(entity_id)
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform system health check."""
        health = {}
        
        try:
            health['graph_store'] = await self.graph_store.health_check()
        except Exception as e:
            logger.error(f"Graph store health check failed: {e}")
            health['graph_store'] = False
        
        try:
            health['vector_store'] = await self.vector_store.health_check()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            health['vector_store'] = False
        
        health['overall'] = all(health.values())
        return health
    
    async def enhance_with_medical_concepts(
        self,
        documents: List[Document],
        batch_size: int = 10
    ) -> List[Document]:
        """
        Enhance documents with SNOMED CT medical concepts.
        
        Args:
            documents: Documents to enhance
            batch_size: Batch size for processing
            
        Returns:
            Enhanced documents with medical concepts
        """
        if not self.snomed_integrator:
            logger.warning("SNOMED CT integration not available - skipping medical concept enhancement")
            return documents
        
        logger.info(f"Enhancing {len(documents)} documents with SNOMED CT concepts")
        
        enhanced_documents = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            for doc in batch:
                try:
                    enhanced_doc = await self.snomed_integrator.enhance_document_with_medical_concepts(
                        doc, self
                    )
                    enhanced_documents.append(enhanced_doc)
                    
                except Exception as e:
                    logger.error(f"Failed to enhance document {doc.id} with medical concepts: {e}")
                    enhanced_documents.append(doc)  # Keep original if enhancement fails
            
            # Small delay between batches
            if i + batch_size < len(documents):
                await asyncio.sleep(0.5)
        
        return enhanced_documents
    
    async def search_medical_concepts(
        self,
        query: str,
        max_results: int = 20,
        exact_match: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for medical concepts using SNOMED CT.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            exact_match: Require exact match
            
        Returns:
            List of SNOMED CT concepts
        """
        if not self.snomed_integrator:
            logger.warning("SNOMED CT integration not available")
            return []
        
        concepts = await self.snomed_integrator.search_concepts(
            query, exact_match=exact_match, max_results=max_results
        )
        
        return [
            {
                'id': concept.concept_id,
                'label': concept.pref_label,
                'definition': concept.definition,
                'semantic_types': concept.semantic_types,
                'cui': concept.cui,
                'synonyms': concept.synonyms
            }
            for concept in concepts
        ]
    
    async def annotate_text_with_medical_concepts(
        self,
        text: str,
        longest_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Annotate text with SNOMED CT medical concepts.
        
        Args:
            text: Text to annotate
            longest_only: Return only longest matches
            
        Returns:
            List of annotation results
        """
        if not self.snomed_integrator:
            logger.warning("SNOMED CT integration not available")
            return []
        
        annotations = await self.snomed_integrator.annotate_text(
            text, longest_only=longest_only
        )
        
        return [
            {
                'text': ann.text,
                'start_position': ann.start_pos,
                'end_position': ann.end_pos,
                'concept': {
                    'id': ann.concept.concept_id,
                    'label': ann.concept.pref_label,
                    'definition': ann.concept.definition,
                    'semantic_types': ann.concept.semantic_types
                },
                'context': ann.context
            }
            for ann in annotations
        ]
    
    async def close(self):
        """Close all connections and cleanup resources."""
        logger.info("Shutting down Knowledge Fabric system")
        
        await self.graph_store.close()
        await self.vector_store.close()
        
        # Close SNOMED CT integrator if available
        if self.snomed_integrator:
            await self.snomed_integrator.close()
        
        logger.info("Knowledge Fabric system shutdown complete")