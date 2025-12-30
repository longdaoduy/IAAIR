"""
FastAPI REST API for the Knowledge Fabric system.

Provides endpoints for document ingestions, hybrid search, attribution,
and evidence bundle generation with comprehensive OpenAPI documentation.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

from knowledge_fabric import KnowledgeFabric

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Knowledge Fabric API",
    description="Multi-modal scientific literature retrieval with attribution",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global knowledge fabric instance
knowledge_fabric: Optional[KnowledgeFabric] = None


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    max_results: int = 10
    vector_weight: Optional[float] = None
    graph_weight: Optional[float] = None
    include_attribution: bool = True


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    processing_time: float


class EvidenceBundleRequest(BaseModel):
    query: str
    max_sources: int = 5


class EvidenceBundleResponse(BaseModel):
    query: str
    evidence_bundle: Dict[str, Any]
    processing_time: float


class IngestionRequest(BaseModel):
    source: str  # "openalex", "semantic_scholar", "manual"
    query: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None
    max_documents: int = 1000


class IngestionResponse(BaseModel):
    status: str
    message: str
    statistics: Dict[str, Any]


# Dependency to get knowledge fabric instance
async def get_knowledge_fabric() -> KnowledgeFabric:
    global knowledge_fabric
    if knowledge_fabric is None:
        knowledge_fabric = KnowledgeFabric(settings)
    return knowledge_fabric


@app.on_event("startup")
async def startup_event():
    """Initialize the knowledge fabric system on startup."""
    global knowledge_fabric
    knowledge_fabric = KnowledgeFabric(settings)
    logger.info("Knowledge Fabric API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    global knowledge_fabric
    if knowledge_fabric:
        await knowledge_fabric.close()
    logger.info("Knowledge Fabric API shutdown")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Knowledge Fabric API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check(fabric: KnowledgeFabric = Depends(get_knowledge_fabric)):
    """Health check endpoint."""
    health = await fabric.health_check()
    status_code = 200 if health["overall"] else 503
    
    return {
        "status": "healthy" if health["overall"] else "unhealthy",
        "components": health
    }


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """
    Perform hybrid search across the knowledge fabric.
    
    Combines graph_store-based citation networks with vector_store semantic search
    to find relevant scientific literature with source attribution.
    """
    import time
    start_time = time.time()
    
    try:
        # Create query plan if weights provided
        query_plan = None
        if request.vector_weight is not None or request.graph_weight is not None:
            from models.schemas.schemas import QueryPlan
            query_plan = QueryPlan(
                query=request.query,
                vector_weight=request.vector_weight or settings.retrieval.default_vector_weight,
                graph_weight=request.graph_weight or settings.retrieval.default_graph_weight,
                max_results=request.max_results
            )
        
        # Perform search
        results = await fabric.search(
            query=request.query,
            max_results=request.max_results,
            query_plan=query_plan,
            include_attribution=request.include_attribution
        )
        
        # Convert to response format
        result_dicts = []
        for result in results:
            result_dict = {
                "document": {
                    "id": result.document.id,
                    "title": result.document.title,
                    "abstract": result.document.abstract,
                    "authors": [author.name for author in result.document.authors],
                    "venue": result.document.venue.name if result.document.venue else None,
                    "doi": result.document.doi,
                    "publication_date": result.document.publication_date.isoformat() if result.document.publication_date else None
                },
                "score": result.score,
                "retrieval_method": result.retrieval_method,
                "attribution_spans": [
                    {
                        "text": span.text,
                        "confidence": span.confidence,
                        "start_char": span.start_char,
                        "end_char": span.end_char
                    }
                    for span in result.attribution_spans
                ]
            }
            result_dicts.append(result_dict)
        
        processing_time = time.time() - start_time
        
        return SearchResponse(
            query=request.query,
            results=result_dicts,
            total_results=len(results),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evidence", response_model=EvidenceBundleResponse)
async def generate_evidence_bundle(
    request: EvidenceBundleRequest,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """
    Generate a comprehensive evidence bundle for a research query.
    
    Provides structured evidence with source attribution, confidence scores,
    and reasoning traces suitable for scientific fact-checking and research.
    """
    import time
    start_time = time.time()
    
    try:
        evidence_bundle = await fabric.generate_evidence_bundle(
            query=request.query,
            max_sources=request.max_sources
        )
        
        # Convert to response format
        bundle_dict = {
            "query": evidence_bundle.query,
            "sources": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "authors": [author.name for author in doc.authors],
                    "doi": doc.doi
                }
                for doc in evidence_bundle.sources
            ],
            "attributions": [
                {
                    "document_id": attr.document_id,
                    "text": attr.text,
                    "confidence": attr.confidence,
                    "start_char": attr.start_char,
                    "end_char": attr.end_char
                }
                for attr in evidence_bundle.attributions
            ],
            "overall_confidence": evidence_bundle.confidence,
            "reasoning_trace": evidence_bundle.reasoning_trace,
            "relevant_figures": len(evidence_bundle.relevant_figures),
            "relevant_tables": len(evidence_bundle.relevant_tables),
            "created_at": evidence_bundle.created_at.isoformat()
        }
        
        processing_time = time.time() - start_time
        
        return EvidenceBundleResponse(
            query=request.query,
            evidence_bundle=bundle_dict,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Evidence bundle generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestionResponse)
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """
    Ingest documents into the knowledge fabric.
    
    Supports ingestions from OpenAlex, Semantic Scholar, or manual document upload
    with complete provenance tracking and entity linking.
    """
    try:
        if request.source == "openalex":
            if not request.query:
                raise HTTPException(status_code=400, detail="Query required for OpenAlex ingestions")
            
            # Start background ingestions
            background_tasks.add_task(
                fabric.ingestion_pipeline.ingest_from_openalex,
                request.query,
                request.max_documents
            )
            
            return IngestionResponse(
                status="started",
                message=f"OpenAlex ingestions started for query: {request.query}",
                statistics={"max_documents": request.max_documents}
            )
            
        elif request.source == "semantic_scholar":
            if not request.query:
                raise HTTPException(status_code=400, detail="Query required for Semantic Scholar ingestions")
            
            # Start background ingestions
            background_tasks.add_task(
                fabric.ingestion_pipeline.ingest_from_semantic_scholar,
                request.query,
                request.max_documents
            )
            
            return IngestionResponse(
                status="started",
                message=f"Semantic Scholar ingestions started for query: {request.query}",
                statistics={"max_documents": request.max_documents}
            )
            
        elif request.source == "manual":
            if not request.documents:
                raise HTTPException(status_code=400, detail="Documents required for manual ingestions")
            
            # Convert document dicts to Document objects
            from models.schemas.schemas import Document
            documents = [Document(**doc_dict) for doc_dict in request.documents]
            
            # Start background ingestions
            background_tasks.add_task(
                fabric.ingest_documents,
                documents
            )
            
            return IngestionResponse(
                status="started",
                message=f"Manual ingestions started for {len(documents)} documents",
                statistics={"document_count": len(documents)}
            )
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported ingestions source: {request.source}"
            )
            
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """Get a specific document by ID."""
    try:
        document = await fabric.get_document_by_id(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "title": document.title,
            "abstract": document.abstract,
            "content": document.content,
            "authors": [
                {"name": author.name, "orcid": author.orcid, "affiliation": author.affiliation}
                for author in document.authors
            ],
            "venue": {
                "name": document.venue.name,
                "type": document.venue.venue_type.value,
                "impact_factor": document.venue.impact_factor
            } if document.venue else None,
            "publication_date": document.publication_date.isoformat() if document.publication_date else None,
            "doi": document.doi,
            "keywords": document.keywords,
            "subjects": document.subjects
        }
        
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/citation-network/{document_id}")
async def get_citation_network(
    document_id: str,
    depth: int = 2,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """Get citation network around a specific document."""
    try:
        network = await fabric.get_citation_network(document_id, depth)
        return network
        
    except Exception as e:
        logger.error(f"Failed to get citation network for {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/provenance/{entity_id}")
async def get_provenance(
    entity_id: str,
    fabric: KnowledgeFabric = Depends(get_knowledge_fabric)
):
    """Get complete provenance trail for an entity."""
    try:
        provenance = await fabric.get_provenance(entity_id)
        return {"entity_id": entity_id, "provenance": provenance}
        
    except Exception as e:
        logger.error(f"Failed to get provenance for {entity_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.api.log_level
    )