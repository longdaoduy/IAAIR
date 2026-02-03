"""
Unified IAAIR API - Academic Paper Ingestion and Retrieval System

This unified API provides comprehensive endpoints for:
1. Paper Ingestion Pipeline:
   - Pull papers from OpenAlex
   - Enrich abstracts with Semantic Scholar
   - Upload to Neo4j and Zilliz
2. Graph Query System:
   - Cypher queries for Neo4j
   - Author, paper, and citation analysis
   - Collaboration networks and research trends
3. Semantic Search:
   - Vector similarity search
   - Hybrid search capabilities
"""

from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import logging
import os
import asyncio
import numpy as np
from enum import Enum
from fastapi import FastAPI, HTTPException, BackgroundTasks, APIRouter, Query, Path
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

# Import handlers
from pipelines.ingestions.handlers import IngestionHandler
from pipelines.ingestions.handlers import Neo4jHandler
from pipelines.ingestions.handlers import MilvusClient
from pipelines.ingestions.handlers.EmbeddingHandler import EmbeddingHandler
from pipelines.retrieval.GraphQueryHandler import GraphQueryHandler
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="IAAIR Unified API",
    description="Unified API for academic paper ingestion, graph queries, and semantic search",
    version="2.0.0"
)

# ===============================================================================
# PYDANTIC MODELS
# ===============================================================================

# Ingestion Models
class PaperRequest(BaseModel):
    """Request model for paper ingestion."""
    num_papers: int = Field(..., gt=0, le=1000, description="Number of papers to pull (1-1000)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for OpenAlex API")
    include_neo4j: bool = Field(False, description="Whether to upload to Neo4j")
    include_zilliz: bool = Field(False, description="Whether to upload to Zilliz")

class PaperResponse(BaseModel):
    """Response model for paper ingestion."""
    success: bool
    message: str
    papers_processed: int
    neo4j_uploaded: bool
    zilliz_uploaded: bool
    json_filename: str
    timestamp: str
    summary: Dict[str, Any]

class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., min_length=1, description="Text query to search for similar papers")
    top_k: int = Field(10, gt=0, le=50, description="Number of top results to return (1-50)")
    include_details: bool = Field(True, description="Whether to include detailed paper information from Neo4j")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (dense + sparse) or dense-only search")

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    success: bool
    message: str
    query: str
    results_found: int
    search_time_seconds: float
    results: List[Dict[str, Any]]

# Graph Query Models
class GraphQueryRequest(BaseModel):
    """Request model for custom Cypher queries."""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum results to return")



class GraphQueryResponse(BaseModel):
    """Response model for graph queries."""
    success: bool
    message: str
    query_time_seconds: float
    results_count: int
    results: List[Dict[str, Any]]
    query_info: Optional[Dict[str, Any]] = None

# Hybrid Fusion Models
class RoutingStrategy(str, Enum):
    """Routing strategies for hybrid fusion."""
    VECTOR_FIRST = "vector_first"  # Vector search -> Graph refinement
    GRAPH_FIRST = "graph_first"    # Graph search -> Vector similarity
    PARALLEL = "parallel"          # Both in parallel with fusion
    ADAPTIVE = "adaptive"          # Auto-select based on query analysis

class QueryType(str, Enum):
    """Query classification types."""
    SEMANTIC = "semantic"          # Concept-based queries
    STRUCTURAL = "structural"      # Relationship-based queries
    HYBRID = "hybrid"             # Mixed semantic and structural
    FACTUAL = "factual"           # Specific fact retrieval

class HybridSearchRequest(BaseModel):
    """Request model for hybrid search with fusion."""
    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(10, gt=0, le=100, description="Number of results to return")
    routing_strategy: RoutingStrategy = Field(RoutingStrategy.ADAPTIVE, description="Routing strategy")
    enable_reranking: bool = Field(True, description="Enable neural reranking")
    enable_attribution: bool = Field(True, description="Track source attribution")
    fusion_weights: Optional[Dict[str, float]] = Field(None, description="Custom fusion weights")
    include_provenance: bool = Field(False, description="Include detailed provenance")

class AttributionSpan(BaseModel):
    """Attribution span with source tracking."""
    text: str
    source_id: str
    source_type: str  # 'paper', 'abstract', 'citation'
    confidence: float = Field(..., ge=0.0, le=1.0)
    char_start: int
    char_end: int
    supporting_passages: List[str] = []

class SearchResult(BaseModel):
    """Enhanced search result with attribution and provenance."""
    paper_id: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    venue: Optional[str] = None
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    # Scoring and ranking
    relevance_score: float = Field(..., ge=0.0, description="Composite relevance score")
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # Attribution and provenance
    attributions: List[AttributionSpan] = []
    source_path: List[str] = []  # Retrieval path for provenance
    confidence_scores: Dict[str, float] = {}

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search."""
    success: bool
    message: str
    query: str
    query_type: QueryType
    routing_used: RoutingStrategy
    results_found: int
    search_time_seconds: float
    fusion_time_seconds: Optional[float] = None
    reranking_time_seconds: Optional[float] = None
    results: List[SearchResult]
    fusion_stats: Dict[str, Any] = {}
    attribution_stats: Dict[str, Any] = {}

class QueryAnalysisRequest(BaseModel):
    """Request for query classification analysis."""
    query: str = Field(..., min_length=1, description="Query to analyze")
    include_routing_suggestion: bool = Field(True, description="Include routing strategy suggestion")

class QueryAnalysisResponse(BaseModel):
    """Response for query analysis."""
    success: bool
    query: str
    query_type: QueryType
    confidence: float
    suggested_routing: RoutingStrategy
    analysis_details: Dict[str, Any]

class RoutingPerformanceResponse(BaseModel):
    """Response for routing performance metrics."""
    success: bool
    timestamp: str
    performance_metrics: Dict[str, Dict[str, float]]
    recommendations: List[str]

class AttributionStatsResponse(BaseModel):
    """Response for attribution statistics."""
    success: bool
    timestamp: str
    total_queries_tracked: int
    attribution_accuracy: float
    high_confidence_rate: float
    source_type_distribution: Dict[str, int]
    average_attributions_per_result: float



# ===============================================================================
# GLOBAL HANDLERS
# ===============================================================================

# Initialize handlers
ingestion_handler = IngestionHandler()
neo4j_handler = Neo4jHandler()
zilliz_handler = MilvusClient()
embedding_handler = EmbeddingHandler()
query_handler = None

def get_query_handler():
    """Get or create graph query handler."""
    global query_handler
    if query_handler is None:
        query_handler = GraphQueryHandler()
    return query_handler

# ===============================================================================
# HYBRID FUSION SYSTEM
# ===============================================================================

class QueryClassifier:
    """Classify queries to determine optimal routing strategy."""
    
    def __init__(self):
        # Keywords indicating different query types
        self.semantic_keywords = {'similar', 'related', 'about', 'concerning', 'regarding'}
        self.structural_keywords = {'cited', 'authored', 'collaborated', 'published', 'references'}
        self.factual_keywords = {'who', 'what', 'when', 'where', 'which', 'how many'}
    
    def classify_query(self, query: str) -> Tuple[QueryType, float]:
        """Classify query and return confidence score."""
        query_lower = query.lower()
        
        # Count keyword matches
        semantic_score = sum(1 for kw in self.semantic_keywords if kw in query_lower)
        structural_score = sum(1 for kw in self.structural_keywords if kw in query_lower)
        factual_score = sum(1 for kw in self.factual_keywords if kw in query_lower)
        
        # Simple heuristic-based classification
        scores = {
            QueryType.SEMANTIC: semantic_score + (0.5 if len(query.split()) > 5 else 0),
            QueryType.STRUCTURAL: structural_score + (0.3 if any(op in query_lower for op in ['and', 'or', 'not']) else 0),
            QueryType.FACTUAL: factual_score + (0.4 if query_lower.startswith(tuple(self.factual_keywords)) else 0)
        }
        
        max_type = max(scores, key=scores.get)
        max_score = scores[max_type]
        
        # If no clear winner, classify as hybrid
        if max_score < 1.0 or sum(scores.values()) > 1.5:
            return QueryType.HYBRID, 0.6
        
        confidence = min(max_score / 2.0, 1.0)
        return max_type, confidence

class RoutingDecisionEngine:
    """Decide optimal routing strategy based on query and system state."""
    
    def __init__(self):
        self.query_classifier = QueryClassifier()
        self.performance_history = {}  # Track routing performance
    
    def decide_routing(self, query: str, request: HybridSearchRequest) -> RoutingStrategy:
        """Decide routing strategy based on query analysis."""
        if request.routing_strategy != RoutingStrategy.ADAPTIVE:
            return request.routing_strategy
        
        query_type, confidence = self.query_classifier.classify_query(query)
        
        # Routing decision logic
        if query_type == QueryType.SEMANTIC and confidence > 0.7:
            return RoutingStrategy.VECTOR_FIRST
        elif query_type == QueryType.STRUCTURAL and confidence > 0.7:
            return RoutingStrategy.GRAPH_FIRST
        else:
            return RoutingStrategy.PARALLEL
    
    def update_performance(self, strategy: RoutingStrategy, query_type: QueryType, 
                          latency: float, relevance_score: float):
        """Update performance tracking for adaptive routing."""
        key = f"{strategy}_{query_type}"
        if key not in self.performance_history:
            self.performance_history[key] = {'latencies': [], 'relevance_scores': []}
        
        self.performance_history[key]['latencies'].append(latency)
        self.performance_history[key]['relevance_scores'].append(relevance_score)
        
        # Keep only recent history (last 100 queries)
        for metric_list in self.performance_history[key].values():
            if len(metric_list) > 100:
                metric_list.pop(0)

class ResultFusion:
    """Fuse results from different retrieval strategies."""
    
    def __init__(self):
        self.default_weights = {
            'vector_score': 0.4,
            'graph_score': 0.3,
            'rerank_score': 0.3
        }
    
    def fuse_results(self, vector_results: List[Dict], graph_results: List[Dict],
                    fusion_weights: Optional[Dict[str, float]] = None) -> List[SearchResult]:
        """Fuse results from vector and graph search."""
        weights = fusion_weights or self.default_weights
        
        # Create result index by paper_id
        all_results = {}
        
        # Process vector results
        for result in vector_results:
            paper_id = result.get('paper_id')
            if paper_id:
                    all_results[paper_id] = {
                        'paper_id': paper_id,
                        'title': result.get('title', ''),
                        'abstract': result.get('abstract'),
                        'authors': result.get('authors', []),
                        'venue': result.get('venue'),
                        'publication_date': result.get('publication_date'),
                        'doi': result.get('doi'),
                        'vector_score': min(result.get('similarity_score', 0.0), 1.0),  # Normalize to [0,1]
                        'graph_score': 0.0,
                        'source_path': ['vector_search']
                    }        # Process graph results and merge
        for result in graph_results:
            paper_id = result.get('paper_id') or result.get('id')
            if paper_id:
                if paper_id in all_results:
                    all_results[paper_id]['graph_score'] = min(result.get('relevance_score', 0.5), 1.0)  # Normalize to [0,1]
                    all_results[paper_id]['source_path'].append('graph_search')
                else:
                    all_results[paper_id] = {
                        'paper_id': paper_id,
                        'title': result.get('title', ''),
                        'abstract': result.get('abstract'),
                        'authors': result.get('authors', []),
                        'venue': result.get('venue'),
                        'publication_date': result.get('publication_date'),
                        'doi': result.get('doi'),
                        'vector_score': 0.0,
                        'graph_score': min(result.get('relevance_score', 0.5), 1.0),  # Normalize to [0,1]
                        'source_path': ['graph_search']
                    }
        
        # Calculate fusion scores
        fused_results = []
        for result_data in all_results.values():
            # Calculate weighted fusion score
            raw_relevance_score = (
                weights['vector_score'] * result_data['vector_score'] +
                weights['graph_score'] * result_data['graph_score']
            )
            
            # Normalize the final score to ensure it's reasonable (optional cap at 1.0)
            relevance_score = min(raw_relevance_score, 1.0)
            
            search_result = SearchResult(
                paper_id=result_data['paper_id'],
                title=result_data['title'],
                abstract=result_data['abstract'],
                authors=result_data['authors'],
                venue=result_data['venue'],
                publication_date=result_data['publication_date'],
                doi=result_data['doi'],
                relevance_score=relevance_score,
                vector_score=result_data['vector_score'],
                graph_score=result_data['graph_score'],
                source_path=result_data['source_path'],
                attributions=[],
                confidence_scores={
                    'vector_confidence': result_data['vector_score'],
                    'graph_confidence': result_data['graph_score'],
                    'raw_fusion_score': raw_relevance_score  # Keep track of original score
                }
            )
            fused_results.append(search_result)
        
        # Sort by relevance score
        fused_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return fused_results

class ScientificReranker:
    """Rerank results using scientific domain knowledge."""
    
    def __init__(self):
        # Scientific relevance factors
        self.citation_weight = 0.3
        self.recency_weight = 0.2
        self.venue_weight = 0.2
        self.author_weight = 0.15
        self.semantic_weight = 0.15
    
    async def rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using scientific relevance factors."""
        for result in results:
            # Calculate domain-specific scores
            citation_score = self._calculate_citation_score(result)
            recency_score = self._calculate_recency_score(result)
            venue_score = self._calculate_venue_score(result)
            author_score = self._calculate_author_score(result)
            semantic_score = result.relevance_score  # Use existing relevance
            
            # Weighted combination
            rerank_score = (
                self.citation_weight * citation_score +
                self.recency_weight * recency_score +
                self.venue_weight * venue_score +
                self.author_weight * author_score +
                self.semantic_weight * semantic_score
            )
            
            result.rerank_score = rerank_score
            result.confidence_scores.update({
                'citation_score': citation_score,
                'recency_score': recency_score,
                'venue_score': venue_score,
                'author_score': author_score
            })
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
        return results
    
    def _calculate_citation_score(self, result: SearchResult) -> float:
        """Calculate citation-based relevance score."""
        # Placeholder - would use actual citation counts
        return 0.5
    
    def _calculate_recency_score(self, result: SearchResult) -> float:
        """Calculate recency-based relevance score."""
        # Placeholder - would use publication date
        return 0.5
    
    def _calculate_venue_score(self, result: SearchResult) -> float:
        """Calculate venue-based relevance score."""
        # Placeholder - would use venue impact factor
        return 0.5
    
    def _calculate_author_score(self, result: SearchResult) -> float:
        """Calculate author-based relevance score."""
        # Placeholder - would use author h-index/reputation
        return 0.5

class AttributionTracker:
    """Track source attribution for retrieved content."""
    
    def __init__(self):
        self.confidence_threshold = 0.7
    
    def track_attributions(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Add attribution tracking to search results."""
        for result in results:
            # Create basic attribution spans
            attributions = self._create_attribution_spans(result, query)
            result.attributions = attributions
        
        return results
    
    def _create_attribution_spans(self, result: SearchResult, query: str) -> List[AttributionSpan]:
        """Create attribution spans for a result."""
        attributions = []
        
        # Title attribution
        if result.title:
            attributions.append(AttributionSpan(
                text=result.title,
                source_id=result.paper_id,
                source_type='paper',
                confidence=0.9,
                char_start=0,
                char_end=len(result.title),
                supporting_passages=[result.title]
            ))
        
        # Abstract attribution (if available)
        if result.abstract:
            # Simple span creation - would be more sophisticated in practice
            abstract_words = result.abstract.split()
            query_words = set(query.lower().split())
            
            for i, word in enumerate(abstract_words):
                if word.lower() in query_words:
                    # Create span around matching word (simplified)
                    start_pos = sum(len(w) + 1 for w in abstract_words[:i])
                    end_pos = start_pos + len(word)
                    
                    attributions.append(AttributionSpan(
                        text=word,
                        source_id=result.paper_id,
                        source_type='abstract',
                        confidence=0.8,
                        char_start=start_pos,
                        char_end=end_pos,
                        supporting_passages=[result.abstract[:100] + '...']
                    ))
        
        return attributions

# Global hybrid system components
routing_engine = RoutingDecisionEngine()
result_fusion = ResultFusion()
scientific_reranker = ScientificReranker()
attribution_tracker = AttributionTracker()

# ===============================================================================
# ROOT ENDPOINT
# ===============================================================================

@app.get("/")
async def root():
    """Root endpoint providing comprehensive API information."""
    return {
        "name": "IAAIR Unified API",
        "version": "2.0.0",
        "description": "Unified API for academic paper ingestion, graph queries, and semantic search",
        "endpoints": {
            "ingestion": {
                "/pull-papers": "POST - Pull papers from OpenAlex and process through pipeline",
                "/download/{filename}": "GET - Download generated JSON files"
            },
            "semantic_search": {
                "/search": "POST - Semantic search for similar papers",
                "/hybrid-search": "POST - Hybrid fusion search with attribution"
            },
            "analytics": {
                "/analytics/routing-performance": "GET - Routing strategy performance metrics",
                "/analytics/query-classification": "POST - Analyze query classification",
                "/analytics/attribution-stats": "GET - Attribution tracking statistics"
            },
            "graph_queries": {
                "/graph/query": "POST - Execute custom Cypher queries"
            },
            "system": {
                "/health": "GET - Health check endpoint",
                "/docs": "GET - API documentation"
            }
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ingestion": "available",
            "graph_queries": "available",
            "semantic_search": "available"
        }
    }

# ===============================================================================
# PAPER INGESTION ENDPOINTS
# ===============================================================================

@app.post("/pull-papers", response_model=PaperResponse)
async def pull_papers(request: PaperRequest):
    """
    Pull papers from OpenAlex, enrich with Semantic Scholar, and upload to databases.
    
    This endpoint:
    1. Fetches papers from OpenAlex API
    2. Enriches abstracts using Semantic Scholar
    3. Uploads to Neo4j (if requested)
    4. Generates embeddings and uploads to Zilliz (if requested)
    5. Returns JSON file and processing summary
    """
    timestamp = datetime.now()
    
    try:
        logger.info(f"Starting paper ingestion for {request.num_papers} papers")
        
        # Step 1: Pull papers from OpenAlex
        logger.info("Step 1: Pulling papers from OpenAlex...")
        papers_data = ingestion_handler.pull_OpenAlex_Paper(
            count=request.num_papers,
            filters=request.filters,
            save_to_file=True,
            upload_to_neo4j=False  # We'll handle this separately
        )
        
        if not papers_data:
            raise HTTPException(status_code=400, detail="Failed to fetch papers from OpenAlex")
        
        # Step 2: Enrich with Semantic Scholar
        logger.info("Step 2: Enriching papers with Semantic Scholar...")
        enriched_papers = ingestion_handler.enrich_papers_with_semantic_scholar(
            papers_data=papers_data,
            save_to_file=True
        )
        
        # Step 3: Upload to Neo4j (if requested)
        neo4j_success = True
        if request.include_neo4j:
            logger.info("Step 3: Uploading to Neo4j...")
            neo4j_success = await neo4j_handler.upload_papers_to_neo4j(enriched_papers)
        
        # Step 4: Generate embeddings and upload to Zilliz (if requested)
        zilliz_success = True
        if request.include_zilliz:
            logger.info("Step 4: Processing embeddings and uploading to Zilliz...")
            embedding_success = await generate_and_upload_embeddings(enriched_papers, timestamp)
            zilliz_success = embedding_success
        
        # Generate summary
        total_authors = sum(len(pd.get('authors', [])) for pd in enriched_papers)
        total_citations = sum(len(pd.get('citations', [])) for pd in enriched_papers)
        
        summary = {
            "papers_fetched": len(enriched_papers),
            "authors_extracted": total_authors,
            "citations_extracted": total_citations,
            "avg_citations_per_paper": total_citations / len(enriched_papers) if enriched_papers else 0,
            "processing_time_seconds": (datetime.now() - timestamp).total_seconds()
        }
        
        # Create response filename
        json_filename = f"enriched_openalex_papers_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        logger.info(f"Paper ingestion completed successfully. Processed {len(enriched_papers)} papers")
        
        return PaperResponse(
            success=True,
            message=f"Successfully processed {len(enriched_papers)} papers",
            papers_processed=len(enriched_papers),
            neo4j_uploaded=neo4j_success if request.include_neo4j else False,
            zilliz_uploaded=zilliz_success if request.include_zilliz else False,
            json_filename=json_filename,
            timestamp=timestamp.isoformat(),
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error during paper ingestion: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated JSON file."""
    file_path = os.path.join(os.getcwd(), filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type='application/json',
        filename=filename
    )

async def generate_and_upload_embeddings(papers_data: List[Dict], timestamp: datetime) -> bool:
    """Generate embeddings for papers and upload to Zilliz."""
    try:
        # Create a temporary JSON file for the embedding handler
        temp_filename = f"temp_papers_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        # Save papers to temporary file
        ingestion_handler.save_papers_to_json(papers_data, temp_filename)
        
        # Generate embeddings using the process_papers method
        output_filename = embedding_handler.process_papers(input_file=temp_filename)
        
        if not output_filename or not os.path.exists(output_filename):
            logger.error("Failed to generate embeddings")
            return False
        
        # Connect to Zilliz and upload embeddings
        if not zilliz_handler.connect():
            logger.error("Failed to connect to Zilliz")
            return False
        
        # Upload embeddings using the generated embedding file with papers data for hybrid search
        upload_success = zilliz_handler.upload_embeddings(
            embedding_file=output_filename,
            papers_data=papers_data  # Pass papers data for sparse embeddings
        )
        
        # Cleanup temporary files
        try:
            os.remove(temp_filename)
            os.remove(output_filename)  # Also remove the embedding file after upload
        except:
            pass  # Ignore cleanup errors
        
        return upload_success
        
    except Exception as e:
        logger.error(f"Error in embedding generation and upload: {e}")
        return False

# ===============================================================================
# SEMANTIC SEARCH ENDPOINTS
# ===============================================================================

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic or hybrid search for similar papers.
    
    This endpoint:
    1. Generates dense embedding for the query text
    2. Optionally generates sparse embedding for hybrid search
    3. Searches Zilliz using hybrid search (dense + sparse) or dense-only search
    4. Retrieves detailed information from Neo4j (if requested)
    5. Returns ranked results with similarity scores
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Starting semantic search for query: '{request.query}'")
        
        # Step 1: Connect to Zilliz and perform similarity search
        if not zilliz_handler.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to Zilliz vector database")
        
        # Search for similar papers in Zilliz
        similar_papers = zilliz_handler.search_similar_papers(
            query_text=request.query,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid
        )
        
        if not similar_papers:
            return SearchResponse(
                success=True,
                message="No similar papers found",
                query=request.query,
                results_found=0,
                search_time_seconds=(datetime.now() - start_time).total_seconds(),
                results=[]
            )
        
        results = similar_papers
        
        # Step 2: Enrich with detailed information from Neo4j (if requested)
        if request.include_details:
            logger.info(f"Enriching {len(similar_papers)} results with Neo4j data...")
            
            paper_ids = [paper["paper_id"] for paper in similar_papers if paper.get("paper_id")]
            
            if paper_ids:
                detailed_papers = await neo4j_handler.get_papers_by_ids(paper_ids)
                
                # Create a lookup dict for detailed paper data
                detailed_lookup = {paper["id"]: paper for paper in detailed_papers}
                
                # Merge Zilliz results with Neo4j details
                enriched_results = []
                for zilliz_result in similar_papers:
                    paper_id = zilliz_result.get("paper_id")
                    detailed_paper = detailed_lookup.get(paper_id, {})
                    
                    # Combine data, prioritizing Neo4j details where available
                    enriched_result = {
                        "paper_id": paper_id,
                        "similarity_score": zilliz_result["similarity_score"],
                        "distance": zilliz_result["distance"],
                        "title": detailed_paper.get("title") or zilliz_result.get("title"),
                        "abstract": detailed_paper.get("abstract") or zilliz_result.get("abstract"),
                        "doi": detailed_paper.get("doi"),
                        "publication_date": detailed_paper.get("publication_date"),
                        "authors": detailed_paper.get("authors", []),
                        "venue": detailed_paper.get("venue"),
                        "cited_by_count": detailed_paper.get("cited_by_count", 0),
                        "citations_count": len(detailed_paper.get("citations", [])),
                        "source": detailed_paper.get("source")
                    }
                    enriched_results.append(enriched_result)
                
                results = enriched_results
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Semantic search completed. Found {len(results)} results in {search_time:.2f} seconds")
        
        return SearchResponse(
            success=True,
            message=f"Found {len(results)} similar papers",
            query=request.query,
            results_found=len(results),
            search_time_seconds=search_time,
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_fusion_search(request: HybridSearchRequest):
    """
    Advanced hybrid search with fusion, reranking, and attribution.
    
    This endpoint implements:
    1. Query classification and adaptive routing
    2. Vector-first, graph-first, or parallel search strategies
    3. Result fusion with configurable weights
    4. Scientific domain-aware reranking
    5. Source attribution and provenance tracking
    """
    start_time = datetime.now()
    fusion_start = None
    reranking_start = None
    
    try:
        logger.info(f"Starting hybrid search for query: '{request.query}'")
        
        # Step 1: Query classification and routing decision
        query_type, confidence = routing_engine.query_classifier.classify_query(request.query)
        routing_strategy = routing_engine.decide_routing(request.query, request)
        
        logger.info(f"Query classified as {query_type} (confidence: {confidence:.2f}), using {routing_strategy} routing")
        
        # Step 2: Execute search based on routing strategy
        vector_results = []
        graph_results = []
        
        fusion_start = datetime.now()
        
        if routing_strategy == RoutingStrategy.VECTOR_FIRST:
            # Vector search first, then graph refinement
            vector_results = await _execute_vector_search(request.query, request.top_k * 2)
            if vector_results:
                # Use top vector results to inform graph search
                paper_ids = [r.get('paper_id') for r in vector_results[:request.top_k]]
                graph_results = await _execute_graph_refinement(paper_ids, request.query)
            
        elif routing_strategy == RoutingStrategy.GRAPH_FIRST:
            # Graph search first, then vector similarity
            graph_results = await _execute_graph_search(request.query, request.top_k * 2)
            if graph_results:
                # Use graph results to inform vector search
                paper_ids = [r.get('paper_id', r.get('id')) for r in graph_results[:request.top_k]]
                vector_results = await _execute_vector_refinement(paper_ids, request.query)
            
        elif routing_strategy == RoutingStrategy.PARALLEL:
            # Execute both searches in parallel
            vector_task = _execute_vector_search(request.query, request.top_k)
            graph_task = _execute_graph_search(request.query, request.top_k)
            
            vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Step 3: Result fusion
        fused_results = result_fusion.fuse_results(
            vector_results or [],
            graph_results or [],
            request.fusion_weights
        )
        
        fusion_time = (datetime.now() - fusion_start).total_seconds()
        
        # Limit to requested number of results
        fused_results = fused_results[:request.top_k]
        
        # Step 4: Reranking (if enabled)
        reranking_time = None
        if request.enable_reranking and fused_results:
            reranking_start = datetime.now()
            fused_results = await scientific_reranker.rerank_results(fused_results, request.query)
            reranking_time = (datetime.now() - reranking_start).total_seconds()
        
        # Step 5: Attribution tracking (if enabled)
        if request.enable_attribution and fused_results:
            fused_results = attribution_tracker.track_attributions(fused_results, request.query)
        
        # Step 6: Calculate statistics
        total_time = (datetime.now() - start_time).total_seconds()
        
        fusion_stats = {
            'vector_results_count': len(vector_results or []),
            'graph_results_count': len(graph_results or []),
            'fusion_method': routing_strategy.value,
            'fusion_weights': request.fusion_weights or result_fusion.default_weights
        }
        
        attribution_stats = {
            'total_attributions': sum(len(r.attributions) for r in fused_results),
            'high_confidence_attributions': sum(
                1 for r in fused_results 
                for a in r.attributions 
                if a.confidence > attribution_tracker.confidence_threshold
            ),
            'attribution_enabled': request.enable_attribution
        }
        
        # Update routing performance tracking
        avg_relevance = sum(r.relevance_score for r in fused_results) / len(fused_results) if fused_results else 0
        routing_engine.update_performance(routing_strategy, query_type, total_time, avg_relevance)
        
        logger.info(f"Hybrid search completed. Found {len(fused_results)} results in {total_time:.2f}s")
        
        return HybridSearchResponse(
            success=True,
            message=f"Hybrid search completed using {routing_strategy.value} strategy",
            query=request.query,
            query_type=query_type,
            routing_used=routing_strategy,
            results_found=len(fused_results),
            search_time_seconds=total_time,
            fusion_time_seconds=fusion_time,
            reranking_time_seconds=reranking_time,
            results=fused_results,
            fusion_stats=fusion_stats,
            attribution_stats=attribution_stats
        )
        
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

# Helper functions for hybrid search
async def _execute_vector_search(query: str, top_k: int) -> List[Dict]:
    """Execute vector search."""
    try:
        if not zilliz_handler.connect():
            logger.warning("Vector search failed: Could not connect to Zilliz")
            return []
        
        results = zilliz_handler.search_similar_papers(
            query_text=query,
            top_k=top_k,
            use_hybrid=True
        )
        return results or []
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        return []

async def _execute_graph_search(query: str, top_k: int) -> List[Dict]:
    """Execute graph search using Cypher query."""
    try:
        handler = get_query_handler()
        
        # Simple graph search - would be more sophisticated in practice
        cypher_query = f"""
        MATCH (p:Paper)
        WHERE p.title CONTAINS $query OR p.abstract CONTAINS $query
        RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
               p.doi as doi, p.publication_date as publication_date,
               [(p)<-[:AUTHORED]-(a:Author) | a.name] as authors,
               [(p)-[:PUBLISHED_IN]->(v:Venue) | v.name][0] as venue
        LIMIT $limit
        """
        
        results = handler.execute_query(cypher_query, {"query": query, "limit": top_k})
        
        # Add basic relevance scores
        for result in results:
            result['relevance_score'] = 0.5  # Placeholder score
        
        return results
    except Exception as e:
        logger.error(f"Graph search error: {e}")
        return []

async def _execute_graph_refinement(paper_ids: List[str], query: str) -> List[Dict]:
    """Refine vector results using graph relationships."""
    try:
        handler = get_query_handler()
        
        # Find related papers through citations and collaborations
        cypher_query = f"""
        MATCH (seed:Paper)
        WHERE seed.id IN $paper_ids
        MATCH (related:Paper)-[:CITED_BY|:CITES*1..2]-(seed)
        WHERE related.title CONTAINS $query OR related.abstract CONTAINS $query
        RETURN DISTINCT related.id as paper_id, related.title as title, 
               related.abstract as abstract, related.doi as doi,
               related.publication_date as publication_date,
               [(related)<-[:AUTHORED]-(a:Author) | a.name] as authors,
               [(related)-[:PUBLISHED_IN]->(v:Venue) | v.name][0] as venue
        LIMIT 20
        """
        
        results = handler.execute_query(cypher_query, {
            "paper_ids": paper_ids,
            "query": query
        })
        
        # Add relevance scores based on graph distance
        for result in results:
            result['relevance_score'] = 0.7  # Higher score for graph-refined results
        
        return results
    except Exception as e:
        logger.error(f"Graph refinement error: {e}")
        return []

async def _execute_vector_refinement(paper_ids: List[str], query: str) -> List[Dict]:
    """Refine graph results using vector similarity."""
    try:
        if not zilliz_handler.connect():
            logger.warning("Vector refinement failed: Could not connect to Zilliz")
            return []
        
        # Use paper IDs to find similar papers in vector space
        # This would require additional implementation in MilvusClient
        # For now, fall back to regular vector search
        results = zilliz_handler.search_similar_papers(
            query_text=query,
            top_k=20,
            use_hybrid=True
        )
        
        # Filter to only include results related to the paper_ids
        # This would need more sophisticated implementation
        return results or []
    except Exception as e:
        logger.error(f"Vector refinement error: {e}")
        return []

# ===============================================================================
# ANALYTICS AND MONITORING ENDPOINTS
# ===============================================================================

@app.get("/analytics/routing-performance", response_model=RoutingPerformanceResponse)
async def get_routing_performance():
    """Get routing strategy performance metrics."""
    try:
        performance_metrics = {}
        recommendations = []
        
        # Calculate performance metrics for each routing strategy
        for key, metrics in routing_engine.performance_history.items():
            if not metrics['latencies'] or not metrics['relevance_scores']:
                continue
            
            avg_latency = sum(metrics['latencies']) / len(metrics['latencies'])
            avg_relevance = sum(metrics['relevance_scores']) / len(metrics['relevance_scores'])
            
            performance_metrics[key] = {
                'average_latency_seconds': round(avg_latency, 3),
                'average_relevance_score': round(avg_relevance, 3),
                'query_count': len(metrics['latencies']),
                'efficiency_score': round(avg_relevance / max(avg_latency, 0.1), 2)
            }
        
        # Generate recommendations
        if performance_metrics:
            best_strategy = max(performance_metrics.items(), 
                              key=lambda x: x[1]['efficiency_score'])
            recommendations.append(
                f"Best performing strategy: {best_strategy[0]} "
                f"(efficiency score: {best_strategy[1]['efficiency_score']})"
            )
            
            # Find strategies with high latency
            high_latency_strategies = [
                k for k, v in performance_metrics.items() 
                if v['average_latency_seconds'] > 2.0
            ]
            if high_latency_strategies:
                recommendations.append(
                    f"Consider optimizing high-latency strategies: {', '.join(high_latency_strategies)}"
                )
        else:
            recommendations.append("Insufficient performance data. Run more hybrid searches to generate metrics.")
        
        return RoutingPerformanceResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            performance_metrics=performance_metrics,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error retrieving routing performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}")

@app.post("/analytics/query-classification", response_model=QueryAnalysisResponse)
async def analyze_query_classification(request: QueryAnalysisRequest):
    """Analyze query classification and routing suggestions."""
    try:
        # Classify the query
        query_type, confidence = routing_engine.query_classifier.classify_query(request.query)
        
        # Get routing suggestion
        mock_hybrid_request = HybridSearchRequest(
            query=request.query,
            routing_strategy=RoutingStrategy.ADAPTIVE
        )
        suggested_routing = routing_engine.decide_routing(request.query, mock_hybrid_request)
        
        # Detailed analysis
        query_lower = request.query.lower()
        analysis_details = {
            'query_length': len(request.query.split()),
            'has_semantic_keywords': any(kw in query_lower for kw in routing_engine.query_classifier.semantic_keywords),
            'has_structural_keywords': any(kw in query_lower for kw in routing_engine.query_classifier.structural_keywords),
            'has_factual_keywords': any(kw in query_lower for kw in routing_engine.query_classifier.factual_keywords),
            'complexity_estimate': 'high' if len(request.query.split()) > 10 else 'medium' if len(request.query.split()) > 5 else 'low',
            'routing_reasoning': _get_routing_reasoning(query_type, confidence)
        }
        
        return QueryAnalysisResponse(
            success=True,
            query=request.query,
            query_type=query_type,
            confidence=confidence,
            suggested_routing=suggested_routing,
            analysis_details=analysis_details
        )
        
    except Exception as e:
        logger.error(f"Error analyzing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query analysis failed: {str(e)}")

@app.get("/analytics/attribution-stats", response_model=AttributionStatsResponse)
async def get_attribution_statistics():
    """Get attribution tracking statistics."""
    try:
        # In a real implementation, these would come from a database
        # For now, return mock statistics
        
        return AttributionStatsResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            total_queries_tracked=0,  # Would track actual queries
            attribution_accuracy=0.85,  # Mock accuracy score
            high_confidence_rate=0.72,  # Mock high confidence rate
            source_type_distribution={
                'paper': 45,
                'abstract': 38,
                'citation': 17
            },
            average_attributions_per_result=2.3
        )
        
    except Exception as e:
        logger.error(f"Error retrieving attribution stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve attribution statistics: {str(e)}")

def _get_routing_reasoning(query_type: QueryType, confidence: float) -> str:
    """Generate human-readable routing reasoning."""
    if query_type == QueryType.SEMANTIC and confidence > 0.7:
        return "High semantic content detected - vector search will capture conceptual similarity effectively"
    elif query_type == QueryType.STRUCTURAL and confidence > 0.7:
        return "Structural query detected - graph search will leverage relationship patterns"
    elif query_type == QueryType.FACTUAL and confidence > 0.7:
        return "Factual query detected - graph search will provide precise entity-based results"
    elif query_type == QueryType.HYBRID:
        return "Mixed query type - parallel search will capture both semantic and structural aspects"
    else:
        return f"Uncertain classification (confidence: {confidence:.2f}) - using parallel search for comprehensive coverage"

# ===============================================================================
# GRAPH QUERY ENDPOINTS
# ===============================================================================

@app.post("/graph/query", response_model=GraphQueryResponse)
async def execute_custom_query(request: GraphQueryRequest):
    """Execute a custom Cypher query."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        
        # Add LIMIT clause if not present and limit is specified
        query = request.query.strip()
        if request.limit and not query.upper().endswith('LIMIT'):
            if not any(keyword in query.upper() for keyword in ['LIMIT', 'SKIP']):
                query += f" LIMIT {request.limit}"
        
        results = handler.execute_query(query, request.parameters)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Query executed successfully, found {len(results)} results",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "query": query,
                "parameters": request.parameters
            }
        )
        
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        raise HTTPException(status_code=400, detail=f"Query execution failed: {str(e)}")


# ===============================================================================
# MAIN ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
