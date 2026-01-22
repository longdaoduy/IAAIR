"""
Cypher Subgraph API for Neo4j graph queries.

This module provides RESTful API endpoints for querying the academic paper
knowledge graph using Cypher queries. It serves as the foundation for
hybrid graph-vector retrieval systems.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging
from fastapi import APIRouter, HTTPException, Query, Path
from pydantic import BaseModel, Field, validator

from pipelines.retrieval.GraphQueryHandler import GraphQueryHandler, CypherQueryBuilder


# Pydantic models for request/response
class GraphQueryRequest(BaseModel):
    """Request model for custom Cypher queries."""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum results to return")

class AuthorPapersRequest(BaseModel):
    """Request model for finding papers by author."""
    author_name: str = Field(..., min_length=1, description="Author name to search for")
    limit: int = Field(10, ge=1, le=100, description="Maximum papers to return")

class PaperCitationsRequest(BaseModel):
    """Request model for finding paper citations."""
    paper_id: str = Field(..., description="Paper ID to find citations for")
    direction: str = Field("citing", pattern="^(citing|cited)$", description="Direction: 'citing' or 'cited'")
    limit: int = Field(10, ge=1, le=100, description="Maximum citations to return")

class CoauthorsRequest(BaseModel):
    """Request model for finding coauthors."""
    author_name: str = Field(..., min_length=1, description="Author name to find coauthors for")
    limit: int = Field(10, ge=1, le=50, description="Maximum coauthors to return")

class VenuePapersRequest(BaseModel):
    """Request model for finding papers in venue."""
    venue_name: str = Field(..., min_length=1, description="Venue name to search for")
    limit: int = Field(10, ge=1, le=100, description="Maximum papers to return")

class CitationPathRequest(BaseModel):
    """Request model for finding citation paths."""
    source_paper_id: str = Field(..., description="Source paper ID")
    target_paper_id: str = Field(..., description="Target paper ID")
    max_depth: int = Field(3, ge=1, le=5, description="Maximum path depth")

class TrendAnalysisRequest(BaseModel):
    """Request model for research trend analysis."""
    start_year: int = Field(..., ge=1900, le=2030, description="Start year for analysis")
    end_year: int = Field(..., ge=1900, le=2030, description="End year for analysis")
    
    @validator('end_year')
    def end_year_must_be_after_start(cls, v, values):
        if 'start_year' in values and v < values['start_year']:
            raise ValueError('end_year must be greater than or equal to start_year')
        return v

class GraphQueryResponse(BaseModel):
    """Response model for graph queries."""
    success: bool
    message: str
    query_time_seconds: float
    results_count: int
    results: List[Dict[str, Any]]
    query_info: Optional[Dict[str, Any]] = None

class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics."""
    success: bool
    timestamp: str
    stats: Dict[str, int]


# Create router
router = APIRouter(prefix="/graph", tags=["Graph Queries"])

# Global query handler
query_handler = None

def get_query_handler():
    """Get or create graph query handler."""
    global query_handler
    if query_handler is None:
        query_handler = GraphQueryHandler()
    return query_handler

@router.get("/stats", response_model=DatabaseStatsResponse)
async def get_database_statistics():
    """Get database statistics including node and relationship counts."""
    try:
        start_time = datetime.now()
        handler = get_query_handler()
        
        stats = handler.get_database_stats()
        
        return DatabaseStatsResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            stats=stats
        )
        
    except Exception as e:
        logging.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve database stats: {str(e)}")

@router.post("/query", response_model=GraphQueryResponse)
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
        logging.error(f"Query execution error: {e}")
        raise HTTPException(status_code=400, detail=f"Query execution failed: {str(e)}")

@router.post("/papers/by-author", response_model=GraphQueryResponse)
async def find_papers_by_author(request: AuthorPapersRequest):
    """Find papers authored by a specific author."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_papers_by_author(request.author_name, request.limit)
        
        results = handler.execute_query(query, {"author_name": request.author_name})
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} papers by author '{request.author_name}'",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "author_searched": request.author_name,
                "search_type": "papers_by_author"
            }
        )
        
    except Exception as e:
        logging.error(f"Author papers query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find papers by author: {str(e)}")

@router.post("/papers/citations", response_model=GraphQueryResponse)
async def find_paper_citations(request: PaperCitationsRequest):
    """Find papers that cite or are cited by a specific paper."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        
        if request.direction == "citing":
            query = CypherQueryBuilder.find_papers_citing_paper(request.paper_id, request.limit)
            message_prefix = "citing"
        else:
            query = CypherQueryBuilder.find_papers_cited_by_paper(request.paper_id, request.limit)
            message_prefix = "cited by"
        
        results = handler.execute_query(query, {"paper_id": request.paper_id})
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} papers {message_prefix} paper '{request.paper_id}'",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "paper_id": request.paper_id,
                "direction": request.direction,
                "search_type": "paper_citations"
            }
        )
        
    except Exception as e:
        logging.error(f"Paper citations query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find paper citations: {str(e)}")

@router.post("/authors/coauthors", response_model=GraphQueryResponse)
async def find_coauthors(request: CoauthorsRequest):
    """Find coauthors of a specific author."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_coauthors(request.author_name, request.limit)
        
        results = handler.execute_query(query, {"author_name": request.author_name})
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} coauthors for '{request.author_name}'",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "author_searched": request.author_name,
                "search_type": "coauthors"
            }
        )
        
    except Exception as e:
        logging.error(f"Coauthors query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find coauthors: {str(e)}")

@router.post("/venues/papers", response_model=GraphQueryResponse)
async def find_papers_in_venue(request: VenuePapersRequest):
    """Find papers published in a specific venue."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_papers_in_venue(request.venue_name, request.limit)
        
        results = handler.execute_query(query, {"venue_name": request.venue_name})
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} papers in venue '{request.venue_name}'",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "venue_searched": request.venue_name,
                "search_type": "papers_in_venue"
            }
        )
        
    except Exception as e:
        logging.error(f"Venue papers query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find papers in venue: {str(e)}")

@router.get("/papers/most-cited", response_model=GraphQueryResponse)
async def find_most_cited_papers(
    limit: int = Query(10, ge=1, le=100, description="Maximum papers to return"),
    min_citations: int = Query(0, ge=0, description="Minimum citation count")
):
    """Find most cited papers in the database."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_most_cited_papers(limit, min_citations)
        
        results = handler.execute_query(query)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} most cited papers (min citations: {min_citations})",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "min_citations": min_citations,
                "search_type": "most_cited_papers"
            }
        )
        
    except Exception as e:
        logging.error(f"Most cited papers query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find most cited papers: {str(e)}")

@router.post("/authors/network", response_model=GraphQueryResponse)
async def find_collaboration_network(
    author_name: str = Field(..., description="Author name for network center"),
    depth: int = Query(2, ge=1, le=3, description="Network depth"),
    limit: int = Query(20, ge=1, le=100, description="Maximum nodes to return")
):
    """Find author collaboration network."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_author_collaboration_network(author_name, depth, limit)
        
        results = handler.execute_query(query, {"author_name": author_name})
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} nodes in collaboration network for '{author_name}'",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "central_author": author_name,
                "network_depth": depth,
                "search_type": "collaboration_network"
            }
        )
        
    except Exception as e:
        logging.error(f"Collaboration network query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find collaboration network: {str(e)}")

@router.post("/analysis/trends", response_model=GraphQueryResponse)
async def analyze_research_trends(request: TrendAnalysisRequest):
    """Analyze research publication trends by year."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_research_trends_by_year(request.start_year, request.end_year)
        
        results = handler.execute_query(query)
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found research trends for years {request.start_year}-{request.end_year}",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "start_year": request.start_year,
                "end_year": request.end_year,
                "search_type": "research_trends"
            }
        )
        
    except Exception as e:
        logging.error(f"Research trends query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to analyze research trends: {str(e)}")

@router.post("/papers/citation-path", response_model=GraphQueryResponse)
async def find_citation_path(request: CitationPathRequest):
    """Find citation paths between two papers."""
    start_time = datetime.now()
    
    try:
        handler = get_query_handler()
        query = CypherQueryBuilder.find_citation_path(
            request.source_paper_id, 
            request.target_paper_id, 
            request.max_depth
        )
        
        results = handler.execute_query(query, {
            "source_paper_id": request.source_paper_id,
            "target_paper_id": request.target_paper_id
        })
        
        query_time = (datetime.now() - start_time).total_seconds()
        
        return GraphQueryResponse(
            success=True,
            message=f"Found {len(results)} citation paths between papers",
            query_time_seconds=query_time,
            results_count=len(results),
            results=results,
            query_info={
                "source_paper_id": request.source_paper_id,
                "target_paper_id": request.target_paper_id,
                "max_depth": request.max_depth,
                "search_type": "citation_path"
            }
        )
        
    except Exception as e:
        logging.error(f"Citation path query error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to find citation path: {str(e)}")

# Cleanup handler
@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup database connections on shutdown."""
    global query_handler
    if query_handler:
        query_handler.close()
        query_handler = None