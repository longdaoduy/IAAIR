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

from typing import Dict, List, Optional
from datetime import datetime
import logging
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# Import handlers
from pipelines.ingestions import IngestionHandler
from pipelines.ingestions import GraphNeo4jHandler
from clients.vector_store.MilvusClient import MilvusClient
from pipelines.ingestions.EmbeddingSciBERTHandler import EmbeddingSciBERTHandler
from pipelines.retrievals.RetrievalHandler import RetrievalHandler
from pipelines.retrievals.GraphQueryHandler import GraphQueryHandler
from models.engines.RoutingDecisionEngine import RoutingDecisionEngine
from models.engines.ResultFusion import ResultFusion
from models.engines.ScientificReranker import ScientificReranker
from models.engines.AttributionTracker import AttributionTracker
from models.entities.ingestion.PaperRequest import PaperRequest
from models.entities.ingestion.PaperResponse import PaperResponse
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.GraphQueryRequest import GraphQueryRequest
from models.entities.retrieval.GraphQueryResponse import GraphQueryResponse
from models.entities.retrieval.SearchRequest import SearchRequest
from models.entities.retrieval.SearchResponse import SearchResponse
from models.entities.retrieval.HybridSearchResponse import HybridSearchResponse
from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.RoutingPerformanceResponse import RoutingPerformanceResponse
from models.entities.retrieval.QueryAnalysisResponse import QueryAnalysisResponse
from models.entities.retrieval.QueryAnalysisRequest import QueryAnalysisRequest
from models.entities.retrieval.QueryType import QueryType
from models.entities.retrieval.AttributionStatsResponse import AttributionStatsResponse

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

# Initialize handlers
ingestion_handler = IngestionHandler()
neo4j_handler = GraphNeo4jHandler()
vector_handler = MilvusClient()
embedding_handler = EmbeddingSciBERTHandler()
query_handler = None

def get_query_handler():
    """Get or create graph query handler."""
    global query_handler
    if query_handler is None:
        query_handler = GraphQueryHandler()
    return query_handler

# Global hybrid system components
routing_engine = RoutingDecisionEngine()
result_fusion = ResultFusion()
scientific_reranker = ScientificReranker()
attribution_tracker = AttributionTracker()
retrieval_handler = RetrievalHandler()

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
        if not vector_handler.connect():
            logger.error("Failed to connect to Zilliz")
            return False
        
        # Upload embeddings using the generated embedding file with papers data for hybrid search
        upload_success = vector_handler.upload_embeddings(
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
        if not vector_handler.connect():
            raise HTTPException(status_code=500, detail="Failed to connect to Zilliz vector database")
        
        # Search for similar papers in Zilliz
        similar_papers = vector_handler.search_similar_papers(
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
    Advanced hybrid search with fusion, reranking, attribution, and AI response generation.
    
    This endpoint implements:
    1. Query classification and adaptive routing
    2. Vector-first, graph-first, or parallel search strategies
    3. Result fusion with configurable weights
    4. Scientific domain-aware reranking
    5. Source attribution and provenance tracking
    6. AI-powered response generation using Gemini
    """
    start_time = datetime.now()
    
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
            vector_results = await retrieval_handler._execute_vector_search(request.query, request.top_k * 2)
            if vector_results:
                # Use top vector results to inform graph search
                paper_ids = [r.get('paper_id') for r in vector_results[:request.top_k]]
                graph_results = await retrieval_handler._execute_graph_refinement(paper_ids, request.query)
            
        elif routing_strategy == RoutingStrategy.GRAPH_FIRST:
            # Graph search first, then vector similarity
            graph_results = await retrieval_handler._execute_graph_search(request.query, request.top_k * 2)
            
            # Check if this is a specific paper ID query BEFORE checking if graph_results exist
            if retrieval_handler._is_paper_id_query(request.query):
                # For paper ID queries, return only the graph results, no vector search
                logger.info("Paper ID query detected - using only graph search results")
                vector_results = []
                if graph_results:
                    graph_results = graph_results[:1]  # Limit to single exact match
                    logger.info(f"GRAPH_FIRST paper ID: graph_results count: {len(graph_results)}, vector_results count: {len(vector_results)}")
                else:
                    logger.warning("Graph search returned no results for paper ID query")
            elif graph_results and not retrieval_handler._is_paper_id_query(request.query):
                # Use graph results to inform vector search for general queries
                paper_ids = [r.get('paper_id', r.get('id')) for r in graph_results[:request.top_k]]
                vector_results = await retrieval_handler._execute_vector_refinement(paper_ids, request.query)
            else:
                vector_results = []
            
        elif routing_strategy == RoutingStrategy.PARALLEL:
            # Execute both searches in parallel
            vector_task = retrieval_handler._execute_vector_search(request.query, request.top_k)
            graph_task = retrieval_handler._execute_graph_search(request.query, request.top_k)
            
            vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
        
        # Step 3: Result fusion
        logger.info(f"Before fusion - vector_results: {len(vector_results or [])}, graph_results: {len(graph_results or [])}")
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
        
        # Step 7: Generate AI response using Gemini
        ai_response = None
        response_generation_time = None
        if request.enable_ai_response and fused_results:
            response_start = datetime.now()
            ai_response = await retrieval_handler._generate_ai_response(request.query, fused_results, query_type)
            response_generation_time = (datetime.now() - response_start).total_seconds()
        
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
            response_generation_time_seconds=response_generation_time,
            results=fused_results,
            ai_response=ai_response,
            fusion_stats=fusion_stats,
            attribution_stats=attribution_stats
        )
        
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")

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
