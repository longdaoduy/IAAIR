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

from typing import Dict, List
from datetime import datetime
import logging
import os
import asyncio
from fastapi.responses import FileResponse

# Import handlers
from models.entities.ingestion.PaperRequest import PaperRequest
from models.entities.ingestion.PaperResponse import PaperResponse
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.entities.retrieval.GraphQueryRequest import GraphQueryRequest
from models.entities.retrieval.GraphQueryResponse import GraphQueryResponse
from models.entities.retrieval.SearchRequest import SearchRequest

from pipelines.evaluation.SciMMIRBenchmarkIntegration import (
    SciMMIRResultAnalyzer
)
from pipelines.evaluation.MockDataEvaluator import (
    MockDataEvaluator,
    MockEvaluationResult,
    MockEvaluationSummary
)
from models.entities.retrieval.SearchResponse import SearchResponse
from models.entities.retrieval.HybridSearchResponse import HybridSearchResponse
from models.entities.retrieval.RoutingStrategy import RoutingStrategy

import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from models.engines.ServiceFactory import ServiceFactory

# Global factory container
services = ServiceFactory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and connect to DBs
    await services.connect_all()
    yield
    # Shutdown: Clean up resources
    await services.disconnect_all()


app = FastAPI(title="IAAIR Unified API", lifespan=lifespan)


# Dependency to inject services into routes
def get_services() -> ServiceFactory:
    return services


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
            "graph_queries": {
                "/graph/query": "POST - Execute custom Cypher queries"
            },
            "evaluation": {
                "/evaluation/comprehensive": "POST - Run comprehensive evaluation suite",
                "/evaluation/retrieval-quality": "POST - Evaluate retrieval quality with nDCG@k",
                "/evaluation/attribution-fidelity": "POST - Evaluate attribution accuracy",
                "/evaluation/verification": "POST - Run SciFact claim verification",
                "/evaluation/regression-test": "POST - Run performance regression testing",
                "/evaluation/mock-data": "POST - Evaluate system on 50-question mock dataset",
                "/evaluation/scimmir-benchmark": "POST - Run SciMMIR multi-modal benchmark evaluation"
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
async def pull_papers(request: PaperRequest, factory: ServiceFactory = Depends(get_services)):
    """
    Pull papers from OpenAlex, enrich with Semantic Scholar, and upload to databases.
    
    This endpoint:
    1. Fetches papers from OpenAlex API with PDF URLs and optionally processes PDFs to extract figures/tables
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
        papers_data = factory.ingestion_handler.pull_open_alex_paper(
            count=request.num_papers,
            filters=request.filters,
            save_to_file=True,
            process_pdfs=request.process_pdfs
        )

        if not papers_data:
            raise HTTPException(status_code=400, detail="Failed to fetch papers from OpenAlex")

        # Step 2: Enrich with Semantic Scholar
        logger.info("Step 2: Enriching papers with Semantic Scholar...")
        enriched_papers = factory.ingestion_handler.enrich_papers_with_semantic_scholar(
            papers_data=papers_data,
            save_to_file=True
        )

        # Step 3: Upload to Neo4j (if requested)
        neo4j_success = True
        if request.include_neo4j:
            logger.info("Step 3: Uploading to Neo4j...")
            neo4j_success = await factory.neo4j_handler.upload_papers_to_neo4j(enriched_papers)

        # Step 4: Generate embeddings and upload to Zilliz (if requested)
        zilliz_success = True
        if request.include_zilliz:
            logger.info("Step 4: Processing embeddings and uploading to Zilliz...")
            embedding_success = await generate_and_upload_embeddings(enriched_papers, timestamp, factory)
            zilliz_success = embedding_success

        # Generate summary
        total_authors = sum(len(pd.get('authors', [])) for pd in enriched_papers)
        total_citations = sum(len(pd.get('citations', [])) for pd in enriched_papers)
        total_institutions = sum(len(pd.get('institutions', [])) for pd in enriched_papers)
        total_figures = sum(len(pd.get('figures', [])) for pd in enriched_papers)
        total_tables = sum(len(pd.get('tables', [])) for pd in enriched_papers)

        summary = {
            "papers_fetched": len(enriched_papers),
            "authors_extracted": total_authors,
            "citations_extracted": total_citations,
            "institutions_extracted": total_institutions,
            "figures_extracted": total_figures,
            "tables_extracted": total_tables,
            "avg_citations_per_paper": total_citations / len(enriched_papers) if enriched_papers else 0,
            "processing_time_seconds": (datetime.now() - timestamp).total_seconds(),
            "pdf_processing_enabled": request.process_pdfs
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


async def generate_and_upload_embeddings(papers_data: List[Dict], timestamp: datetime,
                                         factory: ServiceFactory = Depends(get_services)) -> bool:
    """Generate embeddings for papers and upload to Zilliz."""
    try:
        # Create a temporary JSON file for the embedding handler
        temp_filename = f"temp_papers_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        # Save papers to temporary file
        factory.ingestion_handler.save_papers_to_json(papers_data, temp_filename)

        # Generate embeddings using the process_papers method
        output_filename = factory.embedding_handler.process_papers(input_file=temp_filename)

        if not output_filename or not os.path.exists(output_filename):
            logger.error("Failed to generate embeddings")
            return False

        # Connect to Zilliz and upload embeddings
        if not factory.vector_handler.connect():
            logger.error("Failed to connect to Zilliz")
            return False

        # Upload embeddings using the generated embedding file with papers data for hybrid search
        upload_success = factory.vector_handler.upload_embeddings(
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
async def semantic_search(request: SearchRequest, factory: ServiceFactory = Depends(get_services)):
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

        # Search for similar papers in Zilliz
        similar_papers = factory.retrieval_handler.search_similar_papers(
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
                detailed_papers = await factory.neo4j_handler.get_papers_by_ids(paper_ids)

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
async def hybrid_fusion_search(request: HybridSearchRequest, factory: ServiceFactory = Depends(get_services)):
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
        query_type, confidence = factory.routing_engine.query_classifier.classify_query(request.query)

        # Check if user specified a routing strategy explicitly
        if request.routing_strategy != RoutingStrategy.ADAPTIVE:
            # User provided specific routing strategy - use it
            routing_strategy = request.routing_strategy
            logger.info(f"Using user-specified routing strategy: {routing_strategy}")
        else:
            # Use AI routing decision for adaptive strategy
            routing_result = factory.routing_engine.decide_routing(request.query, request)

            # Handle routing result (could be tuple with 3 values)
            if isinstance(routing_result, tuple) and len(routing_result) == 3:
                routing_strategy, _, _ = routing_result
            else:
                routing_strategy = routing_result

        logger.info(
            f"Query classified as {query_type} (confidence: {confidence:.2f}), using {routing_strategy} routing")

        # Step 2: Execute search based on routing strategy
        vector_results = []
        graph_results = []

        fusion_start = datetime.now()

        if routing_strategy == RoutingStrategy.VECTOR_FIRST:
            logger.info('Vector-first')
            # Vector search first, then graph refinement
            vector_results = await factory.retrieval_handler.execute_vector_search(request.query, request.top_k * 2)
            if vector_results:
                # Use top vector results to inform graph search
                paper_ids = [r.get('paper_id') for r in vector_results[:request.top_k]]
                graph_results = await factory.retrieval_handler._execute_graph_refinement(paper_ids, request.query,
                                                                                          request.top_k)

        elif routing_strategy == RoutingStrategy.GRAPH_FIRST:
            logger.info('Graph-first')
            # Graph search first, then vector similarity
            graph_results = await factory.retrieval_handler.execute_graph_search(request.query, request.top_k * 2)

            logger.info("Paper ID query detected - using only graph search results")
            vector_results = []

        elif routing_strategy == RoutingStrategy.PARALLEL:
            logger.info('Parallel')
            # Execute both searches in parallel
            vector_task = factory.retrieval_handler.execute_vector_search(request.query, request.top_k)
            graph_task = factory.retrieval_handler.execute_graph_search(request.query, request.top_k)

            results = await asyncio.gather(vector_task, graph_task)
            # Safely unpack results with fallbacks
            vector_results = results[0] if results[0] is not None else []
            graph_results = results[1] if results[1] is not None else []

        # Step 3: Result fusion
        logger.info(
            f"Before fusion - vector_results: {len(vector_results or [])}, graph_results: {len(graph_results or [])}")
        logger.info(graph_results)
        fused_results = factory.result_fusion.fuse_results(
            vector_results or [],
            graph_results or [],
            request.fusion_weights
        )

        fusion_time = (datetime.now() - fusion_start).total_seconds()

        # Limit to requested number of results
        fused_results = fused_results[:request.top_k]

        # Step 4: Reranking (if enabled)
        reranking_time = None
        if request.enable_reranking and routing_strategy == RoutingStrategy.PARALLEL:
            reranking_start = datetime.now()
            fused_results = await factory.scientific_reranker.rerank_results(fused_results, request.query)
            reranking_time = (datetime.now() - reranking_start).total_seconds()

        # # Step 5: Attribution tracking (if enabled)
        # if request.enable_attribution and fused_results:
        #     fused_results = attribution_tracker.track_attributions(fused_results, request.query)
        #
        # Step 6: Calculate statistics
        total_time = (datetime.now() - start_time).total_seconds()

        fusion_stats = {
            'vector_results_count': len(vector_results or []),
            'graph_results_count': len(graph_results or []),
            'fusion_method': routing_strategy.value,
            'fusion_weights': request.fusion_weights or factory.result_fusion.default_weights
        }

        attribution_stats = {
            'total_attributions': sum(len(r.attributions) for r in fused_results),
            'high_confidence_attributions': sum(
                1 for r in fused_results
                for a in r.attributions
                if a.confidence > factory.attribution_tracker.confidence_threshold
            ),
            'attribution_enabled': request.enable_attribution
        }

        # Step 7: Generate AI response using Gemini
        ai_response = None
        response_generation_time = None
        if request.enable_ai_response and fused_results:
            response_start = datetime.now()
            ai_response = await factory.retrieval_handler.generate_ai_response(request.query, fused_results, query_type)
            response_generation_time = (datetime.now() - response_start).total_seconds()

        # Update routing performance tracking
        avg_relevance = sum(r.relevance_score for r in fused_results) / len(fused_results) if fused_results else 0
        factory.routing_engine.update_performance(routing_strategy, query_type, total_time, avg_relevance)

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
# GRAPH QUERY ENDPOINTS
# ===============================================================================

@app.post("/graph/query", response_model=GraphQueryResponse)
async def execute_custom_query(request: GraphQueryRequest, factory: ServiceFactory = Depends(get_services)):
    """Execute a custom Cypher query."""
    start_time = datetime.now()

    try:

        # Add LIMIT clause if not present and limit is specified
        query = request.query.strip()
        if request.limit and not query.upper().endswith('LIMIT'):
            if not any(keyword in query.upper() for keyword in ['LIMIT', 'SKIP']):
                query += f" LIMIT {request.limit}"

        results = factory.query_handler.execute_query(query, request.parameters)

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
# EVALUATION ENDPOINTS
# ===============================================================================

# @app.post("/evaluation/comprehensive")
# async def run_comprehensive_evaluation(version: str = "current", factory: ServiceFactory = Depends(get_services)):
#     """Run comprehensive evaluation across all dimensions."""
#     try:
#         logger.info(f"Starting comprehensive evaluation for version {version}")
#
#         # Initialize evaluation suite
#         eval_suite = ComprehensiveEvaluationSuite(factory)
#
#         # Run full evaluation
#         results = eval_suite.run_full_evaluation(version)
#
#         # Generate report
#         report = eval_suite.generate_evaluation_report(results)
#
#         return {
#             "success": True,
#             "version": version,
#             "timestamp": datetime.now().isoformat(),
#             "results": results,
#             "report": report,
#             "overall_score": results.get('overall_score', 0.0)
#         }
#
#     except Exception as e:
#         logger.error(f"Comprehensive evaluation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

#
# @app.post("/evaluation/retrieval-quality")
# async def evaluate_retrieval_quality(factory: ServiceFactory = Depends(get_services)):
#     """Evaluate retrieval quality using nDCG@k and other metrics."""
#     try:
#         logger.info("Starting retrieval quality evaluation")
#
#         # Initialize components
#         evaluator = RetrievalEvaluator()
#         benchmark_loader = ScientificBenchmarkLoader()
#         benchmarks = benchmark_loader.load_default_benchmarks()
#
#         # Define retrieval function
#         def retrieval_function(query_text: str):
#             return factory.retrieval_handler.search_similar_papers(
#                 query_text=query_text,
#                 top_k=20,
#                 use_hybrid=True
#             )
#
#         # Run evaluation
#         results = evaluator.evaluate_benchmark_suite(benchmarks, retrieval_function)
#
#         return {
#             "success": True,
#             "timestamp": datetime.now().isoformat(),
#             "metrics": {
#                 "avg_ndcg_at_10": results.avg_ndcg_at_10,
#                 "avg_ndcg_at_5": results.avg_ndcg_at_5,
#                 "avg_mrr": results.avg_mrr,
#                 "avg_precision_at_10": results.avg_precision_at_10,
#                 "avg_recall_at_10": results.avg_recall_at_10
#             },
#             "by_query_type": results.by_query_type,
#             "by_domain": results.by_domain,
#             "total_queries": results.total_queries
#         }
#
#     except Exception as e:
#         logger.error(f"Retrieval evaluation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Retrieval evaluation failed: {str(e)}")
#
#
# @app.post("/evaluation/attribution-fidelity")
# async def evaluate_attribution_fidelity(factory: ServiceFactory = Depends(get_services)):
#     """Evaluate attribution fidelity and exact span matching."""
#     try:
#         logger.info("Starting attribution fidelity evaluation")
#
#         # Initialize components
#         evaluator = AttributionFidelityEvaluator()
#         from pipelines.evaluation.AttributionFidelityEvaluator import AttributionBenchmarkLoader
#         benchmark_loader = AttributionBenchmarkLoader()
#         benchmarks = benchmark_loader.load_default_attribution_benchmarks()
#
#         # Generate search results with attributions
#         search_results = []
#         for benchmark in benchmarks:
#             results = factory.retrieval_handler.search_similar_papers(
#                 query_text=benchmark.query_text,
#                 top_k=10,
#                 use_hybrid=True
#             )
#
#             # Add attribution tracking
#             if results:
#                 results = factory.attribution_tracker.track_attributions(
#                     results, benchmark.query_text
#                 )
#
#             search_results.extend(results)
#
#         # Evaluate attribution quality
#         metrics = evaluator.evaluate_attribution_quality(search_results, benchmarks)
#
#         # Generate report
#         report = evaluator.create_attribution_report(metrics)
#
#         return {
#             "success": True,
#             "timestamp": datetime.now().isoformat(),
#             "metrics": {
#                 "exact_span_match_rate": metrics.exact_span_match_rate,
#                 "partial_span_match_rate": metrics.partial_span_match_rate,
#                 "citation_coverage": metrics.citation_coverage,
#                 "wrong_source_rate": metrics.wrong_source_rate,
#                 "attribution_precision": metrics.attribution_precision,
#                 "attribution_recall": metrics.attribution_recall,
#                 "average_confidence": metrics.average_confidence,
#                 "high_confidence_rate": metrics.high_confidence_rate
#             },
#             "report": report
#         }
#
#     except Exception as e:
#         logger.error(f"Attribution evaluation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Attribution evaluation failed: {str(e)}")
#
#
# @app.post("/evaluation/verification")
# async def evaluate_verification(factory: ServiceFactory = Depends(get_services)):
#     """Run SciFact-style claim verification evaluation."""
#     try:
#         logger.info("Starting scientific claim verification evaluation")
#
#         # Initialize verification pipeline
#         verification_pipeline = SciFractVerificationPipeline(
#             retrieval_client=factory.retrieval_handler,
#             ai_agent=factory.deepseek_client
#         )
#
#         from pipelines.evaluation.SciFractVerificationPipeline import (
#             create_verification_benchmarks,
#             VerificationEvaluator
#         )
#
#         # Load benchmarks
#         benchmarks = create_verification_benchmarks()
#
#         # Run verification on each benchmark
#         verification_results = []
#         for benchmark in benchmarks:
#             try:
#                 result = verification_pipeline.verify_claim(benchmark.claim)
#                 verification_results.append(result)
#             except Exception as e:
#                 logger.warning(f"Error verifying claim {benchmark.claim.claim_id}: {e}")
#                 continue
#
#         # Evaluate verification accuracy
#         evaluator = VerificationEvaluator()
#         metrics = evaluator.evaluate_verification(verification_results, benchmarks)
#
#         return {
#             "success": True,
#             "timestamp": datetime.now().isoformat(),
#             "metrics": metrics,
#             "verified_claims": len(verification_results),
#             "benchmark_claims": len(benchmarks),
#             "verification_results": [{
#                 "claim_id": result.claim.claim_id,
#                 "claim_text": result.claim.claim_text,
#                 "predicted_label": result.final_label.value,
#                 "confidence": result.confidence,
#                 "evidence_count": len(result.evidence_pieces),
#                 "reasoning": result.reasoning
#             } for result in verification_results]
#         }
#
#     except Exception as e:
#         logger.error(f"Verification evaluation error: {e}")
#         raise HTTPException(status_code=500, detail=f"Verification evaluation failed: {str(e)}")
#
#
# @app.post("/evaluation/regression-test")
# async def run_regression_test(version: str = "current", baseline: str = "latest",
#                               factory: ServiceFactory = Depends(get_services)):
#     """Run performance regression testing against baseline."""
#     try:
#         logger.info(f"Running regression test for version {version} against baseline {baseline}")
#
#         # Initialize regression tester
#         regression_tester = PerformanceRegressionTester()
#
#         # Load benchmarks for performance testing
#         benchmark_loader = ScientificBenchmarkLoader()
#         benchmarks = benchmark_loader.load_default_benchmarks()
#
#         try:
#             # Try to run regression test against existing baseline
#             result = regression_tester.run_regression_test(factory, benchmarks, baseline)
#
#             # Generate report
#             report = regression_tester.generate_regression_report(result)
#
#             return {
#                 "success": True,
#                 "timestamp": datetime.now().isoformat(),
#                 "test_passed": result.passed,
#                 "regressions": result.regressions,
#                 "improvements": result.improvements,
#                 "current_metrics": result.current_metrics,
#                 "baseline_metrics": result.baseline_metrics,
#                 "report": report
#             }
#
#         except ValueError:
#             # No baseline exists, create one
#             logger.info(f"No baseline found for {baseline}, creating new baseline")
#             baseline_result = regression_tester.capture_performance_baseline(
#                 factory, benchmarks, version
#             )
#
#             return {
#                 "success": True,
#                 "timestamp": datetime.now().isoformat(),
#                 "baseline_created": True,
#                 "baseline_version": version,
#                 "baseline_metrics": baseline_result.__dict__,
#                 "message": f"Created new performance baseline for version {version}"
#             }
#
#     except Exception as e:
#         logger.error(f"Regression test error: {e}")
#         raise HTTPException(status_code=500, detail=f"Regression test failed: {str(e)}")


@app.get("/evaluation/mock-data/preview")
async def preview_mock_data():
    """Preview the mock evaluation dataset questions."""
    try:
        from pipelines.evaluation.MockDataEvaluator import MockDataEvaluator
        
        # Create temporary evaluator just to load data
        evaluator = MockDataEvaluator(None)  # Don't need service factory for data loading
        questions = evaluator.load_mock_data()
        
        if not questions:
            raise HTTPException(status_code=404, detail="Mock data not found")
        
        # Group questions by type and category
        graph_questions = [q for q in questions if q['type'] == 'graph']
        semantic_questions = [q for q in questions if q['type'] == 'semantic']
        
        # Category breakdown
        categories = {}
        for q in questions:
            category = q['category']
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'id': q['id'],
                'question': q['question'],
                'type': q['type']
            })
        
        return {
            "success": True,
            "total_questions": len(questions),
            "breakdown": {
                "graph_questions": len(graph_questions),
                "semantic_questions": len(semantic_questions)
            },
            "categories": {
                category: len(questions) 
                for category, questions in categories.items()
            },
            "sample_questions": {
                "graph_sample": [
                    {
                        'id': q['id'],
                        'question': q['question'],
                        'category': q['category'],
                        'expected_papers': q['expected_evidence'].get('paper_ids', [])
                    }
                    for q in graph_questions[:3]
                ],
                "semantic_sample": [
                    {
                        'id': q['id'],
                        'question': q['question'],
                        'category': q['category'],
                        'expected_papers': q['expected_evidence'].get('paper_ids', [])
                    }
                    for q in semantic_questions[:3]
                ]
            }
        }
        
    except Exception as e:
        logger.error(f"Mock data preview error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to preview mock data: {str(e)}")


@app.post("/evaluation/mock-data")
async def evaluate_mock_data(
        limit_questions: int = None,
        save_results: bool = True,
        factory: ServiceFactory = Depends(get_services)
):
    """Evaluate system performance on mock evaluation dataset.
    
    This endpoint runs evaluation on the 50-question mock dataset covering:
    - 25 graph questions (authors, citations, venues)
    - 25 semantic questions (topics, methods, findings)
    
    Args:
        limit_questions: Limit number of questions to evaluate (default: all 50)
        save_results: Save detailed results and report to files
    """
    try:
        logger.info("Starting mock data evaluation")
        start_time = datetime.now()
        
        # Initialize evaluator
        evaluator = MockDataEvaluator(factory)
        
        # Run evaluation
        results = evaluator.run_evaluation(limit=limit_questions)
        
        if not results:
            raise HTTPException(status_code=500, detail="No evaluation results generated")
        
        # Generate summary
        summary = evaluator.generate_summary(results)
        
        # Save results if requested
        saved_files = {}
        if save_results:
            saved_files = evaluator.save_results(results, summary)
            logger.info(f"Saved results to: {saved_files}")
        
        # Generate report
        report = evaluator.generate_detailed_report(results, summary)
        
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        # Prepare detailed results for API response
        detailed_results = []
        for result in results[:10]:  # Limit to first 10 for API response
            detailed_results.append({
                "question_id": result.question_id,
                "question": result.question,
                "type": result.question_type,
                "category": result.category,
                "success": result.success,
                "response_time": result.response_time,
                "retrieved_count": len(result.retrieved_papers),
                "expected_count": len(result.expected_papers),
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "ai_response": result.ai_response[:200] + "..." if result.ai_response and len(result.ai_response) > 200 else result.ai_response,
                "ai_response_similarity": result.ai_response_similarity,
                "ai_generation_time": result.ai_generation_time,
                "error": result.error_message
            })
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "evaluation_time": evaluation_time,
            "summary": {
                "total_questions": summary.total_questions,
                "successful_questions": summary.successful_questions,
                "failed_questions": summary.failed_questions,
                "success_rate": summary.successful_questions / summary.total_questions * 100,
                "avg_response_time": summary.avg_response_time,
                "overall_metrics": {
                    "precision": summary.overall_precision,
                    "recall": summary.overall_recall,
                    "f1_score": summary.overall_f1
                },
                "ai_response_metrics": {
                    "success_rate": summary.ai_response_success_rate * 100,
                    "avg_generation_time": summary.avg_ai_generation_time,
                    "avg_similarity_score": summary.avg_ai_response_similarity
                }
            },
            "performance_by_type": {
                "graph": summary.graph_performance,
                "semantic": summary.semantic_performance
            },
            "category_breakdown": summary.category_breakdown,
            "sample_results": detailed_results,
            "report_preview": report[:1000] + "..." if len(report) > 1000 else report,
            "saved_files": saved_files
        }
        
    except Exception as e:
        logger.error(f"Mock data evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mock evaluation failed: {str(e)}")


@app.post("/evaluation/scimmir-benchmark")
async def run_scimmir_benchmark(
        limit_samples: int = 50,
        generate_report: bool = True,
        use_streaming: bool = False,
        use_mock: bool = False,
        factory: ServiceFactory = Depends(get_services)
):
    """Run SciMMIR multi-modal benchmark evaluation.
    
    Args:
        limit_samples: Number of samples to evaluate (default: 50 for quick testing)
        generate_report: Generate markdown report
        use_streaming: Use streaming mode to avoid downloading entire dataset
        use_mock: Use mock data for instant testing (no download required)
    """
    try:
        mode = "mock data" if use_mock else ("streaming" if use_streaming else "cached")
        logger.info(f"Starting SciMMIR benchmark with {limit_samples} samples using {mode}")

        # Run SciMMIR benchmark with memory-efficient options
        result = factory.run_scimmir_benchmark_suite(
            limit_samples=limit_samples,
            cache_dir="./data/scimmir_cache",
            report_path="./data/scimmir_benchmark_report.md" if generate_report else None,
        )

        # Generate comparison analysis
        analyzer = SciMMIRResultAnalyzer()
        comparison = analyzer.compare_with_baselines(result)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "model_name": result.model_name,
            "total_samples": result.total_samples,
            "performance": {
                "text_to_image": {
                    "mrr": round(result.text2img_mrr, 4),
                    "mrr_percentage": round(result.text2img_mrr * 100, 2),
                    "recall_at_1": round(result.text2img_recall_at_1, 4),
                    "recall_at_5": round(result.text2img_recall_at_5, 4),
                    "recall_at_10": round(result.text2img_recall_at_10, 4)
                },
                "image_to_text": {
                    "mrr": round(result.img2text_mrr, 4),
                    "mrr_percentage": round(result.img2text_mrr * 100, 2),
                    "recall_at_1": round(result.img2text_recall_at_1, 4),
                    "recall_at_5": round(result.img2text_recall_at_5, 4),
                    "recall_at_10": round(result.img2text_recall_at_10, 4)
                }
            },
            "baseline_comparison": {
                "your_rank": comparison['performance_ranking']['your_rank'],
                "total_models": comparison['performance_ranking']['total_models'],
                "percentile": round(comparison['performance_ranking']['percentile'], 1),
                "improvements": comparison['improvement_analysis']
            },
            "report_path": "./data/scimmir_benchmark_report.md" if generate_report else None
        }

    except Exception as e:
        logger.error(f"SciMMIR benchmark error: {e}")
        raise HTTPException(status_code=500, detail=f"SciMMIR benchmark failed: {str(e)}")


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
