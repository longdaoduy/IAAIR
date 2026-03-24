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
from pydantic import BaseModel
import logging

# Import handlers
from models.entities.ingestions.PaperRequest import PaperRequest
from models.entities.ingestions.PaperResponse import PaperResponse
from models.entities.retrievals.HybridSearchRequest import HybridSearchRequest
from models.entities.retrievals.GraphQueryRequest import GraphQueryRequest
from models.entities.retrievals.GraphQueryResponse import GraphQueryResponse

# Import evaluations components
from pipelines.evaluations.SciMMIRResultAnalyzer import SciMMIRResultAnalyzer
from models.entities.retrievals.HybridSearchResponse import HybridSearchResponse

import uvicorn
import base64
import io
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from models.engines.ServiceFactory import ServiceFactory
from pipelines.evaluations.MockDataEvaluator import MockDataEvaluator
from models.engines.ConversationMemory import ConversationMemory
from pipelines.retrievals.HybridRetrievalHandler import sanitize_query
from utils.async_utils import run_blocking
import time
from typing import Callable

# Global factory container
services = ServiceFactory()
conversation_memory = ConversationMemory()


class RequestCounterMiddleware(BaseHTTPMiddleware):
    """Lightweight middleware — actual metrics are recorded by PerformanceMonitor."""

    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load models and connect to DBs
    await services.connect_all()
    yield
    # Shutdown: Clean up resources
    await services.disconnect_all()


app = FastAPI(title="IAAIR Unified API", lifespan=lifespan)

# CORS — allow frontend and external tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request counting middleware
app.add_middleware(RequestCounterMiddleware)

# Serve frontend static files
import os

_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="frontend")


# Dependency to inject services into routes
def get_services() -> ServiceFactory:
    return services


# ===============================================================================
# ROOT ENDPOINT & FRONTEND
# ===============================================================================

@app.get("/ui")
async def serve_frontend():
    """Serve the IAAIR frontend UI."""
    index_path = os.path.join(_frontend_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/")
async def root():
    """Root endpoint providing comprehensive API information."""
    return {
        "name": "IAAIR Unified API",
        "version": "2.0.0",
        "description": "Unified API for academic paper ingestions, neo4j queries, and semantic search",
        "frontend": "Visit /ui for the web interface",
        "endpoints": {
            "ingestions": {
                "/pull-papers": "POST - Pull papers from OpenAlex and process through pipeline",
                "/download/{filename}": "GET - Download generated JSON files"
            },
            "semantic_search": {
                "/search": "POST - Semantic search for similar papers",
                "/hybrid-search": "POST - Hybrid fusion search with attribution and caching",
                "/image-search": "POST - Search figures/tables by uploading an image",
                "/image-search/base64": "POST - Search figures/tables using base64-encoded image"
            },
            "graph_queries": {
                "/neo4j/query": "POST - Execute custom Cypher queries"
            },
            "evaluations": {
                "/evaluations/comprehensive": "POST - Run comprehensive evaluations suite",
                "/evaluations/retrievals-quality": "POST - Evaluate retrievals quality with nDCG@k",
                "/evaluations/attribution-fidelity": "POST - Evaluate attribution accuracy",
                "/evaluations/verification": "POST - Run SciFact claim verification",
                "/evaluations/regression-test": "POST - Run performance regression testing",
                "/evaluations/mock-data": "POST - Evaluate system on 50-question mock dataset",
                "/evaluations/scimmir-benchmark": "POST - Run SciMMIR multi-modal benchmark evaluations"
            },
            "performance": {
                "/performance/stats": "GET - Get performance statistics and bottleneck analysis",
                "/performance/report": "GET - Export detailed performance report",
                "/performance/tune": "POST - Tune performance parameters at runtime",
                "/cache/stats": "GET - Get cache performance statistics",
                "/cache/clear": "POST - Clear system caches"
            },
            "system": {
                "/health": "GET - Health check endpoint",
                "/docs": "GET - API documentation",
                "/api/stats": "GET - Detailed API endpoint request statistics",
                "/metrics": "GET - Prometheus metrics for Grafana"
            }
        },
        "performance_optimizations": {
            "caching": "Query embeddings, search results, and AI responses cached",
            "intelligent_routing": "Smart query routing to avoid unnecessary milvus/neo4j calls",
            "selective_reranking": "Reranking only when beneficial with limited candidates",
            "optimized_search": "Tuned Milvus parameters for speed vs accuracy trade-off",
            "latency_monitoring": "Real-time performance tracking and bottleneck analysis"
        }
    }


@app.get("/api/stats")
async def api_endpoint_statistics(factory: ServiceFactory = Depends(get_services)):
    """Get Prometheus metrics summary."""
    try:
        prom = factory.performance_monitor.prometheus_integration
        if not prom:
            return {"error": "Prometheus monitoring not available"}

        # Expose the raw Prometheus text — Grafana scrapes /metrics directly
        return {
            "timestamp": datetime.now().isoformat(),
            "monitoring": {
                "status": "enabled",
                "metrics_endpoint": f"http://localhost:{prom.port}/metrics",
                "grafana_dashboard": "http://localhost:3000"
            },
            "message": "Scrape /metrics for Prometheus exposition format"
        }

    except Exception as e:
        logger.error(f"Error getting API statistics: {e}")
        return {
            "error": "Failed to retrieve API statistics",
            "details": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": "running",
            "ingestions": "available",
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
        logger.info(f"Starting paper ingestions for {request.num_papers} papers")

        # Step 1: Pull papers from OpenAlex
        logger.info("Step 1: Pulling papers from OpenAlex...")
        papers_data = factory.ingestion_handler.pull_open_alex_paper(
            count=request.num_papers,
            filters=request.filters,
            save_to_file=True,
            process_pdfs=request.process_pdfs,
            resume=request.resume_from_last
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

        logger.info(f"Paper ingestions completed successfully. Processed {len(enriched_papers)} papers")

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
        logger.error(f"Error during paper ingestions: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/ingestion/cursor-state")
async def get_cursor_state(factory: ServiceFactory = Depends(get_services)):
    """Get the current OpenAlex pagination cursor state."""
    try:
        state = factory.ingestion_handler.openalex_client._load_cursor_state()
        if state:
            return {
                "success": True,
                "has_cursor": True,
                "total_papers_fetched": state.get("total_papers_fetched", 0),
                "total_pages_fetched": state.get("total_pages_fetched", 0),
                "last_updated": state.get("last_updated"),
                "filter": state.get("filter", ""),
                "message": f"Cursor saved — next pull will continue from paper #{state.get('total_papers_fetched', 0) + 1}"
            }
        return {
            "success": True,
            "has_cursor": False,
            "message": "No cursor state saved — next pull will start from the beginning"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/ingestion/cursor-state")
async def reset_cursor_state(factory: ServiceFactory = Depends(get_services)):
    """Reset the OpenAlex cursor state to start fetching from the beginning."""
    try:
        factory.ingestion_handler.openalex_client._clear_cursor_state()
        return {
            "success": True,
            "message": "Cursor state cleared — next pull will start from the beginning"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        if not factory.milvus_client.connect():
            logger.error("Failed to connect to Zilliz")
            return False

        # Upload embeddings using the generated embedding file with papers data for hybrid search
        upload_success = factory.milvus_client.upload_embeddings(
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

@app.post("/hybrid-search", response_model=HybridSearchResponse)
async def hybrid_fusion_search(request: HybridSearchRequest, factory: ServiceFactory = Depends(get_services)):
    """
    Advanced hybrid search with fusion, reranking, attribution, and AI response generation.

    This endpoint implements:
    1. Query classification and adaptive routing with smart optimization
    2. Vector-first, neo4j-first, or parallel search strategies
    3. Result fusion with configurable weights
    4. Selective scientific domain-aware reranking (only when beneficial)
    5. Source attribution and provenance tracking
    6. AI-powered response generation with caching
    7. Comprehensive performance monitoring and caching
    """
    start_time = datetime.now()
    query_warnings = []
    try:
        # ── Input sanitization (Fix #7: Guardrails) ──
        try:
            sanitized_query, query_warnings = sanitize_query(request.query)
            if sanitized_query != request.query:
                logger.info(f"Query sanitized: {query_warnings}")
                request.query = sanitized_query
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        logger.info(f"Starting hybrid search for query: '{request.query}'")

        # Start Prometheus search tracking
        factory.performance_monitor.start_search_tracking()

        fusion_start = datetime.now()

        hybrid_results, template_info, visual_data = await factory.retrieval_handler.execute_hybrid_search(
            query=request.query,
            template_cypher=request.graph_template,
            top_k=request.top_k,
            search_strategy=request.search_strategy
        )

        logger.info(f"Template used: {template_info}")
        logger.info(f"Hybrid results (graph+vector combined): {len(hybrid_results or [])} results")

        # Visual data is collected during multi-modal vector search (SciBERT + CLIP + re-ranking)
        # and passed through for display/evidence purposes — ranking is already applied
        visual_figures = visual_data.get('figure_results', [])
        visual_tables = visual_data.get('table_results', [])
        logger.info(
            f"Cross-modal visual evidence: {len(visual_figures)} figures, "
            f"{len(visual_tables)} tables"
        )

        with factory.performance_monitor.track_operation('fusion'):
            fused_results = factory.result_fusion.fuse_results(
                hybrid_results or [],
                visual_data=visual_data
            )

        fusion_time = (datetime.now() - fusion_start).total_seconds()

        # Limit to requested number of results
        fused_results = fused_results[:request.top_k]
        # reranking_time = None
        # if request.enable_reranking and factory.scientific_reranker:
        #     reranking_start = datetime.now()
        #     # Use selective reranking with limited candidates for speed
        #     fused_results = await factory.scientific_reranker.rerank_results(
        #         fused_results, request.query,
        #         selective=True,  # Enable selective reranking
        #         max_rerank_candidates=20  # Limit candidates for speed
        #     )
        #     reranking_time = (datetime.now() - reranking_start).total_seconds()
        # Step 4: Selective reranking (only when beneficial)
        # reranking_time = None
        # if request.enable_reranking and factory.scientific_reranker:
        #     reranking_start = datetime.now()
        #     # Use selective reranking with limited candidates for speed
        #     fused_results = await factory.scientific_reranker.rerank_results(
        #         fused_results, request.query,
        #         selective=True,  # Enable selective reranking
        #         max_rerank_candidates=20  # Limit candidates for speed
        #     )
        #     reranking_time = (datetime.now() - reranking_start).total_seconds()

        # Step 5: Calculate statistics
        total_time = (datetime.now() - start_time).total_seconds()

        fusion_stats = {
            'hybrid_results_count': len(hybrid_results or []),
            'visual_figures_count': len(visual_figures),
            'visual_tables_count': len(visual_tables),
            'papers_with_visual_evidence': len(visual_data.get('paper_visual_scores', {})),
        }

        visual_stats = {
            'figures_found': len(visual_figures),
            'tables_found': len(visual_tables),
            'papers_with_visual_evidence': len(visual_data.get('paper_visual_scores', {})),
            'cross_modal_search_enabled': bool(factory.retrieval_handler.clip_client),
        }

        # attribution_stats = {
        #     'total_attributions': sum(len(r.attributions) for r in fused_results),
        #     'high_confidence_attributions': sum(
        #         1 for r in fused_results
        #         for a in r.attributions
        #         if a.confidence > factory.attribution_tracker.confidence_threshold
        #     ),
        #     'attribution_enabled': request.enable_attribution
        # }

        # Step 6: Generate AI response with caching
        ai_response = None
        response_generation_time = None
        verification_results = []
        verification_summary = {}
        verification_time = None

        # ── Conversation memory (Fix #2: Multi-turn) ──
        conversation_context = ""
        if request.session_id:
            conversation_context = conversation_memory.get_context(request.session_id)
            if conversation_context:
                logger.info(f"Injecting conversation context for session {request.session_id[:8]}...")

        if request.enable_ai_response and fused_results:
            response_start = datetime.now()

            # Check AI response cache
            results_hash = str(hash(str([r.paper_id for r in fused_results[:5]])))
            cached_response = factory.cache_manager.get_ai_response(request.query, results_hash)

            if cached_response:
                factory.performance_monitor.record_cache_hit('ai_response', True)
                ai_response = cached_response
                logger.debug("AI response cache hit")
            else:
                factory.performance_monitor.record_cache_hit('ai_response', False)
                with factory.performance_monitor.track_operation('ai_response'):
                    # 1. Generate the raw synthesis with conversation context
                    ai_response = await factory.retrieval_handler.generate_ai_response(
                        request.query, fused_results, template_info=template_info,
                        conversation_context=conversation_context,
                    )

                    # 1b. Build attributions (Fix #6) — map AI text to source papers
                    if ai_response:
                        fused_results = factory.result_fusion.build_attributions(
                            ai_response, fused_results
                        )

                    # # 2. SciFact verification — check claims against retrieved evidence
                    # if ai_response:
                    #     verification_start = datetime.now()
                    #     try:
                    #         papers_for_verification = [result.dict() for result in fused_results[:10]]
                    #         verification_results = await factory.retrieval_handler.verify_claims_scifact(
                    #             ai_response, papers_for_verification
                    #         )

                    #         # Build summary counts
                    #         label_counts = {'SUPPORTED': 0, 'CONTRADICTED': 0, 'NO_EVIDENCE': 0}
                    #         for v in verification_results:
                    #             lbl = v.get('label', 'NO_EVIDENCE')
                    #             label_counts[lbl] = label_counts.get(lbl, 0) + 1
                    #             # Push each label to Prometheus
                    #             factory.performance_monitor.record_verification_label(lbl)

                    #         total_claims = len(verification_results)
                    #         has_contradiction = label_counts['CONTRADICTED'] > 0

                    #         # Determine overall verdict
                    #         if total_claims == 0:
                    #             verdict = 'NO_CLAIMS'
                    #         elif has_contradiction:
                    #             verdict = 'CONTRADICTED'
                    #         elif label_counts['SUPPORTED'] == total_claims:
                    #             verdict = 'FULLY_SUPPORTED'
                    #         elif label_counts['SUPPORTED'] > 0:
                    #             verdict = 'PARTIALLY_SUPPORTED'
                    #         else:
                    #             verdict = 'NO_EVIDENCE'

                    #         verification_summary = {
                    #             'total_claims': total_claims,
                    #             'supported': label_counts['SUPPORTED'],
                    #             'contradicted': label_counts['CONTRADICTED'],
                    #             'no_evidence': label_counts['NO_EVIDENCE'],
                    #             'verdict': verdict,
                    #         }

                    #         logger.info(
                    #             f"Verification: {label_counts['SUPPORTED']}✓ "
                    #             f"{label_counts['CONTRADICTED']}✗ "
                    #             f"{label_counts['NO_EVIDENCE']}? → {verdict}"
                    #         )

                    #         # Only cache if no contradictions
                    #         if not has_contradiction:
                    #             factory.cache_manager.cache_ai_response(
                    #                 request.query, results_hash, ai_response
                    #             )
                    #         else:
                    #             logger.warning("AI response has contradicted claims — not cached")

                    #     except Exception as ve:
                    #         logger.error(f"Verification failed: {ve}")
                    #         verification_summary = {'error': str(ve)}

                    #     verification_time = (datetime.now() - verification_start).total_seconds()
                    #     factory.performance_monitor.record_verification_duration(verification_time)

            response_generation_time = (datetime.now() - response_start).total_seconds()

        # Finish performance tracking — push search duration, result count, template to Prometheus
        factory.performance_monitor.finish_search_tracking(
            result_count=len(fused_results),
            template_key=template_info.get('template_key'),
            search_strategy=template_info.get('search_strategy')
        )

        logger.info(f"Hybrid search completed. Found {len(fused_results)} results in {total_time:.2f}s")

        # Combine all visual results for the response
        all_visual_results = visual_figures + visual_tables

        # ── Record conversation turn (Fix #2: Multi-turn) ──
        conversation_turn = None
        if request.session_id:
            paper_ids = [r.paper_id for r in fused_results[:10]]
            conversation_memory.add_turn(
                session_id=request.session_id,
                query=request.query,
                ai_response=ai_response,
                template_key=template_info.get('template_key'),
                paper_ids=paper_ids,
            )
            turns = conversation_memory._sessions.get(request.session_id, [])
            conversation_turn = len(turns)

        # ── Attribution stats (Fix #6) ──
        attribution_stats = {}
        total_attr = sum(len(r.attributions) for r in fused_results)
        if total_attr > 0:
            high_conf = sum(
                1 for r in fused_results for a in r.attributions if a.confidence >= 0.8
            )
            papers_attributed = sum(1 for r in fused_results if r.attributions)
            attribution_stats = {
                'total_attributions': total_attr,
                'high_confidence_attributions': high_conf,
                'papers_attributed': papers_attributed,
            }

        return HybridSearchResponse(
            success=True,
            # message=f"Hybrid search completed using {routing_strategy.value} strategy",
            query=request.query,
            results_found=len(fused_results),
            search_time_seconds=total_time,
            fusion_time_seconds=fusion_time,
            # reranking_time_seconds=reranking_time,
            response_generation_time_seconds=response_generation_time,
            results=fused_results,
            ai_response=ai_response,
            graph_template_used=template_info.get('template_key') or request.graph_template or None,
            fusion_stats=fusion_stats,
            visual_results=all_visual_results,
            visual_stats=visual_stats,
            verification_results=verification_results,
            verification_summary=verification_summary,
            verification_time_seconds=verification_time,
            attribution_stats=attribution_stats,
            session_id=request.session_id,
            conversation_turn=conversation_turn,
            query_warnings=query_warnings,
        )

    except Exception as e:
        # Make sure to finish tracking even on error
        factory.performance_monitor.finish_search_tracking()
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


# # ===============================================================================
# # IMAGE SEARCH ENDPOINTS
# # ===============================================================================
#
# @app.post("/image-search")
# async def image_search(
#         file: UploadFile = File(...),
#         text_query: str = Form(None),
#         top_k: int = Form(10),
#         search_figures: bool = Form(True),
#         search_tables: bool = Form(True),
#         factory: ServiceFactory = Depends(get_services)
# ):
#     """
#     Search for similar figures and tables using an uploaded image.
#
#     This endpoint:
#     1. Accepts an uploaded image (PNG, JPG, WEBP)
#     2. Generates a CLIP embedding from the image
#     3. Searches figures and tables collections by visual similarity
#     4. Optionally combines with text query for hybrid image+text search
#     5. Returns matching figures, tables, and their parent papers
#     """
#     start_time = datetime.now()
#
#     try:
#         # Validate CLIP client
#         if not factory.clip_client:
#             raise HTTPException(status_code=503, detail="CLIP model not initialized")
#
#         # Read and validate the uploaded image
#         contents = await file.read()
#         if not contents:
#             raise HTTPException(status_code=400, detail="Empty file uploaded")
#
#         # Validate file type
#         allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
#         if file.content_type and file.content_type not in allowed_types:
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Unsupported image type: {file.content_type}. Allowed: {', '.join(allowed_types)}"
#             )
#
#         # Convert to PIL Image
#         from PIL import Image as PILImage
#         image = PILImage.open(io.BytesIO(contents)).convert("RGB")
#
#         logger.info(f"Processing image search: {file.filename} ({image.size[0]}x{image.size[1]})")
#
#         # Generate CLIP embedding (with caching)
#         image_embedding = None
#         if factory.cache_manager:
#             image_embedding = factory.cache_manager.get_image_embedding(contents)
#
#         if image_embedding is None:
#             image_embedding = await run_blocking(
#                 factory.clip_client.generate_image_embedding, image
#             )
#             if not image_embedding:
#                 raise HTTPException(status_code=500, detail="Failed to generate image embedding")
#             if factory.cache_manager:
#                 factory.cache_manager.cache_image_embedding(contents, image_embedding)
#
#         # Execute image search
#         results = await factory.retrieval_handler.search_by_image(
#             image_embedding=image_embedding,
#             top_k=top_k,
#             search_figures=search_figures,
#             search_tables=search_tables,
#             text_query=text_query
#         )
#
#         search_time = (datetime.now() - start_time).total_seconds()
#
#         figure_results = results.get("figure_results", [])
#         table_results = results.get("table_results", [])
#         related_papers = results.get("related_papers", [])
#
#         logger.info(
#             f"Image search completed: {len(figure_results)} figures, "
#             f"{len(table_results)} tables, {len(related_papers)} papers in {search_time:.2f}s"
#         )
#
#         return {
#             "success": True,
#             "message": f"Found {len(figure_results)} figures and {len(table_results)} tables from {len(related_papers)} papers",
#             "search_time_seconds": search_time,
#             "text_query": text_query,
#             "image_filename": file.filename,
#             "results_found": len(related_papers),
#             "results": related_papers,
#             "totals": {
#                 "figures": len(figure_results),
#                 "tables": len(table_results),
#                 "related_papers": len(related_papers)
#             },
#             "figure_results": figure_results,
#             "table_results": table_results
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Image search error: {e}")
#         raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
#
#
# class ImageSearchBase64Request(BaseModel):
#     image_base64: str
#     text_query: Optional[str] = None
#     query: Optional[str] = None
#     top_k: int = 10
#     search_figures: bool = True
#     search_tables: bool = True
#
#
# @app.post("/image-search/base64")
# async def image_search_base64(
#         request: ImageSearchBase64Request,
#         factory: ServiceFactory = Depends(get_services)
# ):
#     """
#     Search for similar figures and tables using a base64-encoded image.
#
#     Accepts JSON body with base64 image data — useful for frontend integration
#     where file upload forms are not convenient.
#     """
#     start_time = datetime.now()
#
#     image_base64 = request.image_base64
#     text_query = request.text_query
#     top_k = request.top_k
#     search_figures = request.search_figures
#     search_tables = request.search_tables
#
#     try:
#         if not factory.clip_client:
#             raise HTTPException(status_code=503, detail="CLIP model not initialized")
#
#         if not image_base64:
#             raise HTTPException(status_code=400, detail="image_base64 is required")
#
#         # Strip data URL prefix if present (e.g., "data:image/png;base64,...")
#         if "," in image_base64:
#             image_base64 = image_base64.split(",", 1)[1]
#
#         # Decode base64 to image
#         try:
#             image_bytes = base64.b64decode(image_base64)
#         except Exception:
#             raise HTTPException(status_code=400, detail="Invalid base64 image data")
#
#         from PIL import Image as PILImage
#         image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
#
#         logger.info(f"Processing base64 image search ({image.size[0]}x{image.size[1]})")
#
#         # Generate CLIP embedding (with caching)
#         image_embedding = None
#         if factory.cache_manager:
#             image_embedding = factory.cache_manager.get_image_embedding(image_bytes)
#
#         if image_embedding is None:
#             image_embedding = await run_blocking(
#                 factory.clip_client.generate_image_embedding, image
#             )
#             if not image_embedding:
#                 raise HTTPException(status_code=500, detail="Failed to generate image embedding")
#             if factory.cache_manager:
#                 factory.cache_manager.cache_image_embedding(image_bytes, image_embedding)
#
#         # Execute search
#         results = await factory.retrieval_handler.search_by_image(
#             image_embedding=image_embedding,
#             top_k=top_k,
#             search_figures=search_figures,
#             search_tables=search_tables,
#             text_query=text_query
#         )
#
#         search_time = (datetime.now() - start_time).total_seconds()
#
#         figure_results = results.get("figure_results", [])
#         table_results = results.get("table_results", [])
#         related_papers = results.get("related_papers", [])
#
#         # Generate AI response if query is provided
#         ai_response = None
#         response_generation_time = None
#         if request.query and request.query.strip() and related_papers:
#             response_start = datetime.now()
#             try:
#                 if factory.retrieval_handler and hasattr(factory.retrieval_handler,
#                                                          'ai_agent') and factory.retrieval_handler.ai_agent:
#                     # Build context from image search results
#                     context_parts = []
#                     # Add figure descriptions
#                     for i, fig in enumerate(figure_results[:5]):
#                         desc = fig.get('description', 'No description')
#                         score = fig.get('similarity_score', 0)
#                         context_parts.append(f"Figure {i + 1} (similarity: {score:.2f}): {desc}")
#                     # Add table descriptions
#                     for i, tbl in enumerate(table_results[:5]):
#                         desc = tbl.get('description', 'No description')
#                         score = tbl.get('similarity_score', 0)
#                         context_parts.append(f"Table {i + 1} (similarity: {score:.2f}): {desc}")
#                     # Add paper info
#                     for i, p in enumerate(related_papers[:5]):
#                         title = p.get('title', 'Untitled')
#                         abstract = (p.get('abstract', '') or '')[:300]
#                         context_parts.append(f"Paper {i + 1}: {title}\nAbstract: {abstract}")
#
#                     visual_context = "\n\n".join(context_parts)
#                     description_note = f"\nImage description provided by user: {text_query}" if text_query else ""
#
#                     prompt = f"""Based on the visual search results from academic papers, answer this question: \"{request.query}\"
# {description_note}
#
# Search Results (figures, tables, and related papers):
# {visual_context}
#
# Instructions:
# - Answer the question based on the search results above
# - Reference specific figures, tables, or papers when relevant
# - Be concise and informative (3-5 sentences)
# - If the results don't contain enough information to answer, say so
#
# Answer:"""
#
#                     system_prompt = "You are a research assistant analyzing visual search results from academic papers. Be factual, concise, and reference specific results."
#                     ai_response = factory.retrieval_handler.ai_agent.generate_content(
#                         prompt=prompt,
#                         system_prompt=system_prompt,
#                         purpose='image_search_answer'
#                     )
#                 else:
#                     logger.info("AI Agent not available for image search query response")
#             except Exception as ai_err:
#                 logger.warning(f"AI response generation failed for image search: {ai_err}")
#                 ai_response = None
#             response_generation_time = (datetime.now() - response_start).total_seconds()
#
#         search_time = (datetime.now() - start_time).total_seconds()
#
#         return {
#             "success": True,
#             "message": f"Found {len(figure_results)} figures and {len(table_results)} tables from {len(related_papers)} papers",
#             "search_time_seconds": search_time,
#             "text_query": text_query,
#             "query": request.query,
#             "ai_response": ai_response,
#             "response_generation_time_seconds": response_generation_time,
#             "results_found": len(related_papers),
#             "results": related_papers,
#             "totals": {
#                 "figures": len(figure_results),
#                 "tables": len(table_results),
#                 "related_papers": len(related_papers)
#             },
#             "figure_results": figure_results,
#             "table_results": table_results
#         }
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Base64 image search error: {e}")
#         raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


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

        results = await run_blocking(factory.neo4j_client.execute_query, query, request.parameters)

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


@app.get("/graph/ai-templates")
async def list_graph_ai_templates():
    """List AI neo4j query templates from data/graph_templates.json."""
    try:
        templates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'graph_templates.json')
        with open(templates_path, 'r', encoding='utf-8') as f:
            templates_dict = json.load(f)

        # Map of template keys to icons
        icon_map = {
            'search_by_author': '👤',
            'search_by_keywords': '🔑', 'search_by_venue': '🏛️',
            'search_by_institution': '🏫', 'search_by_year': '📅',
            'search_citations': '🔗', 'search_author_by_keywords': '👤🔑',
            'search_by_year_range': '📆', 'top_cited_papers': '🏆',
            'coauthor_network': '🤝', 'author_venue_stats': '📊',
        }

        templates = []
        for key, tpl in templates_dict.items():
            templates.append({
                'key': key,
                'name': key.replace('_', ' ').title(),
                'icon': icon_map.get(key, '📋'),
                'description': tpl.get('description', ''),
                'cypher': tpl.get('cypher', ''),
                'triggers': tpl.get('triggers', []),
            })
        return {"success": True, "templates": templates}
    except FileNotFoundError:
        return {"success": False, "templates": [], "error": "graph_templates.json not found"}
    except Exception as e:
        logger.error(f"Failed to load AI neo4j templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/templates")
async def list_cypher_templates(factory: ServiceFactory = Depends(get_services)):
    """List all saved Cypher query templates."""
    try:
        # templates = factory.mongo_client.get_cypher_templates()
        # # Serialize datetime fields
        # for t in templates:
        #     for key in ("created_at", "updated_at"):
        #         if key in t and hasattr(t[key], "isoformat"):
        #             t[key] = t[key].isoformat()
        return {"success": True, "templates": []}
    except Exception as e:
        logger.error(f"Failed to list Cypher templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CypherTemplateSaveRequest(BaseModel):
    name: str
    query: str
    icon: str = "📋"
    description: str = ""


@app.post("/graph/templates")
async def save_cypher_template(request: CypherTemplateSaveRequest,
                               factory: ServiceFactory = Depends(get_services)):
    """Save a custom Cypher query template."""
    try:
        if not request.name.strip():
            raise HTTPException(status_code=400, detail="Template name is required")
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query is required")
        # if not mongo_client:
        #     raise HTTPException(status_code=400, detail="MongoDB is required")
        result = await run_blocking(
            factory.mongo_client.save_cypher_template,
            name=request.name,
            query=request.query,
            icon=request.icon,
            description=request.description
        )
        if result:
            if "created_at" in result and hasattr(result["created_at"], "isoformat"):
                result["created_at"] = result["created_at"].isoformat()
            if "updated_at" in result and hasattr(result["updated_at"], "isoformat"):
                result["updated_at"] = result["updated_at"].isoformat()
            return {"success": True, "message": f"Template '{request.name}' saved", "template": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to save template")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save Cypher template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/graph/templates/{name}")
async def delete_cypher_template(name: str, factory: ServiceFactory = Depends(get_services)):
    """Delete a saved Cypher query template."""
    try:
        deleted = await run_blocking(factory.mongo_client.delete_cypher_template, name)
        if deleted:
            return {"success": True, "message": f"Template '{name}' deleted"}
        else:
            raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete Cypher template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ===============================================================================
# EVALUATION ENDPOINTS
# ===============================================================================
@app.post("/evaluation/mock-data")
async def evaluate_mock_data(
        limit_questions: int = None,
        save_results: bool = True,
        factory: ServiceFactory = Depends(get_services)
):
    """Evaluate system performance on mock evaluations dataset.

    This endpoint runs evaluations on the 50-question mock dataset covering:
    - 25 neo4j questions (authors, citations, venues)
    - 25 semantic questions (topics, methods, findings)

    Args:
        limit_questions: Limit number of questions to evaluate (default: all 50)
        save_results: Save detailed results and report to files
    """
    try:
        logger.info("Starting mock data evaluations")
        start_time = datetime.now()

        # Initialize evaluator
        evaluator = MockDataEvaluator(factory)

        # Run evaluations
        results = await evaluator.run_evaluation(limit=limit_questions)

        if not results:
            raise HTTPException(status_code=500, detail="No evaluations results generated")

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
                "ai_response": result.ai_response[:200] + "..." if result.ai_response and len(
                    result.ai_response) > 200 else result.ai_response,
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
                "neo4j": summary.graph_performance,
                "semantic": summary.semantic_performance
            },
            "category_breakdown": summary.category_breakdown,
            "sample_results": detailed_results,
            "report_preview": report[:1000] + "..." if len(report) > 1000 else report,
            "saved_files": saved_files
        }

    except Exception as e:
        logger.error(f"Mock data evaluations error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mock evaluations failed: {str(e)}")

@app.post("/evaluation/scimmir-benchmark")
async def run_scimmir_benchmark(
        limit_samples: int = 50,
        generate_report: bool = True,
        use_streaming: bool = False,
        use_mock: bool = False,
        factory: ServiceFactory = Depends(get_services)
):
    """Run SciMMIR multi-modal benchmark evaluations.

    Args:
        limit_samples: Number of samples to evaluate (default: 50 for quick testing)
        generate_report: Generate markdown report
        use_streaming: Use streaming mode to avoid downloading entire dataset
        use_mock: Use mock data for instant testing (no download required)
    """
    try:
        mode = "mock data" if use_mock else ("streaming" if use_streaming else "cached")
        logger.info(f"Starting SciMMIR benchmark with {limit_samples} samples using {mode}")

        # Run SciMMIR benchmark with memory-efficient options — offload to thread pool
        result = await run_blocking(
            factory.run_scimmir_benchmark_suite,
            cache_dir="./data/scimmir_cache",
            report_path="./data/scimmir_benchmark_report.md" if generate_report else None,
        )

        # Generate comparison analysis
        analyzer = SciMMIRResultAnalyzer()
        comparison = await run_blocking(analyzer.compare_with_baselines, result)

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
# PERFORMANCE & CACHE MANAGEMENT ENDPOINTS
# ===============================================================================

@app.get("/performance/stats")
async def get_performance_stats(factory: ServiceFactory = Depends(get_services)):
    """Get cache performance statistics."""
    try:
        cache_stats = factory.cache_manager.get_cache_stats()

        # Push latest cache sizes to Prometheus
        factory.performance_monitor.update_cache_metrics(cache_stats)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "cache_performance": cache_stats,
            "note": "Detailed search/AI metrics available via Prometheus at /metrics"
        }

    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance stats: {str(e)}")


@app.get("/performance/report")
async def export_performance_report(factory: ServiceFactory = Depends(get_services)):
    """Get Prometheus metrics in text format."""
    try:
        prom = factory.performance_monitor.prometheus_integration
        if not prom:
            return {"error": "Prometheus monitoring not available"}

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "metrics_text": prom.metrics.get_metrics()
        }

    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate performance report: {str(e)}")


@app.post("/cache/clear")
async def clear_caches(cache_type: str = "all", factory: ServiceFactory = Depends(get_services)):
    """Clear system caches.

    Args:
        cache_type: Type of cache to clear ("embedding", "search", "ai_response", "all")
    """
    try:
        if cache_type == "all":
            factory.cache_manager.clear_all_caches()
            message = "All caches cleared"
        elif cache_type == "embedding":
            factory.cache_manager.embedding_cache.clear()
            message = "Embedding cache cleared"
        elif cache_type == "search":
            factory.cache_manager.search_cache.clear()
            message = "Search results cache cleared"
        elif cache_type == "ai_response":
            factory.cache_manager.ai_response_cache.clear()
            message = "AI response cache cleared"
        else:
            raise HTTPException(status_code=400, detail=f"Invalid cache_type: {cache_type}")

        return {
            "success": True,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@app.get("/cache/stats")
async def get_cache_stats(factory: ServiceFactory = Depends(get_services)):
    """Get cache statistics and performance metrics."""
    try:
        stats = factory.cache_manager.get_cache_stats()

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "cache_stats": stats,
            "recommendations": [
                "Consider increasing embedding cache size" if stats['embedding_cache']['hit_rate'] < 60 else None,
                "Search cache performing well" if stats['search_cache'][
                                                      'hit_rate'] > 40 else "Consider query pattern analysis",
                "AI response cache needs attention" if stats['ai_response_cache']['hit_rate'] < 30 else None
            ]
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@app.get("/metrics")
async def get_prometheus_metrics(factory: ServiceFactory = Depends(get_services)):
    """Prometheus metrics endpoint for Grafana integration."""
    try:
        # Update cache metrics
        cache_stats = factory.cache_manager.get_cache_stats()
        factory.performance_monitor.update_cache_metrics(cache_stats)

        # Get metrics from Prometheus integration
        if factory.performance_monitor.prometheus_integration:
            metrics_output = factory.performance_monitor.prometheus_integration.metrics.get_metrics()
            return Response(
                content=metrics_output,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        else:
            raise HTTPException(status_code=503, detail="Prometheus monitoring not enabled")

    except Exception as e:
        logger.error(f"Error serving metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve metrics: {str(e)}")


# ===============================================================================
# CONVERSATION MEMORY ENDPOINTS
# ===============================================================================

@app.get("/conversation/stats")
async def get_conversation_stats():
    """Get conversation memory statistics."""
    return {
        "success": True,
        "active_sessions": conversation_memory.get_session_count(),
        "total_turns": conversation_memory.get_total_turns(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/conversation/clear")
async def clear_conversation(session_id: Optional[str] = None):
    """Clear conversation memory.

    Args:
        session_id: If provided, clears only that session. Otherwise clears all.
    """
    if session_id:
        conversation_memory.clear_session(session_id)
        return {"success": True, "message": f"Session {session_id} cleared"}
    else:
        conversation_memory.clear_all()
        return {"success": True, "message": "All conversation memory cleared"}


@app.get("/conversation/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session.

    Args:
        session_id: The session ID to retrieve history for.
    """
    turns = conversation_memory.get_history(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "turns": turns,
        "total_turns": len(turns),
    }


# ===============================================================================
# MODEL MANAGEMENT ENDPOINTS
# ===============================================================================

@app.get("/models/list")
async def list_supported_models():
    """List all supported LLM models with their specifications."""
    from models.configurators.LLMConfig import SUPPORTED_MODELS

    models = []
    for name, info in SUPPORTED_MODELS.items():
        models.append({
            "model_name": name,
            **info,
            "is_current": name == (services.llms_client.config.model_name if services.llms_client else None)
        })

    return {
        "current_model": services.llms_client.config.model_name if services.llms_client else None,
        "supported_models": models,
        "usage": {
            "env_var": "Set LLM_MODEL=<model_name> before starting the API",
            "api": "POST /models/switch with body {\"model_name\": \"<model_name>\"}"
        }
    }


@app.post("/models/switch")
async def switch_model(model_name: str, factory: ServiceFactory = Depends(get_services)):
    """Hot-swap the LLM model at runtime (downloads if not cached).

    Args:
        model_name: HuggingFace model identifier (e.g. 'Qwen/Qwen2.5-3B-Instruct')
    """
    if not factory.llms_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    old_model = factory.llms_client.config.model_name
    if old_model == model_name:
        return {"message": f"Already using {model_name}", "status": "no_change"}

    try:
        result = factory.llms_client.reload_model(model_name)
        return {
            "status": "success",
            "message": result,
            "old_model": old_model,
            "new_model": model_name,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model '{model_name}': {str(e)}")


@app.get("/models/stats")
async def get_llm_stats(factory: ServiceFactory = Depends(get_services)):
    """Get LLM usage statistics (call counts, latency per purpose)."""
    if not factory.llms_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    stats = factory.llms_client.get_llm_stats()
    stats["current_model"] = factory.llms_client.config.model_name
    return stats


# ===============================================================================
# NGROK TUNNEL SETUP
# ===============================================================================

# def start_ngrok(port: int = 8000):
#     """Start ngrok tunnel and return public URL."""
#     try:
#         from pyngrok import ngrok
#
#         ngrok.set_auth_token("3ApO0xfxobpgobKXjS3fsAiLDy7_5fyBT8kZyQUKz9Jf8HikS")
#
#         # Clean up old tunnels
#         for t in ngrok.get_tunnels():
#             ngrok.disconnect(t.public_url)
#
#         public_url = ngrok.connect(port, domain="dressily-duskish-felisha.ngrok-free.dev")
#         logger.info(f"🚀 ngrok public URL: {public_url}")
#         print(f"🚀 Public URL: {public_url}")
#         return public_url
#     except Exception as e:
#         logger.warning(f"ngrok tunnel failed: {e}")
#         print(f"⚠️  ngrok failed: {e} — running on localhost only")
#         return None


# ===============================================================================
# MAIN ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    import sys

    # Start ngrok tunnel before uvicorn
    # start_ngrok(8000)

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")