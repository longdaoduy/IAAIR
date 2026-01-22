"""
FastAPI application for paper ingestion pipeline.

This API provides an endpoint to:
1. Pull papers from OpenAlex
2. Enrich abstracts with Semantic Scholar
3. Upload to Neo4j
4. Upload to Zilliz
5. Return JSON file and confirmation of uploads
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
import logging

from pipelines.ingestions.handlers import IngestionHandler
from pipelines.ingestions.handlers import Neo4jHandler
from pipelines.ingestions.handlers import MilvusClient
from pipelines.ingestions.handlers.EmbeddingHandler import EmbeddingHandler
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IAAIR Paper Ingestion API",
    description="API for ingesting academic papers from OpenAlex, enriching with Semantic Scholar, and uploading to Neo4j and Zilliz",
    version="1.0.0"
)

class PaperRequest(BaseModel):
    """Request model for paper ingestion."""
    num_papers: int = Field(..., gt=0, le=1000, description="Number of papers to pull (1-1000)")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters for OpenAlex API")
    include_neo4j: bool = Field(True, description="Whether to upload to Neo4j")
    include_zilliz: bool = Field(True, description="Whether to upload to Zilliz")

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

class SearchResponse(BaseModel):
    """Response model for semantic search."""
    success: bool
    message: str
    query: str
    results_found: int
    search_time_seconds: float
    results: List[Dict[str, Any]]

# Global handlers (consider using dependency injection in production)
ingestion_handler = IngestionHandler()
neo4j_handler = Neo4jHandler()
zilliz_handler = MilvusClient()
embedding_handler = EmbeddingHandler()

@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "IAAIR Paper Ingestion API",
        "version": "1.0.0",
        "description": "API for academic paper ingestion and processing",
        "endpoints": {
            "/pull-papers": "POST - Pull papers from OpenAlex and process through pipeline",
            "/search": "POST - Semantic search for similar papers",
            "/health": "GET - Health check endpoint",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/pull-papers", response_model=PaperResponse)
async def pull_papers(request: PaperRequest, background_tasks: BackgroundTasks):
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
            # Generate embeddings for the papers
            embedding_success = await generate_and_upload_embeddings(enriched_papers, timestamp)
            zilliz_success = embedding_success
        
        # Generate summary
        total_authors = sum(len(pd['authors']) for pd in enriched_papers)
        total_citations = sum(len(pd['citations']) for pd in enriched_papers)
        
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

@app.post("/search", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """
    Perform semantic search for similar papers.
    
    This endpoint:
    1. Generates embedding for the query text
    2. Searches Zilliz for semantically similar papers
    3. Retrieves detailed information from Neo4j (if requested)
    4. Returns ranked results with similarity scores
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
            top_k=request.top_k
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
        
        # Upload embeddings using the generated embedding file
        upload_success = zilliz_handler.upload_embeddings(embedding_file=output_filename)
        
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)