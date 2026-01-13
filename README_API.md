# IAAIR Paper Ingestion API

FastAPI application for automated academic paper ingestion and processing pipeline.

## Overview

This API provides a complete pipeline for:

1. **Pulling papers** from OpenAlex API
2. **Enriching abstracts** with Semantic Scholar
3. **Uploading to Neo4j** graph database
4. **Generating embeddings** using SciBERT
5. **Uploading to Zilliz** vector database
6. **Returning JSON files** with processed data

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Using the startup script
./start_api.sh

# Or directly with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## API Endpoints

### `GET /`
Root endpoint providing API information and available endpoints.

### `GET /health`
Health check endpoint that tests API services and connections.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-13T10:30:00.123456",
  "services": {
    "openalex": "connected",
    "api": "running"
  }
}
```

### `POST /pull-papers`
Main endpoint for paper ingestion pipeline.

### `POST /search`
Semantic search endpoint for finding similar papers.

**Request Body:**
```json
{
  "query": "machine learning in healthcare",
  "top_k": 10,
  "include_details": true
}
```

**Parameters:**
- `query` (required): Text query to search for similar papers
- `top_k` (optional): Number of top results to return (1-50, default: 10)
- `include_details` (optional): Whether to include detailed paper information from Neo4j (default: true)

**Response:**
```json
{
  "success": true,
  "message": "Found 10 similar papers",
  "query": "machine learning in healthcare",
  "results_found": 10,
  "search_time_seconds": 0.45,
  "results": [
    {
      "paper_id": "https://openalex.org/W1234567890",
      "similarity_score": 0.892,
      "distance": 0.108,
      "title": "Deep Learning Applications in Medical Diagnosis",
      "abstract": "This paper explores...",
      "doi": "10.1000/example.doi",
      "publication_date": "2023-05-15",
      "authors": [
        {
          "id": "https://openalex.org/A1234567890",
          "name": "Dr. Jane Smith",
          "orcid": "0000-0000-0000-0000"
        }
      ],
      "venue": {
        "id": "https://openalex.org/S1234567890",
        "name": "Journal of Medical AI",
        "type": "journal"
      },
      "cited_by_count": 45,
      "citations_count": 23
    }
  ]
}
```

### `POST /pull-papers`
Main endpoint for paper ingestion pipeline.

**Request Body:**
```json
{
  "num_papers": 10,
  "filters": {
    "from_publication_date": "2023-01-01",
    "to_publication_date": "2024-12-31"
  },
  "include_neo4j": true,
  "include_zilliz": true
}
```

**Parameters:**
- `num_papers` (required): Number of papers to pull (1-1000)
- `filters` (optional): OpenAlex API filters
- `include_neo4j` (optional): Whether to upload to Neo4j (default: true)
- `include_zilliz` (optional): Whether to upload to Zilliz (default: true)

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 10 papers",
  "papers_processed": 10,
  "neo4j_uploaded": true,
  "zilliz_uploaded": true,
  "json_filename": "enriched_openalex_papers_20260113_103000.json",
  "timestamp": "2026-01-13T10:30:00.123456",
  "summary": {
    "papers_fetched": 10,
    "authors_extracted": 45,
    "citations_extracted": 120,
    "avg_citations_per_paper": 12.0,
    "processing_time_seconds": 45.6
  }
}
```

### `GET /download/{filename}`
Download generated JSON files.

**Usage:**
```bash
curl -O http://localhost:8000/download/enriched_openalex_papers_20260113_103000.json
```

## Pipeline Steps

### 1. OpenAlex Paper Fetching
- Connects to OpenAlex API
- Fetches academic papers based on specified count and filters
- Extracts paper metadata, authors, and citation information

### 2. Semantic Scholar Enrichment
- Enriches papers with additional abstract information
- Uses DOI/title matching to find corresponding entries
- Improves data quality and completeness

### 3. Neo4j Graph Upload
- Stores papers, authors, and citation relationships
- Creates graph structure for research network analysis
- Maintains referential integrity

### 4. Embedding Generation
- Uses SciBERT model to generate paper embeddings
- Processes abstracts and titles for vector representations
- Prepares data for similarity search

### 5. Zilliz Vector Upload
- Uploads embeddings to Zilliz Cloud (managed Milvus)
- Enables semantic search and similarity queries
- Configures appropriate indexing for performance

## Configuration

### Environment Variables

Create a `.env` file with required configurations:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Zilliz Configuration  
ZILLIZ_URI=your_zilliz_endpoint
ZILLIZ_TOKEN=your_zilliz_token

# OpenAlex Configuration (optional)
OPENALEX_API_KEY=your_api_key

# Semantic Scholar Configuration (optional)
SEMANTIC_SCHOLAR_API_KEY=your_api_key
```

### Handler Configuration

The API uses several handler classes that can be configured:

- `IngestionHandler`: Manages OpenAlex and Semantic Scholar integration
- `Neo4jHandler`: Handles graph database operations
- `ZillizHandler`: Manages vector database operations  
- `EmbeddingHandler`: Processes SciBERT embeddings

## Usage Examples

### Basic Paper Ingestion

```python
import requests

# Pull 50 papers and upload to both databases
response = requests.post("http://localhost:8000/pull-papers", json={
    "num_papers": 50,
    "include_neo4j": True,
    "include_zilliz": True
})

result = response.json()
print(f"Processed {result['papers_processed']} papers")
```

### Filtered Paper Search

```python
# Pull papers from specific time period with filters
response = requests.post("http://localhost:8000/pull-papers", json={
    "num_papers": 100,
    "filters": {
        "from_publication_date": "2023-01-01",
        "to_publication_date": "2024-12-31",
        "concepts.id": "C41008148"  # Computer science concept
    },
    "include_neo4j": True,
    "include_zilliz": False  # Skip vector upload
})
```

### Testing with the Test Client

```python
# Run the included test client
python test_api.py
```

### Semantic Search Examples

```python
# Basic semantic search
response = requests.post("http://localhost:8000/search", json={
    "query": "machine learning in healthcare",
    "top_k": 10,
    "include_details": True
})

results = response.json()
print(f"Found {results['results_found']} similar papers")
```

```python
# Search without detailed Neo4j data (faster)
response = requests.post("http://localhost:8000/search", json={
    "query": "neural networks for image processing",
    "top_k": 5,
    "include_details": False
})
```

```python
# Search for specific research topics
response = requests.post("http://localhost:8000/search", json={
    "query": "BERT transformer architecture natural language processing",
    "top_k": 20
})
```

## Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid parameters or failed data fetching
- **404 Not Found**: Requested file doesn't exist  
- **500 Internal Server Error**: Processing failures or database connection issues

All errors include detailed messages for debugging.

## Performance Considerations

- **Rate Limiting**: Respects OpenAlex and Semantic Scholar API limits
- **Batch Processing**: Processes papers in configurable batches
- **Memory Management**: Handles large datasets efficiently
- **Connection Pooling**: Reuses database connections
- **Background Tasks**: Long-running operations don't block responses

## Monitoring and Logging

The API includes structured logging for:
- Request/response tracking
- Pipeline step progress  
- Error diagnostics
- Performance metrics

Logs are written to stdout with INFO level by default.

## Development

### Running in Development Mode

```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Testing

```bash
# Run the test client
python test_api.py

# Or test individual endpoints
curl http://localhost:8000/health
```

## Production Deployment

For production deployment:

1. Use proper environment variables for all configurations
2. Set up reverse proxy (nginx/Apache) 
3. Configure SSL/TLS certificates
4. Set up monitoring and alerting
5. Use proper logging configuration
6. Consider container deployment with Docker

```dockerfile
# Example Dockerfile structure
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **API Connection Failures**
   - Check internet connectivity
   - Verify API keys and credentials
   - Check rate limiting

2. **Database Connection Issues**
   - Verify Neo4j/Zilliz credentials
   - Check network connectivity
   - Validate configuration

3. **Memory Issues**
   - Reduce batch size
   - Lower number of papers per request
   - Monitor system resources

4. **Slow Performance**
   - Check API rate limits
   - Optimize database queries
   - Consider async processing

### Getting Help

- Check the `/health` endpoint for service status
- Review application logs for error details
- Use the interactive documentation at `/docs`
- Test individual components with the test client