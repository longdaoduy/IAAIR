# Cypher Subgraph API Documentation

## Overview

The Cypher Subgraph API provides RESTful endpoints for querying the Neo4j academic paper knowledge graph using Cypher queries. This API serves as the foundation for hybrid graph-vector retrieval systems and enables complex graph-based paper discovery.

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │ GraphQueryHandler│    │     Neo4j       │
│   Endpoints     │───▶│     Module       │───▶│   Database      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
    REST Requests           Cypher Queries          Graph Data
    JSON Responses          Result Processing       Relationships
```

## Components

### 1. GraphQueryHandler
**File**: `pipelines/retrieval/GraphQueryHandler.py`

Core handler for Neo4j connections and Cypher query execution:

- **Connection Management**: Handles Neo4j driver lifecycle
- **Query Execution**: Executes Cypher queries with error handling
- **Result Processing**: Converts Neo4j records to JSON format
- **Statistics**: Database statistics and health monitoring

### 2. CypherQueryBuilder
**File**: `pipelines/retrieval/GraphQueryHandler.py`

Pre-built Cypher queries for common use cases:

- Papers by author
- Citation relationships
- Coauthor networks
- Venue-based queries  
- Research trend analysis
- Citation path finding

### 3. CypherSubgraphAPI
**File**: `pipelines/retrieval/CypherSubgraphAPI.py`

FastAPI router with RESTful endpoints:

- Request/response models
- Parameter validation
- Error handling
- Query performance monitoring

## API Endpoints

### Base URL: `/graph`

#### 1. Database Statistics
```http
GET /graph/stats
```

**Response:**
```json
{
  "success": true,
  "timestamp": "2026-01-22T10:30:00",
  "stats": {
    "total_nodes": 50000,
    "total_relationships": 150000,
    "paper_count": 25000,
    "author_count": 15000,
    "venue_count": 2000,
    "citation_count": 100000
  }
}
```

#### 2. Custom Cypher Query
```http
POST /graph/query
```

**Request:**
```json
{
  "query": "MATCH (p:Paper) WHERE p.cited_by_count > $min_citations RETURN p.title, p.cited_by_count ORDER BY p.cited_by_count DESC LIMIT 5",
  "parameters": {"min_citations": 100},
  "limit": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Query executed successfully, found 5 results",
  "query_time_seconds": 0.234,
  "results_count": 5,
  "results": [
    {
      "p.title": "Deep Learning for Natural Language Processing",
      "p.cited_by_count": 1250
    }
  ]
}
```

#### 3. Papers by Author
```http
POST /graph/papers/by-author
```

**Request:**
```json
{
  "author_name": "John Smith",
  "limit": 10
}
```

**Use Case:** Find all papers authored by a specific researcher.

#### 4. Paper Citations
```http
POST /graph/papers/citations
```

**Request:**
```json
{
  "paper_id": "paper123",
  "direction": "citing",
  "limit": 10
}
```

**Directions:**
- `"citing"`: Papers that cite the target paper
- `"cited"`: Papers cited by the target paper

#### 5. Author Coauthors
```http
POST /graph/authors/coauthors
```

**Request:**
```json
{
  "author_name": "Jane Doe",
  "limit": 10
}
```

**Response includes:**
- Coauthor names
- Collaboration counts
- Sample collaborative papers

#### 6. Papers in Venue
```http
POST /graph/venues/papers
```

**Request:**
```json
{
  "venue_name": "Nature",
  "limit": 10
}
```

**Use Case:** Find papers published in specific journals or conferences.

#### 7. Most Cited Papers
```http
GET /graph/papers/most-cited?limit=10&min_citations=50
```

**Parameters:**
- `limit`: Maximum papers to return (1-100)
- `min_citations`: Minimum citation threshold

#### 8. Collaboration Network
```http
POST /graph/authors/network
```

**Request:**
```json
{
  "author_name": "Alice Johnson",
  "depth": 2,
  "limit": 20
}
```

**Network Depth:**
- `1`: Direct collaborators only
- `2`: Collaborators + their collaborators  
- `3`: Extended network

#### 9. Research Trends
```http
POST /graph/analysis/trends
```

**Request:**
```json
{
  "start_year": 2020,
  "end_year": 2024
}
```

**Analysis includes:**
- Papers per year
- Average citations
- Venue distribution
- Sample paper titles

#### 10. Citation Paths
```http
POST /graph/papers/citation-path
```

**Request:**
```json
{
  "source_paper_id": "paper123",
  "target_paper_id": "paper456", 
  "max_depth": 3
}
```

**Use Case:** Find citation chains between two papers (how one paper influences another through citations).

## Query Examples

### Find Papers by Specific Author
```cypher
MATCH (a:Author)-[:Authored]->(p:Paper)
WHERE toLower(a.name) CONTAINS toLower("einstein")
RETURN a.name, p.title, p.cited_by_count
ORDER BY p.cited_by_count DESC
LIMIT 10
```

### Find Citation Network
```cypher
MATCH (p1:Paper {id: "paper123"})-[:CitedBy*1..2]->(p2:Paper)
RETURN p1.title as source, p2.title as target, 
       p2.cited_by_count as target_citations
ORDER BY target_citations DESC
LIMIT 20
```

### Find Research Collaboration Patterns
```cypher
MATCH (a1:Author)-[:Authored]->(p:Paper)<-[:Authored]-(a2:Author)
WHERE a1.name CONTAINS "Smith" AND a1 <> a2
RETURN a2.name as coauthor, count(p) as collaborations,
       collect(p.title)[0..3] as sample_papers
ORDER BY collaborations DESC
LIMIT 10
```

### Analyze Venue Impact
```cypher
MATCH (p:Paper)-[:PublishedIn]->(v:Venue)
WITH v, count(p) as paper_count, avg(p.cited_by_count) as avg_citations
WHERE paper_count >= 5
RETURN v.name, v.type, paper_count, round(avg_citations, 2) as avg_cites
ORDER BY avg_citations DESC
LIMIT 15
```

## Integration with IAAIR System

### 1. Hybrid Search Foundation
The Cypher API provides graph-based retrieval that complements vector search:

```python
# Pseudo-code for hybrid retrievals
def hybrid_search(query):
    # Vector search for semantic similarity
    vector_results = vector_search(query)
    
    # Graph search for related papers
    graph_results = []
    for paper in vector_results:
        citations = cypher_api.find_paper_citations(paper.id)
        graph_results.extend(citations)
    
    # Combine and rank results
    return combine_results(vector_results, graph_results)
```

### 2. Author-Centric Search
```python
def find_author_research_domain(author_name):
    # Get author's papers
    papers = cypher_api.find_papers_by_author(author_name)
    
    # Get coauthors and their expertise
    coauthors = cypher_api.find_coauthors(author_name)
    
    # Analyze citation patterns
    citations = []
    for paper in papers:
        citations.extend(cypher_api.find_paper_citations(paper.id))
    
    return analyze_research_domain(papers, coauthors, citations)
```

### 3. Citation-Based Recommendations
```python
def recommend_papers(paper_id):
    # Find papers that cite this paper (similar research)
    citing_papers = cypher_api.find_papers_citing_paper(paper_id)
    
    # Find papers cited by this paper (background research)  
    cited_papers = cypher_api.find_papers_cited_by_paper(paper_id)
    
    # Find citation paths to highly cited papers
    citation_paths = cypher_api.find_citation_paths(paper_id, top_cited_papers)
    
    return rank_recommendations(citing_papers, cited_papers, citation_paths)
```

## Performance Considerations

### Query Optimization
1. **Indexing**: Ensure Neo4j has appropriate indexes on commonly queried fields
2. **Limiting**: Always use `LIMIT` clauses to prevent large result sets
3. **Parameters**: Use parameterized queries to enable query plan caching
4. **Monitoring**: Track query performance via `query_time_seconds` in responses

### Recommended Neo4j Indexes
```cypher
-- Author name index
CREATE INDEX author_name_index FOR (a:Author) ON (a.name)

-- Paper title and ID indexes  
CREATE INDEX paper_title_index FOR (p:Paper) ON (p.title)
CREATE INDEX paper_id_index FOR (p:Paper) ON (p.id)

-- Citation count index for ranking
CREATE INDEX paper_citations_index FOR (p:Paper) ON (p.cited_by_count)

-- Publication date index for temporal queries
CREATE INDEX paper_date_index FOR (p:Paper) ON (p.publication_date)

-- Venue name index
CREATE INDEX venue_name_index FOR (v:Venue) ON (v.name)
```

### Connection Pooling
The API uses Neo4j connection pooling:
- **Max connections**: 50 per pool
- **Connection lifetime**: 3600 seconds  
- **Acquisition timeout**: 60 seconds

## Error Handling

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request - Invalid Cypher syntax | Check query syntax and parameters |
| 500 | Internal Server Error - Database connection | Verify Neo4j is running and accessible |
| 422 | Validation Error - Invalid request format | Check request schema matches API docs |

### Example Error Response
```json
{
  "detail": "Query execution failed: Invalid syntax near 'INVALID'"
}
```

## Testing

Run the test suite to validate API functionality:

```bash
python test_cypher_api.py
```

**Test Coverage:**
- Database connectivity
- All endpoint functionality  
- Query performance
- Error handling
- Response format validation

## Next Steps for Hybrid Fusion

### Week 5 Implementation Plan

1. **Graph ↔ Vector Routing Logic**
   ```python
   def route_query(query, context):
       if is_author_query(query):
           return graph_search(query)
       elif is_semantic_query(query):
           return vector_search(query)  
       else:
           return hybrid_search(query)
   ```

2. **Result Fusion Strategies**
   - **RRF (Reciprocal Rank Fusion)**: Combine graph and vector rankings
   - **Score normalization**: Align different scoring systems
   - **Context-aware weighting**: Adjust based on query type

3. **Performance Optimization**
   - **Parallel execution**: Run graph and vector searches concurrently
   - **Result caching**: Cache common graph queries
   - **Smart routing**: Learn optimal routing strategies

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

The Cypher Subgraph API is now ready to serve as the foundation for sophisticated hybrid graph-vector retrieval systems in the IAAIR platform! 🚀