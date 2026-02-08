"""
Retrieval Handler for unified search operations.

This module provides functionality to execute vector and graph searches,
with intelligent query routing and AI response generation.
"""

from typing import Dict, List, Optional
from models.entities.retrieval.QueryType import QueryType
import logging
import re
from clients.vector.MilvusClient import MilvusClient
from pymilvus import (Collection)

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for vector and graph-based retrieval operations."""

    def __init__(self, milvus_client: MilvusClient, graph_handler, llms_client, embedding_client):
        self.milvus_client = milvus_client
        self.embedding_client = embedding_client
        self.graph_handler = graph_handler
        self.llms_client = llms_client

    async def _execute_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute vector search."""
        try:
            if not self.milvus_client or not self.milvus_client.connect():
                logger.warning("Vector search failed: Could not connect to Zilliz")
                return []

            results = self.search_similar_papers(
                query_text=query,
                top_k=top_k,
                use_hybrid=True
            )
            return results or []
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []

    def search_similar_papers(self, query_text: str, top_k: int = 10, use_hybrid: bool = True) -> List[Dict]:
        """Search for similar papers using hybrid search (dense + sparse) or dense-only search.

        Args:
            query_text: Text query to search for similar papers
            top_k: Number of top results to return
            use_hybrid: Whether to use hybrid search or dense-only search

        Returns:
            List of similar papers with scores and IDs
        """
        try:
            if not self.milvus_client.collection:
                self.milvus_client.collection = Collection(self.milvus_client.config.collection_name)
                self.milvus_client.collection.load()

            # Generate embedding for the query text
            query_embedding = self.embedding_client.generate_embedding(query_text)

            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []

            if use_hybrid and self.milvus_client.is_tfidf_fitted:
                return self.milvus_client._hybrid_search(query_text, query_embedding, top_k)
            else:
                return self.milvus_client._dense_search(query_embedding, top_k)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    async def _execute_graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute graph search using Cypher query with intelligent query parsing."""
        try:
            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return []

            # Parse the query intelligently based on patterns
            cypher_query, parameters = self._build_intelligent_cypher_query(query, top_k)

            # Debug logging
            logger.info(f"Graph search query: {query}")
            logger.info(f"Generated Cypher: {cypher_query}")
            logger.info(f"Parameters: {parameters}")

            results = self.graph_handler.execute_query(cypher_query, parameters)
            logger.info(f"Graph search returned {len(results)} results")

            # Add basic relevance scores based on query type
            for result in results:
                result['relevance_score'] = 0.8 if self._is_structured_query(query) else 0.5

            return results
        except Exception as e:
            logger.error(f"Graph search error: {e}")
            return []

    def _is_structured_query(self, query: str) -> bool:
        """Check if this is a structured query (ID-based, author lookup, etc.)"""
        paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        author_patterns = [
            re.compile(r'who\s+is\s+the\s+author', re.IGNORECASE),
            re.compile(r'authors?\s+of\s+paper', re.IGNORECASE),
            re.compile(r'who\s+wrote', re.IGNORECASE),
            re.compile(r'who\s+authored', re.IGNORECASE)
        ]

        return (paper_id_pattern.search(query) or
                any(pattern.search(query) for pattern in author_patterns))

    def _is_paper_id_query(self, query: str) -> bool:
        """Check if this query contains a specific paper ID"""
        # Enhanced pattern to match various paper ID formats and contexts
        paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        return bool(paper_id_pattern.search(query))

    def _build_intelligent_cypher_query(self, query: str, top_k: int) -> tuple:
        """Build Cypher query based on intelligent query analysis."""
        query_lower = query.lower()

        # Pattern 1: "who is the author of paper have id = W2036113194"
        paper_id_match = re.search(
            r'(id\s*=\s*|with\s+id\s+|have\s+id\s+|paper\s+id\s+)(["\'?])(W\d+|DOI:[^\s]+|doi:[^\s]+)(\2)', query,
            re.IGNORECASE)
        if paper_id_match and any(word in query_lower for word in ['author', 'wrote', 'written']):
            paper_id = paper_id_match.group(3)
            cypher_query = """
            MATCH (p:Paper {id: $paper_id})
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue
            """
            return cypher_query, {"paper_id": paper_id}

        # Pattern 2: Direct paper ID query
        paper_id_match = re.search(r'\b(W\d+)\b', query)
        if paper_id_match:
            paper_id = paper_id_match.group(1)
            cypher_query = """
            MATCH (p:Paper {id: $paper_id})
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue
            """
            return cypher_query, {"paper_id": paper_id}

        # Pattern 3: Author name query
        author_query_patterns = [
            r'papers?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'authored?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'written\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)'
        ]

        for pattern in author_query_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                author_name = match.group(2).strip()
                cypher_query = """
                MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                WHERE toLower(a.name) CONTAINS toLower($author_name)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT a.name) as authors,
                       v.name as venue
                LIMIT $limit
                """
                return cypher_query, {"author_name": author_name, "limit": top_k}

        # Pattern 4: Citation-based queries
        if any(word in query_lower for word in ['cited', 'citations', 'references']):
            cypher_query = """
            MATCH (p:Paper)
            WHERE p.title CONTAINS $query OR p.abstract CONTAINS $query
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            WITH p, collect(DISTINCT a.name) as authors, v, count(cited) as citations_made
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   authors, v.name as venue, citations_made
            LIMIT $limit
            """
            return cypher_query, {"query": query, "limit": top_k}

        # Default: Generic text search with improved author retrievals
        cypher_query = """
        MATCH (p:Paper)
        WHERE p.title CONTAINS $query OR p.abstract CONTAINS $query
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
        RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
               p.doi as doi, p.publication_date as publication_date,
               collect(DISTINCT a.name) as authors,
               v.name as venue
        LIMIT $limit
        """
        return cypher_query, {"query": query, "limit": top_k}

    async def _execute_graph_refinement(self, paper_ids: List[str], query: str, top_k: int) -> List[Dict]:
        """Refine vector results using graph relationships."""
        try:
            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return []

            # Find related papers through citations and collaborations
            cypher_query = """
            MATCH (seed:Paper) 
            WHERE seed.id IN $paper_ids 
            MATCH (related:Paper)-[:CITES*0..2]-(seed) 
            RETURN DISTINCT related.id as paper_id, related.title as title, related.abstract as abstract, related.doi as doi, related.publication_date as publication_date, [(related)<-[:AUTHORED]-(a:Author) | a.name] as authors, [(related)-[:PUBLISHED_IN]->(v:Venue) | v.name][0] as venue 
            LIMIT $top_k
            """

            logger.info(paper_ids)

            results = self.graph_handler.execute_query(cypher_query, {
                "paper_ids": paper_ids,
                "top_k": top_k
            })

            # # Add relevance scores based on graph distance
            # for result in results:
            #     result['relevance_score'] = 0.7  # Higher score for graph-refined results

            return results
        except Exception as e:
            logger.error(f"Graph refinement error: {e}")
            return []

    async def _execute_vector_refinement(self, paper_ids: List[str], query: str) -> List[Dict]:
        """Refine graph results using vector similarity."""
        try:
            if not self.milvus_client or not self.milvus_client.connect():
                logger.warning("Vector refinement failed: Could not connect to Zilliz")
                return []

            # Use paper IDs to find similar papers in vector space
            # This would require additional implementation in MilvusClient
            # For now, fall back to regular vector search
            results = self.search_similar_papers(
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

    async def _generate_ai_response(
            self,
            query: str,
            search_results: List,
            query_type: QueryType
    ) -> Optional[str]:
        """Generate AI response using Llama based on search results from vector and graph searches."""
        try:
            # Use the routing engine's Llama model for response generation
            if not self.llms_client:
                logger.info("Llama not available for response generation")
                return None

            # Prepare context from search results
            context_papers = []
            for result in search_results[:5]:  # Use top 5 results
                # Handle both dict and object types
                if hasattr(result, '__dict__'):
                    # It's an object, use getattr
                    context_papers.append({
                        "title": getattr(result, "title", "") or "",
                        "authors": getattr(result, "authors", []) or [],
                        "abstract": (getattr(result, "abstract", "") or "")[:500],
                        "relevance_score": getattr(result, "relevance_score", 0),
                        "venue": getattr(result, "venue", "") or "",
                        "publication_date": getattr(result, "publication_date", "") or "",
                        "paper_id": getattr(result, "paper_id", "") or getattr(result, "id", ""),
                        "doi": getattr(result, "doi", "") or ""
                    })
                else:
                    # It's a dict, use .get()
                    context_papers.append({
                        "title": result.get("title", "") or "",
                        "authors": result.get("authors", []) or [],
                        "abstract": (result.get("abstract", "") or "")[:500],
                        "relevance_score": result.get("relevance_score", 0),
                        "venue": result.get("venue", "") or "",
                        "publication_date": result.get("publication_date", "") or "",
                        "paper_id": result.get("paper_id", "") or result.get("id", ""),
                        "doi": result.get("doi", "") or ""
                    })

            # Create specialized prompt based on query type
            if query_type == QueryType.STRUCTURAL:
                prompt = f"""
You are a helpful research assistant. Answer this question directly based on the search results provided: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers)}

Instructions:
- Answer the question directly and concisely
- Use specific information from the papers (authors, dates, titles, etc.)
- If asking about authors, list them clearly
- If asking about a specific paper, provide its details
- Be factual and precise

Answer:"""
            elif query_type == QueryType.SEMANTIC:
                prompt = f"""
You are a helpful research assistant. Answer this question comprehensively based on the search results: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers)}

Instructions:
- Provide a thorough answer to the question
- Synthesize information from multiple papers when relevant
- Explain key concepts and findings
- Mention specific papers and authors that support your answer
- Organize your response clearly with main points
- Connect different research findings when applicable

Answer:"""
            else:  # FACTUAL, HYBRID, or other types
                prompt = f"""
You are a helpful research assistant. Answer this question based on the search results provided: "{query}"

Search Results:
{self._format_papers_for_prompt(context_papers)}

Instructions:
- Answer the question directly and clearly
- Use information from the search results to support your answer
- Cite specific papers and authors when relevant
- If the question has multiple aspects, address each one
- Be accurate and only use information from the provided results
- Structure your answer logically

Answer:"""

            # Generate response using Llama
            ai_answer = self.llms_client.generate_content(
                prompt=prompt
            )

            if not ai_answer:
                logger.error("Empty response from Llama for AI generation")
                return None

            logger.info(f"Generated AI response of {len(ai_answer)} characters")
            return ai_answer

        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return None

    def _format_papers_for_prompt(self, papers: List[Dict]) -> str:
        """Format papers for inclusion in response generation."""
        formatted_papers = []

        for i, paper in enumerate(papers, 1):
            authors = paper.get("authors", [])
            authors_str = ", ".join(authors[:3]) if authors else "N/A"
            if len(authors) > 3:
                authors_str += " et al."

            paper_text = f"""{i}. Title: {paper.get('title', 'N/A')}
   Authors: {authors_str}
   Venue: {paper.get('venue', 'N/A')}
   Date: {paper.get('publication_date', 'N/A')}
   Relevance: {paper.get('relevance_score', 0):.2f}
   Abstract: {paper.get('abstract', 'N/A')}"""

            formatted_papers.append(paper_text)

        return "\n\n".join(formatted_papers)
