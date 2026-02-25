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
from pipelines.retrievals.GraphQueryHandler import GraphQueryHandler
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.huggingface.DeepseekClient import DeepseekClient
from pymilvus import (Collection)

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for vector and graph-based retrieval operations."""

    def __init__(self, vector_db: Optional[MilvusClient], graph_db: GraphQueryHandler,
                 ai_agent: Optional[DeepseekClient],
                 embedder: SciBERTClient):
        self.milvus_client = vector_db
        self.embedding_client = embedder
        self.graph_handler = graph_db
        self.ai_agent = ai_agent

    async def execute_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute vector search."""
        try:
            if not self.milvus_client:
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

            if use_hybrid and self.milvus_client.is_tfidf_fitted:
                return self.milvus_client._hybrid_search(query_text, query_embedding, top_k)
            else:
                return self.milvus_client._dense_search(query_embedding, top_k)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    async def execute_graph_search(self, query: str, top_k: int) -> List[Dict]:
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

    def _extract_author_names(self, query: str) -> List[str]:
        """Extract author names from query text using AI agent for better accuracy."""
        try:
            # First try AI-powered extraction if available
            if self.ai_agent:
                ai_extracted_names = self._ai_extract_author_names(query)
                if ai_extracted_names:
                    return ai_extracted_names
        except Exception as e:
            logger.warning(f"AI author extraction failed, falling back to regex: {e}")
        
        # Fallback to regex-based extraction
        return self._regex_extract_author_names(query)

    def _ai_extract_author_names(self, query: str) -> List[str]:
        """Use AI agent to extract author names from query text."""
        prompt = f"""
Extract author names from the following query. Return only the author names, one per line, with no additional text.

Query: "{query}"

Rules:
1. Extract only proper names that appear to be human authors
2. Include full names when available (First Last, First Middle Last)
3. Handle variations like "papers by John Smith", "authored by Mary Johnson and Bob Wilson"
4. Return each name on a separate line
5. If no author names are found, return "NONE"
6. Remove any quotes or extra formatting

Author names:
"""

        try:
            response = self.ai_agent.generate_content(prompt=prompt)
            if not response or response.strip().upper() == "NONE":
                return []

            # Parse the AI response
            names = []
            lines = response.strip().split('\n')
            for line in lines:
                name = line.strip()
                # Clean up any numbering or bullet points
                name = re.sub(r'^\d+\.\s*', '', name)
                name = re.sub(r'^[-*•]\s*', '', name)
                name = name.strip('"\'')
                
                # Validate name format (at least 2 words, proper capitalization)
                if (len(name.split()) >= 2 and 
                    any(c.isupper() for c in name) and
                    not name.lower() in ['no author', 'none', 'not found']):
                    names.append(name)

            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in names:
                name_clean = name.lower().strip()
                if name_clean not in seen:
                    seen.add(name_clean)
                    unique_names.append(name)

            return unique_names

        except Exception as e:
            logger.error(f"Error in AI author extraction: {e}")
            return []

    def _regex_extract_author_names(self, query: str) -> List[str]:
        """Fallback regex-based author name extraction."""
        author_names = []

        # Pattern 1: "papers by John Smith"
        author_patterns = [
            r'papers?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'authored?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'written\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)',
            r'authors?\s+(["\']?)([^"\']+?)\1(?:\s|and\s+)',
            r'co-?authored?\s+by\s+(["\']?)([^"\']+?)\1(?:\s|$)'
        ]

        for pattern in author_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                name = match.group(2).strip()
                if len(name.split()) >= 2:  # At least first and last name
                    author_names.append(name)

        # Pattern 2: "John Smith and Mary Johnson"
        and_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+)(?:\s+and\s+([A-Z][a-z]+\s+[A-Z][a-z]+))*'
        and_matches = re.findall(and_pattern, query)
        for match_group in and_matches:
            for name in match_group:
                if name and len(name.split()) >= 2:
                    author_names.append(name.strip())

        # Remove duplicates while preserving order
        seen = set()
        unique_names = []
        for name in author_names:
            name_clean = name.lower().strip()
            if name_clean not in seen:
                seen.add(name_clean)
                unique_names.append(name)

        return unique_names

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query text."""
        # Remove common stop words and query patterns
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'about', 'papers', 'paper', 'research', 'study', 'studies', 'find', 'search',
            'what', 'where', 'when', 'who', 'how', 'why', 'which', 'is', 'are', 'was', 'were',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'related', 'concerning', 'regarding', 'involving'
        }

        # Clean and tokenize
        clean_query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = clean_query.split()

        # Filter meaningful keywords
        keywords = []
        for word in words:
            if (len(word) > 2 and
                    word not in stop_words and
                    not word.isdigit() and
                    not re.match(r'^w\d+$', word)):  # Not paper IDs
                keywords.append(word)

        # Add the original query as a keyword for exact matches
        if query.strip():
            keywords.append(query.strip())

        return list(set(keywords))  # Remove duplicates

    def _is_paper_id_query(self, query: str) -> bool:
        """Check if this query contains a specific paper ID"""
        # Enhanced pattern to match various paper ID formats and contexts
        paper_id_pattern = re.compile(r'\b(W\d+|DOI:|doi:|pmid:|arxiv:)', re.IGNORECASE)
        return bool(paper_id_pattern.search(query))

    def _build_intelligent_cypher_query(self, query: str, top_k: int) -> tuple:
        """Build Cypher query based on intelligent query analysis with support for multiple entities."""
        query_lower = query.lower()

        # Extract multiple paper IDs from query
        paper_ids = re.findall(r'\b(W\d+)\b', query)

        # Extract multiple author names from query
        author_names = self._extract_author_names(query)

        # Extract DOIs
        dois = re.findall(r'(doi:[^\s]+|DOI:[^\s]+)', query, re.IGNORECASE)

        # Pattern 1: Multiple paper IDs query
        if paper_ids:
            if any(word in query_lower for word in ['author', 'wrote', 'written']):
                # Authors of specific papers
                cypher_query = """
                MATCH (p:Paper)
                WHERE p.id IN $paper_ids
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT a.name) as authors,
                       v.name as venue
                ORDER BY p.id
                """
                return cypher_query, {"paper_ids": paper_ids}

            elif any(word in query_lower for word in ['citation', 'cite', 'cited']):
                # Citation information for papers
                cypher_query = """
                MATCH (p:Paper)
                WHERE p.id IN $paper_ids
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
                OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT a.name) as authors,
                       v.name as venue,
                       count(DISTINCT cited) as citations_made,
                       count(DISTINCT citing) as citations_received
                ORDER BY p.id
                """
                return cypher_query, {"paper_ids": paper_ids}

            else:
                # General paper information
                cypher_query = """
                MATCH (p:Paper)
                WHERE p.id IN $paper_ids
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT a.name) as authors,
                       v.name as venue
                ORDER BY p.id
                """
                return cypher_query, {"paper_ids": paper_ids}

        # Pattern 2: Multiple author names query
        if author_names:
            if any(word in query_lower for word in ['collaboration', 'coauthor', 'co-author', 'together']):
                # Collaboration between authors
                logger.info("author_names: {}".format(author_names))
                cypher_query = """
                MATCH (a1:Author)-[:AUTHORED]->(p:Paper)<-[:AUTHORED]-(a2:Author)
                WHERE any(name IN $author_names WHERE toLower(a1.name) CONTAINS toLower(name))
                  AND any(name IN $author_names WHERE toLower(a2.name) CONTAINS toLower(name))
                  AND a1 <> a2
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT a1.name + ', ' + a2.name) as collaboration_authors,
                       v.name as venue,
                       collect(DISTINCT a.name) as authors,
                LIMIT $limit
                """
                return cypher_query, {"author_names": author_names, "limit": top_k}

            else:
                # Papers by multiple authors (OR condition)
                cypher_query = """
                MATCH (a:Author)-[:AUTHORED]->(p:Paper)
                WHERE any(name IN $author_names WHERE toLower(a.name) CONTAINS toLower(name))
                OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                OPTIONAL MATCH (all_authors:Author)-[:AUTHORED]->(p)
                RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                       p.doi as doi, p.publication_date as publication_date,
                       collect(DISTINCT all_authors.name) as authors,
                       v.name as venue
                LIMIT $limit
                """
                return cypher_query, {"author_names": author_names, "limit": top_k}

        # Pattern 3: DOI-based queries
        if dois:
            cypher_query = """
            MATCH (p:Paper)
            WHERE any(doi IN $dois WHERE p.doi CONTAINS doi)
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue
            ORDER BY p.doi
            """
            return cypher_query, {"dois": dois}

        # Pattern 4: Venue-based queries
        venue_patterns = [
            r'published\s+in\s+(["\']?)([^"\']+?)\1',
            r'from\s+journal\s+(["\']?)([^"\']+?)\1',
            r'in\s+(["\']?)([A-Z][^"\']*(?:journal|review|proceedings|conference)[^"\']*?)\1'
        ]

        venues = []
        for pattern in venue_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            venues.extend([match[1].strip() for match in matches])

        if venues:
            cypher_query = """
            MATCH (p:Paper)-[:PUBLISHED_IN]->(v:Venue)
            WHERE any(venue IN $venues WHERE toLower(v.name) CONTAINS toLower(venue))
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue
            LIMIT $limit
            """
            return cypher_query, {"venues": venues, "limit": top_k}

        # Pattern 5: Citation count queries
        if any(word in query_lower for word in ['most cited', 'highest citation', 'citation count']):
            cypher_query = """
            MATCH (p:Paper)
            WHERE p.title CONTAINS $query OR p.abstract CONTAINS $query
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue,
                   count(DISTINCT citing) as citation_count
            ORDER BY citation_count DESC
            LIMIT $limit
            """
            return cypher_query, {"query": query, "limit": top_k}

        # Pattern 6: Institution/affiliation queries
        institution_patterns = [
            r'from\s+(["\']?)([^"\']*(?:university|institute|laboratory|lab|college|school)[^"\']*?)\1',
            r'at\s+(["\']?)([^"\']*(?:university|institute|laboratory|lab|college|school)[^"\']*?)\1'
        ]

        institutions = []
        for pattern in institution_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            institutions.extend([match[1].strip() for match in matches])

        if institutions:
            cypher_query = """
            MATCH (a:Author)-[:AFFILIATED_WITH]->(i:Institution)
            WHERE any(inst IN $institutions WHERE toLower(i.name) CONTAINS toLower(inst))
            MATCH (a)-[:AUTHORED]->(p:Paper)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
                   p.doi as doi, p.publication_date as publication_date,
                   collect(DISTINCT a.name) as authors,
                   v.name as venue,
                   collect(DISTINCT i.name) as institutions
            LIMIT $limit
            """
            return cypher_query, {"institutions": institutions, "limit": top_k}

        # Default: Enhanced text search with multiple keyword support
        keywords = self._extract_keywords(query)
        cypher_query = """
        MATCH (p:Paper)
        WHERE any(keyword IN $keywords WHERE 
                  p.title CONTAINS keyword OR 
                  p.abstract CONTAINS keyword OR
                  toLower(p.title) CONTAINS toLower(keyword) OR
                  toLower(p.abstract) CONTAINS toLower(keyword))
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
        RETURN p.id as paper_id, p.title as title, p.abstract as abstract,
               p.doi as doi, p.publication_date as publication_date,
               collect(DISTINCT a.name) as authors,
               v.name as venue
        LIMIT $limit
        """
        return cypher_query, {"keywords": keywords, "limit": top_k}

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

    async def generate_ai_response(
            self,
            query: str,
            search_results: List,
            query_type: QueryType
    ) -> Optional[str]:
        """Generate AI response using Llama based on search results from vector and graph searches."""
        try:
            # Use the routing engine's Llama model for response generation
            if not self.ai_agent:
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
            ai_answer = self.ai_agent.generate_content(
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
