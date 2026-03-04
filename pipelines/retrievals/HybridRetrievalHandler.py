"""
Retrieval Handler for unified search operations.

This module provides functionality to execute vector and graph searches,
with intelligent query routing and AI response generation.
"""

from typing import Dict, List, Optional
from models.entities.retrieval.QueryType import QueryType
import logging
import re
import json
from clients.vector.MilvusClient import MilvusClient
from pipelines.retrievals.GraphQueryHandler import GraphQueryHandler
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.huggingface.DeepseekClient import DeepseekClient
from pymilvus import (Collection)

logger = logging.getLogger(__name__)


class HybridRetrievalHandler:
    """Unified handler for vector and graph-based retrieval operations."""

    def __init__(self, vector_db: Optional[MilvusClient], graph_db: GraphQueryHandler,
                 ai_agent: Optional[DeepseekClient], embedder: SciBERTClient,
                 cache_manager=None, performance_monitor=None):
        self.milvus_client = vector_db
        self.embedding_client = embedder
        self.graph_handler = graph_db
        self.ai_agent = ai_agent
        self.cache_manager = cache_manager
        self.performance_monitor = performance_monitor

    async def execute_vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute vector search with performance tracking."""
        if self.performance_monitor:
            with self.performance_monitor.track_operation('vector_search'):
                return await self._execute_vector_search_internal(query, top_k)
        else:
            return await self._execute_vector_search_internal(query, top_k)

    async def _execute_vector_search_internal(self, query: str, top_k: int) -> List[Dict]:
        """Internal vector search implementation."""
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=True, routing_strategy="vector"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('vector', len(cached_results))
                    logger.debug(f"Vector search cache hit for: {query[:50]}...")
                    return cached_results

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

            results = self.search_similar_papers(
                query_text=query,
                top_k=top_k,
                use_hybrid=True
            )

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=True, routing_strategy="vector"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('vector', len(results or []))

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

            # Check embedding cache first
            query_embedding = None
            if self.cache_manager:
                query_embedding = self.cache_manager.get_embedding(query_text)
                if query_embedding is not None and self.performance_monitor:
                    self.performance_monitor.record_cache_hit('embedding', True)

            # Generate embedding if not cached
            if query_embedding is None:
                if self.performance_monitor:
                    self.performance_monitor.record_cache_hit('embedding', False)
                    with self.performance_monitor.track_operation('embedding'):
                        query_embedding = self.embedding_client.generate_embedding(query_text)
                else:
                    query_embedding = self.embedding_client.generate_embedding(query_text)

                # Cache the embedding
                if self.cache_manager and query_embedding is not None:
                    self.cache_manager.cache_embedding(query_text, query_embedding)

            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []

            # Execute search with optimized parameters
            if use_hybrid and self.milvus_client.is_tfidf_fitted:
                return self.milvus_client._hybrid_search_optimized(query_text, query_embedding, top_k)
            else:
                return self.milvus_client._dense_search_optimized(query_embedding, top_k)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    async def execute_graph_search(self, query: str, top_k: int) -> List[Dict]:
        """Execute graph search using Cypher query with intelligent query parsing and caching."""
        if self.performance_monitor:
            with self.performance_monitor.track_operation('graph_search'):
                return await self._execute_graph_search_internal(query, top_k)
        else:
            return await self._execute_graph_search_internal(query, top_k)

    async def _execute_graph_search_internal(self, query: str, top_k: int) -> List[Dict]:
        """Internal graph search implementation with caching."""
        try:
            # Check cache first
            if self.cache_manager:
                cached_results = self.cache_manager.get_search_results(
                    query, top_k, use_hybrid=False, routing_strategy="graph"
                )
                if cached_results is not None:
                    if self.performance_monitor:
                        self.performance_monitor.record_cache_hit('search', True)
                        self.performance_monitor.record_result_count('graph', len(cached_results))
                    logger.debug(f"Graph search cache hit for: {query[:50]}...")
                    return cached_results

            if self.performance_monitor:
                self.performance_monitor.record_cache_hit('search', False)

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

            # Cache results
            if self.cache_manager and results:
                self.cache_manager.cache_search_results(
                    query, results, top_k, use_hybrid=False, routing_strategy="graph"
                )

            if self.performance_monitor:
                self.performance_monitor.record_result_count('graph', len(results))

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

    def _build_intelligent_cypher_query(self, query: str, top_k: int) -> tuple:
        """Build Cypher query using AI agent based on Neo4j schema and user query."""
        try:
            if not self.ai_agent:
                logger.warning("AI agent not available, falling back to basic query generation")
                return self._build_fallback_cypher_query(query, top_k)
            
            # Generate Cypher query using AI agent
            schema_prompt = self._create_schema_prompt(query, top_k)
            ai_response = self.ai_agent.generate_content(prompt=schema_prompt)
            
            if not ai_response:
                logger.error("Empty response from AI agent for Cypher generation")
                return self._build_fallback_cypher_query(query, top_k)
            
            # Parse AI response to extract Cypher query and parameters
            cypher_query, parameters = self._parse_ai_cypher_response(ai_response, query, top_k)
            
            logger.info(f"AI-generated Cypher: {cypher_query}")
            logger.info(f"AI-generated Parameters: {parameters}")
            
            return cypher_query, parameters
            
        except Exception as e:
            logger.error(f"Error in AI Cypher generation: {e}")
            return self._build_fallback_cypher_query(query, top_k)
    
    def _create_schema_prompt(self, query: str, top_k: int) -> str:
        """Create a comprehensive prompt for AI-based Cypher query generation."""
        return f"""
You are a Neo4j Cypher query expert. Generate a precise Cypher query based on the user's natural language question.

Neo4j Schema:

Nodes:
- Paper: {{id, title, abstract, publication_date, doi, pmid, arxiv_id, pdf_url, source, metadata}}
- Author: {{id, name, orcid, email, h_index, metadata}}  
- Venue: {{id, name, type, issn, impact_factor, publisher, metadata}}
- Institution: {{id, name, country, city, type, website, metadata}}
- Figure: {{id, paper_id, caption, image_path, metadata}}
- Table: {{id, paper_id, caption, content, metadata}}

Relations:
- (Author)-[:AUTHORED]->(Paper) - Authorship relationships
- (Paper)-[:PUBLISHED_IN]->(Venue) - Publication venue associations
- (Paper)-[:CITES]->(Paper) - Citation networks between papers
- (Paper)-[:ASSOCIATED_WITH]->(Institution) - Institutional affiliations
- (Paper)-[:CONTAINS_FIGURE]->(Figure) - Figure ownership
- (Paper)-[:CONTAINS_TABLE]->(Table) - Table ownership

User Query: "{query}"
Limit: {top_k}

Instructions:
1. Analyze the user's query to understand what information they're seeking
2. Generate a Cypher query that returns the most relevant information
3. Always include basic paper information (id, title, abstract, doi, publication_date)
4. Use OPTIONAL MATCH for relationships that might not exist
5. Use collect(DISTINCT ...) for aggregating related data
6. Include proper WHERE clauses for filtering
7. Add ORDER BY and LIMIT clauses when appropriate
8. Use parameters for dynamic values (e.g., paper IDs, author names)

Response Format:
CYPHER_QUERY:
[Your Cypher query here]

PARAMETERS:
[JSON object with parameter values, or {{}} if no parameters needed]

EXPLANATION:
[Brief explanation of what the query does]

Generate the query now:"""

    def _parse_ai_cypher_response(self, ai_response: str, original_query: str, top_k: int) -> tuple:
        """Parse AI response to extract Cypher query and parameters."""
        try:
            # Extract Cypher query section
            cypher_start = ai_response.find("CYPHER_QUERY:")
            params_start = ai_response.find("PARAMETERS:")
            
            if cypher_start == -1:
                logger.warning("Could not find CYPHER_QUERY section in AI response")
                return self._build_fallback_cypher_query(original_query, top_k)
            
            # Extract query text
            if params_start != -1:
                cypher_text = ai_response[cypher_start + 13:params_start].strip()
                params_text = ai_response[params_start + 11:].strip()
                
                # Extract parameters if explanation follows
                explanation_start = params_text.find("EXPLANATION:")
                if explanation_start != -1:
                    params_text = params_text[:explanation_start].strip()
            else:
                cypher_text = ai_response[cypher_start + 13:].strip()
                params_text = "{}"
            
            # Clean up the query text
            cypher_query = self._clean_cypher_query(cypher_text)
            
            # Parse parameters
            try:
                import json
                parameters = json.loads(params_text) if params_text and params_text != "{}" else {}
            except json.JSONDecodeError:
                logger.warning(f"Could not parse parameters: {params_text}")
                parameters = {}
            
            # Add top_k limit if not present
            if "limit" not in parameters and "LIMIT" not in cypher_query.upper():
                parameters["limit"] = top_k
            
            # Validate the query has required components
            if not self._validate_cypher_query(cypher_query):
                logger.warning("Generated Cypher query failed validation")
                return self._build_fallback_cypher_query(original_query, top_k)
            
            return cypher_query, parameters
            
        except Exception as e:
            logger.error(f"Error parsing AI Cypher response: {e}")
            return self._build_fallback_cypher_query(original_query, top_k)
    
    def _clean_cypher_query(self, query_text: str) -> str:
        """Clean and format the Cypher query from AI response."""
        # Remove code block markers if present
        query_text = query_text.strip()
        if query_text.startswith("```"):
            lines = query_text.split("\n")[1:-1]  # Remove first and last lines
            query_text = "\n".join(lines)
        
        # Remove any remaining markdown or formatting
        query_text = query_text.replace("```cypher", "").replace("```", "")
        
        return query_text.strip()
    
    def _validate_cypher_query(self, cypher_query: str) -> bool:
        """Basic validation of generated Cypher query."""
        query_upper = cypher_query.upper()
        
        # Must contain MATCH
        if "MATCH" not in query_upper:
            return False
        
        # Must contain RETURN
        if "RETURN" not in query_upper:
            return False
        
        # Should reference Paper node (main entity)
        if ":Paper" not in cypher_query:
            return False
        
        return True
    
    def _build_fallback_cypher_query(self, query: str, top_k: int) -> tuple:
        """Fallback method for basic Cypher query generation when AI fails."""
        query_lower = query.lower()
        
        # Extract basic entities
        paper_ids = re.findall(r'\b(W\d+)\b', query)
        author_names = self._extract_author_names(query)
        
        # Simple fallbacks based on detected entities
        if paper_ids:
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
        
        elif author_names:
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
        
        else:
            # Default text search
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

    async def _execute_graph_refinement(self, paper_ids: List[str], top_k: int) -> List[Dict]:
        """Refine vector results using graph relationships."""
        try:
            if not self.graph_handler:
                logger.error("Graph handler not initialized")
                return []

            # Find related papers through citations and collaborations
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
        """Generate AI response from vector and graph searches."""
        try:
            if not self.ai_agent:
                logger.info("AI Agent is not available for response generation")
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

    async def verify_claims_scifact(self, ai_answer: str, context_papers: List) -> List[Dict]:
        """
        Implements SciFact-style verification by breaking the AI response into
        atomic claims and checking them against the retrieved substrate.
        """
        # Step 1: Claim Extraction
        # Use a prompt to break the response into individual facts
        claims = self.extract_atomic_claims(ai_answer)

        verification_results = []

        for claim in claims:
            # Step 2: Evidence Alignment
            # Compare the claim against the 'top k' abstracts and graph metadata
            logic_prompt = f"""
            Claim: {claim}
            Evidence: {self._format_papers_for_prompt(context_papers)}

            Label this claim as:
            - SUPPORTED: If the evidence explicitly confirms the claim.
            - CONTRADICTED: If the evidence explicitly refutes the claim.
            - NO_EVIDENCE: If the evidence is relevant but doesn't prove/disprove it.
            """

            label = self.ai_agent.generate_content(logic_prompt)
            verification_results.append({"claim": claim, "label": label})

        return verification_results

    def extract_atomic_claims(self, ai_answer: str) -> List[str]:
        """
        Decomposes a complex AI response into atomic, verifiable scientific claims.
        This supports the 'Verification' research goal of reducing hallucinations[cite: 25].
        """
        try:
            if not ai_answer or len(ai_answer.strip()) == 0:
                return []

            # This prompt instructs the LLM to act as a claim extractor,
            # focusing on 'scientific discovery' and 'factual groundedness'[cite: 7, 47].
            extraction_prompt = f"""
            You are a scientific fact-checker. Break the following AI-generated response into a list of atomic, independent claims.

            Rules:
            1. Each claim must be a single sentence.
            2. Each claim must be self-contained (replace pronouns like 'it' or 'they' with the specific entity names).
            3. Remove introductory phrases (e.g., "The paper says", "According to the results").
            4. Focus on scientific facts, authors, dates, and relationships between entities.

            Response to decompose:
            "{ai_answer}"

            Return the claims as a simple bulleted list.
            """

            # Using your existing AI agent infrastructure
            raw_claims = self.ai_agent.generate_content(prompt=extraction_prompt)

            if not raw_claims:
                return []

            # Clean the output into a Python list
            processed_claims = [
                claim.strip().lstrip('- ').lstrip('* ').strip()
                for claim in raw_claims.split('\n')
                if len(claim.strip()) > 10  # Filter out noise or empty lines
            ]

            logger.info(f"Extracted {len(processed_claims)} atomic claims for verification [cite: 25]")
            return processed_claims

        except Exception as e:
            logger.error(f"Error in claim extraction: {e}")
            return []

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