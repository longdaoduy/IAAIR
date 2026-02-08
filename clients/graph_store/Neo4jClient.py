"""
Neo4j Graph Store implementation for citation networks and concept relationships.

Handles storage and querying of the scientific literature graph_store including
citations, author collaborations, and concept hierarchies.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver

from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.nodes.Venue import Venue
from models.schemas.edges.CitedBy import CitedBy
from models.configurators.GraphDBConfig import GraphDBConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j implementation of the graph_store store for scientific literature.
    
    Manages citation networks, author collaboration graphs, and
    concept hierarchies with optimized Cypher queries for retrievals.
    """

    def __init__(self):
        self.config = GraphDBConfig()
        self.driver: Optional[AsyncDriver] = None
        self._connection_pool = None

    async def connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size
            )

            # Verify connectivity
            await self.driver.verify_connectivity()
            logger.info("Successfully connected to Neo4j")
            
            # Create unique constraints and indexes
            await self.create_constraints()
            await self.create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def create_constraints(self):
        """Create unique constraints for node IDs to prevent duplicates."""
        constraints = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE", 
            "CREATE CONSTRAINT venue_id_unique IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE"
        ]
        
        # Note: Neo4j doesn't support unique constraints on relationships directly,
        # but we handle this in the MERGE logic for relationships
        
        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    # Constraint might already exist, log but don't fail
                    logger.debug(f"Constraint already exists or failed: {e}")

    async def create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX paper_doi_index IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX paper_pmid_index IF NOT EXISTS FOR (p:Paper) ON (p.pmid)",
            "CREATE INDEX paper_title_index IF NOT EXISTS FOR (p:Paper) ON (p.title)",
            "CREATE INDEX author_name_index IF NOT EXISTS FOR (a:Author) ON (a.name)",
            "CREATE INDEX author_orcid_index IF NOT EXISTS FOR (a:Author) ON (a.orcid)",
            "CREATE INDEX venue_name_index IF NOT EXISTS FOR (v:Venue) ON (v.name)",
            "CREATE INDEX venue_issn_index IF NOT EXISTS FOR (v:Venue) ON (v.issn)"
        ]
        
        async with self.driver.session() as session:
            for index in indexes:
                try:
                    await session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.debug(f"Index already exists or failed: {e}")

    async def initialize_database(self):
        """Initialize database with constraints and indexes. Call this once per database setup."""
        logger.info("Initializing database schema...")
        await self.create_constraints()
        await self.create_indexes()
        logger.info("Database schema initialization complete")

    async def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def health_check(self) -> bool:
        """Perform health check on the Neo4j connection."""
        try:
            if not self.driver:
                await self.connect()

            async with self.driver.session() as session:
                result = await session.run("RETURN 1 AS health")
                await result.single()
                return True

        except Exception as e:
            logger.error(f"Neo4j health check failed: {e}")
            return False

    async def store_paper(self, paper: Paper, authors: List[Author] = None, venue: Venue = None,
                          citations: List[str] = None):
        """
        Store a paper and its relationships in the graph_store database.
        
        Creates nodes for the paper, authors, venue, and citation relationships.
        """
        async with self.driver.session() as session:
            try:
                # Start transaction
                tx = await session.begin_transaction()
                try:
                    # Create paper node
                    await self._create_paper_node(tx, paper)

                    # Create author nodes and relationships
                    if authors:
                        for author in authors:
                            await self._create_author_node(tx, author)
                            await self._create_authorship_relationship(tx, paper.id, author.id)

                    # Create venue node and relationship
                    if venue:
                        await self._create_venue_node(tx, venue)
                        await self._create_publication_relationship(tx, paper.id, venue.id)

                    # Create citation relationships
                    if citations:
                        for cited_paper_id in citations:
                            await self._create_simple_citation_relationship(tx, paper.id, cited_paper_id)

                    # Commit the transaction
                    await tx.commit()
                    logger.debug(f"Stored paper {paper.id} in graph_store database")

                except Exception as e:
                    # Rollback the transaction on error
                    await tx.rollback()
                    raise e
                finally:
                    await tx.close()

            except Exception as e:
                logger.error(f"Failed to store paper {paper.id}: {e}")
                raise

    async def _create_paper_node(self, tx, paper: Paper, cited_by_count: int = 0):
        """Create a paper node in the graph_store."""
        query = '''
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.publication_date = $publication_date,
            p.doi = $doi,
            p.pmid = $pmid,
            p.arxiv_id = $arxiv_id,
            p.source = $source,
            p.ingested_at = $ingested_at,
            p.last_updated = $last_updated,
            p.cited_by_count = $cited_by_count,
            p.metadata = $metadata
        '''

        await tx.run(query, {
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date.isoformat() if paper.publication_date else None,
            "doi": paper.doi,
            "pmid": getattr(paper, 'pmid', None),
            "arxiv_id": getattr(paper, 'arxiv_id', None),
            "source": paper.source,
            "ingested_at": paper.ingested_at.isoformat(),
            "last_updated": paper.last_updated.isoformat(),
            "cited_by_count": cited_by_count,
            "metadata": json.dumps(paper.metadata) if paper.metadata else "{}"
        })

    async def _create_author_node(self, tx, author: Author):
        """Create an author node in the graph_store."""
        query = '''
        MERGE (a:Author {id: $id})
        SET a.name = $name,
            a.orcid = $orcid,
            a.h_index = $h_index,
            a.ingested_at = $ingested_at,
            a.last_updated = $last_updated,
            a.metadata = $metadata
        '''

        await tx.run(query, {
            "id": author.id,
            "name": author.name,
            "orcid": author.orcid,
            "h_index": author.h_index,
            "ingested_at": author.ingested_at.isoformat(),
            "last_updated": author.last_updated.isoformat(),
            "metadata": json.dumps(author.metadata) if author.metadata else "{}"
        })

    async def _create_venue_node(self, tx, venue: Venue):
        """Create a venue node in the graph_store."""
        query = '''
        MERGE (v:Venue {id: $id})
        SET v.name = $name,
            v.type = $type,
            v.issn = $issn,
            v.impact_factor = $impact_factor,
            v.publisher = $publisher,
            v.ingested_at = $ingested_at,
            v.last_updated = $last_updated,
            v.metadata = $metadata
        '''

        await tx.run(query, {
            "id": venue.id,
            "name": venue.name,
            "type": venue.type.value if venue.type else None,
            "issn": venue.issn,
            "impact_factor": venue.impact_factor,
            "publisher": venue.publisher,
            "ingested_at": venue.ingested_at.isoformat(),
            "last_updated": venue.last_updated.isoformat(),
            "metadata": json.dumps(venue.metadata) if venue.metadata else "{}"
        })

    async def _create_authorship_relationship(self, tx, document_id: str, author_id: str):
        """Create authorship relationship between document and author."""
        query = '''
        MATCH (d:Paper {id: $document_id})
        MATCH (a:Author {id: $author_id})
        MERGE (a)-[:AUTHORED]->(d)
        '''

        await tx.run(query, {
            "document_id": document_id,
            "author_id": author_id
        })

    async def _create_publication_relationship(self, tx, document_id: str, venue_id: str):
        """Create publication relationship between document and venue."""
        query = '''
        MATCH (d:Paper {id: $document_id})
        MATCH (v:Venue {id: $venue_id})
        MERGE (d)-[:PUBLISHED_IN]->(v)
        '''

        await tx.run(query, {
            "document_id": document_id,
            "venue_id": venue_id
        })

    async def _create_citation_relationship(self, tx, citation: CitedBy):
        """Create citation relationship between documents, avoiding duplicates."""
        query = '''
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[c:CITES]->(cited)
        ON CREATE SET c.context = $context,
                     c.intent = $intent,
                     c.confidence = $confidence,
                     c.created_at = $created_at
        ON MATCH SET c.context = $context,
                    c.intent = $intent,
                    c.confidence = $confidence,
                    c.last_updated = $created_at
        RETURN c
        '''

        await tx.run(query, {
            "citing_id": citation.citing_paper_id,
            "cited_id": citation.cited_paper_id,
            "context": citation.context,
            "intent": citation.intent,
            "confidence": citation.confidence,
            "created_at": citation.created_at.isoformat()
        })

    async def _create_simple_citation_relationship(self, tx, citing_paper_id: str, cited_paper_id: str):
        """Create simple citation relationship between papers using IDs, avoiding duplicates."""
        from datetime import datetime
        
        # First check if the cited paper exists, create stub if not
        await tx.run("""
            MERGE (cited:Paper {id: $cited_id})
        """, {"cited_id": cited_paper_id})
        
        # Create the citation relationship only if it doesn't exist
        query = '''
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        ON CREATE SET r.created_at = $created_at
        RETURN r
        '''

        await tx.run(query, {
            "citing_id": citing_paper_id,
            "cited_id": cited_paper_id,
            "created_at": datetime.now().isoformat()
        })

    async def get_citation_subgraph(
            self,
            document_id: str,
            depth: int = 2
    ) -> Dict[str, Any]:
        """
        Get citation subgraph around a specific document.
        
        Args:
            document_id: Center document ID
            depth: Maximum traversal depth
            
        Returns:
            Subgraph data with nodes and relationships
        """
        query = f'''
        MATCH path = (center:Paper {{id: $document_id}})
                    -[:CITES*1..{depth}]-(connected:Paper)
        RETURN path
        LIMIT 1000
        '''

        async with self.driver.session() as session:
            result = await session.run(query, {"document_id": document_id})

            # Process results into subgraph structure
            nodes = {}
            edges = []

            async for record in result:
                path = record["path"]

                # Extract nodes and relationships from path
                for node in path.nodes:
                    if node.id not in nodes:
                        nodes[node.id] = dict(node)

                for rel in path.relationships:
                    edges.append({
                        "source": rel.start_node.id,
                        "target": rel.end_node.id,
                        "type": rel.type,
                        "properties": dict(rel)
                    })

            return {
                "center_document": document_id,
                "nodes": list(nodes.values()),
                "edges": edges,
                "depth": depth
            }

    async def find_similar_by_citations(
            self,
            document_id: str,
            limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find documents with similar citation patterns."""
        query = '''
        MATCH (target:Paper {id: $document_id})-[:CITES]->(cited:Paper)
        MATCH (similar:Paper)-[:CITES]->(cited)
        WHERE similar.id <> target.id
        WITH similar, count(cited) as common_citations
        ORDER BY common_citations DESC
        LIMIT $limit
        RETURN similar, common_citations
        '''

        async with self.driver.session() as session:
            result = await session.run(query, {
                "document_id": document_id,
                "limit": limit
            })

            similar_docs = []
            async for record in result:
                doc_data = dict(record["similar"])
                doc_data["common_citations"] = record["common_citations"]
                similar_docs.append(doc_data)

            return similar_docs

    async def get_author_collaboration_network(
            self,
            author_id: str,
            depth: int = 2
    ) -> Dict[str, Any]:
        """Get collaboration network for an author."""
        query = f'''
        MATCH path = (center:Author {{id: $author_id}})
                    -[:AUTHORED]->(:Paper)<-[:AUTHORED]-(collaborator:Author)
        WHERE center.id <> collaborator.id
        RETURN DISTINCT collaborator
        LIMIT 100
        '''

        async with self.driver.session() as session:
            result = await session.run(query, {"author_id": author_id})

            collaborators = []
            async for record in result:
                collaborators.append(dict(record["collaborator"]))

            return {
                "center_author": author_id,
                "collaborators": collaborators
            }

    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve detailed information for papers by their IDs.
        
        Args:
            paper_ids: List of paper IDs to retrieve
            
        Returns:
            List of paper dictionaries with detailed information
        """
        if not paper_ids:
            return []

        async with self.driver.session() as session:
            # Build query to get papers with authors and venue information
            query = '''
            MATCH (p:Paper)
            WHERE p.id IN $paper_ids
            OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
            OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
            OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
            
            WITH p, 
                 collect(DISTINCT {
                     id: a.id,
                     name: a.name,
                     orcid: a.orcid
                 }) as authors,
                 v,
                 count(DISTINCT cited) as citations_count
                 
            RETURN p.id as id,
                   p.title as title,
                   p.abstract as abstract,
                   p.doi as doi,
                   p.publication_date as publication_date,
                   p.source as source,
                   p.cited_by_count as cited_by_count,
                   authors,
                   {
                       id: v.id,
                       name: v.name,
                       type: v.type
                   } as venue,
                   citations_count
            '''

            result = await session.run(query, {"paper_ids": paper_ids})
            
            papers = []
            async for record in result:
                paper_data = {
                    "id": record["id"],
                    "title": record["title"],
                    "abstract": record["abstract"],
                    "doi": record["doi"],
                    "publication_date": record["publication_date"],
                    "source": record["source"],
                    "cited_by_count": record["cited_by_count"] or 0,
                    "authors": [author for author in record["authors"] if author["id"]],  # Filter out null authors
                    "venue": record["venue"] if record["venue"]["id"] else None,
                    "citations": [],  # We have citations_count but not the actual citations list
                    "citations_count": record["citations_count"] or 0
                }
                papers.append(paper_data)

            return papers
