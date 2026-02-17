"""
Neo4j Graph Store implementation for citation networks and concept relationships.

Handles storage and querying of the scientific literature graph including
citations, author collaborations, and concept hierarchies.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver

from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.nodes.Venue import Venue
from models.schemas.nodes.Institution import Institution
from models.schemas.nodes.Figure import Figure
from models.schemas.nodes.Table import Table
from models.schemas.edges.CitedBy import CitedBy
from models.configurators.GraphDBConfig import GraphDBConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j implementation of the graph store for scientific literature.

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
            # await self.create_constraints()
            await self.create_indexes()

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    async def create_constraints(self):
        """Create unique constraints for node IDs to prevent duplicates."""
        constraints = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE", 
            "CREATE CONSTRAINT venue_id_unique IF NOT EXISTS FOR (v:Venue) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT institution_id_unique IF NOT EXISTS FOR (i:Institution) REQUIRE i.id IS UNIQUE",
            "CREATE CONSTRAINT figure_id_unique IF NOT EXISTS FOR (f:Figure) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT table_id_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.id IS UNIQUE"
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
            "CREATE INDEX author_id_index IF NOT EXISTS FOR (a:Author) ON (a.id)",
            "CREATE INDEX author_orcid_index IF NOT EXISTS FOR (a:Author) ON (a.orcid)",
            "CREATE INDEX venue_name_index IF NOT EXISTS FOR (v:Venue) ON (v.name)",
            "CREATE INDEX venue_issn_index IF NOT EXISTS FOR (v:Venue) ON (v.issn)",
            "CREATE INDEX institution_name_index IF NOT EXISTS FOR (i:Institution) ON (i.name)",
            "CREATE INDEX figure_paper_id_index IF NOT EXISTS FOR (f:Figure) ON (f.paper_id)",
            "CREATE INDEX table_paper_id_index IF NOT EXISTS FOR (t:Table) ON (t.paper_id)"
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
        # await self.create_constraints()
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
        Store a paper and its relationships in the graph database.
        
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
                    logger.debug(f"Stored paper {paper.id} in graph database")

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
        """Create a paper node in the graph."""
        query = '''
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.abstract = $abstract,
            p.publication_date = $publication_date,
            p.doi = $doi,
            p.pmid = $pmid,
            p.arxiv_id = $arxiv_id,
            p.pdf_url = $pdf_url,
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
            "pdf_url": getattr(paper, 'pdf_url', None),
            "source": paper.source,
            "ingested_at": paper.ingested_at.isoformat(),
            "last_updated": paper.last_updated.isoformat(),
            "cited_by_count": cited_by_count,
            "metadata": json.dumps(paper.metadata) if paper.metadata else "{}"
        })

    async def _create_author_node(self, tx, author: Author):
        """Create an author node in the graph."""
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
        """Create a venue node in the graph."""
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

    async def _create_institution_node(self, tx, institution: Institution):
        """Create an institution node in the graph."""
        query = '''
        MERGE (i:Institution {id: $id})
        SET i.name = $name,
            i.country = $country,
            i.city = $city,
            i.type = $type,
            i.website = $website,
            i.ingested_at = $ingested_at,
            i.last_updated = $last_updated,
            i.metadata = $metadata
        '''

        await tx.run(query, {
            "id": institution.id,
            "name": institution.name,
            "country": institution.country,
            "city": institution.city,
            "type": institution.type,
            "website": institution.website,
            "ingested_at": institution.ingested_at.isoformat(),
            "last_updated": institution.last_updated.isoformat(),
            "metadata": json.dumps(institution.metadata) if institution.metadata else "{}"
        })

    async def _create_figure_node(self, tx, figure: Figure):
        """Create a figure node in the graph."""
        query = '''
        MERGE (f:Figure {id: $id})
        SET f.paper_id = $paper_id,
            f.figure_number = $figure_number,
            f.description = $description,
            f.caption = $caption,
            f.page_number = $page_number,
            f.image_path = $image_path,
            f.image_embedding = $image_embedding,
            f.ingested_at = $ingested_at,
            f.last_updated = $last_updated,
            f.metadata = $metadata
        '''

        await tx.run(query, {
            "id": figure.id,
            "paper_id": figure.paper_id,
            "figure_number": figure.figure_number,
            "description": figure.description,
            "caption": figure.caption,
            "page_number": figure.page_number,
            "image_path": figure.image_path,
            "image_embedding": figure.image_embedding,
            "ingested_at": figure.ingested_at.isoformat(),
            "last_updated": figure.last_updated.isoformat(),
            "metadata": json.dumps(figure.metadata) if figure.metadata else "{}"
        })

    async def _create_table_node(self, tx, table: Table):
        """Create a table node in the graph."""
        query = '''
        MERGE (t:Table {id: $id})
        SET t.paper_id = $paper_id,
            t.table_number = $table_number,
            t.description = $description,
            t.caption = $caption,
            t.page_number = $page_number,
            t.headers = $headers,
            t.rows = $rows,
            t.table_text = $table_text,
            t.image_path = $image_path,
            t.image_embedding = $image_embedding,
            t.ingested_at = $ingested_at,
            t.last_updated = $last_updated,
            t.metadata = $metadata
        '''

        await tx.run(query, {
            "id": table.id,
            "paper_id": table.paper_id,
            "table_number": table.table_number,
            "description": table.description,
            "caption": table.caption,
            "page_number": table.page_number,
            "headers": json.dumps(table.headers) if table.headers else None,
            "rows": json.dumps(table.rows) if table.rows else None,
            "table_text": table.table_text,
            "image_path": table.image_path,
            "image_embedding": table.image_embedding,
            "ingested_at": table.ingested_at.isoformat(),
            "last_updated": table.last_updated.isoformat(),
            "metadata": json.dumps(table.metadata) if table.metadata else "{}"
        })

    async def _create_figure_relationship(self, tx, paper_id: str, figure_id: str):
        """Create a relationship between paper and figure."""
        query = '''
        MATCH (p:Paper {id: $paper_id})
        MATCH (f:Figure {id: $figure_id})
        MERGE (p)-[:CONTAINS_FIGURE]->(f)
        '''
        await tx.run(query, {"paper_id": paper_id, "figure_id": figure_id})

    async def _create_table_relationship(self, tx, paper_id: str, table_id: str):
        """Create a relationship between paper and table."""
        query = '''
        MATCH (p:Paper {id: $paper_id})
        MATCH (t:Table {id: $table_id})
        MERGE (p)-[:CONTAINS_TABLE]->(t)
        '''
        await tx.run(query, {"paper_id": paper_id, "table_id": table_id})

    async def _create_institution_paper_relationship(self, tx, paper_id: str, institution_id: str):
        """Create a relationship between paper and institution."""
        query = '''
        MATCH (p:Paper {id: $paper_id})
        MATCH (i:Institution {id: $institution_id})
        MERGE (p)-[:ASSOCIATED_WITH_INSTITUTION]->(i)
        '''
        await tx.run(query, {"paper_id": paper_id, "institution_id": institution_id})

    async def store_paper_with_content(self, paper: Paper, authors: List[Author] = None, 
                                     venue: Venue = None, citations: List[str] = None,
                                     institutions: List[Institution] = None,
                                     figures: List[Figure] = None, tables: List[Table] = None):
        """
        Store a paper with all its associated content including figures and tables.
        
        Args:
            paper: Paper entity
            authors: List of authors
            venue: Venue entity
            citations: List of citation IDs
            institutions: List of institutions for authors
            figures: List of figures extracted from paper
            tables: List of tables extracted from paper
        """
        async with self.driver.session() as session:
            try:
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

                    # Create institution nodes and relationships
                    if institutions:
                        for institution in institutions:
                            await self._create_institution_node(tx, institution)
                            # Create relationship between paper and institution
                            await self._create_institution_paper_relationship(tx, paper.id, institution.id)

                    # Create figure nodes and relationships
                    if figures:
                        for figure in figures:
                            await self._create_figure_node(tx, figure)
                            await self._create_figure_relationship(tx, paper.id, figure.id)

                    # Create table nodes and relationships
                    if tables:
                        for table in tables:
                            await self._create_table_node(tx, table)
                            await self._create_table_relationship(tx, paper.id, table.id)

                    await tx.commit()
                    logger.info(f"Stored paper {paper.id} with all content in graph database")

                except Exception as e:
                    await tx.rollback()
                    raise e
                finally:
                    await tx.close()

            except Exception as e:
                logger.error(f"Failed to store paper with content {paper.id}: {e}")
                raise

    async def diagnose_relationships(self):
        """Diagnose relationship issues in the graph database."""
        async with self.driver.session() as session:
            try:
                # Query to check all relationship types
                query = '''
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType
                ORDER BY relationshipType
                '''
                
                result = await session.run(query)
                records = await result.data()
                
                print("=== All Relationship Types in Database ===")
                for record in records:
                    rel_type = record['relationshipType']
                    print(f"  - {rel_type}")
                
                # Query to check for problematic relationships
                problematic_query = '''
                MATCH ()-[r]-()
                WHERE type(r) = '*' OR type(r) = '' OR type(r) IS NULL
                RETURN type(r) as rel_type, count(*) as count
                '''
                
                result = await session.run(problematic_query)
                records = await result.data()
                
                if records:
                    print("\n=== Problematic Relationships Found ===")
                    for record in records:
                        print(f"  - Type: '{record['rel_type']}', Count: {record['count']}")
                else:
                    print("\n✅ No problematic relationships found")
                
                # Check relationship patterns
                pattern_query = '''
                MATCH (n1)-[r]->(n2)
                RETURN labels(n1)[0] as from_label, type(r) as rel_type, labels(n2)[0] as to_label, count(*) as count
                ORDER BY from_label, rel_type, to_label
                LIMIT 20
                '''
                
                result = await session.run(pattern_query)
                records = await result.data()
                
                print("\n=== Relationship Patterns (Top 20) ===")
                for record in records:
                    print(f"  {record['from_label']} -[{record['rel_type']}]-> {record['to_label']} ({record['count']})")
                
            except Exception as e:
                logger.error(f"Error diagnosing relationships: {e}")
                print(f"Error diagnosing relationships: {e}")

    async def cleanup_invalid_relationships(self):
        """Clean up any invalid relationships that might be causing issues."""
        async with self.driver.session() as session:
            try:
                # Remove relationships with empty or null types
                cleanup_query = '''
                MATCH ()-[r]-()
                WHERE type(r) = '*' OR type(r) = '' OR type(r) IS NULL
                DELETE r
                RETURN count(r) as deleted_count
                '''
                
                result = await session.run(cleanup_query)
                record = await result.single()
                deleted_count = record['deleted_count'] if record else 0
                
                print(f"Cleaned up {deleted_count} invalid relationships")
                
            except Exception as e:
                logger.error(f"Error cleaning up relationships: {e}")
                print(f"Error cleaning up relationships: {e}")
