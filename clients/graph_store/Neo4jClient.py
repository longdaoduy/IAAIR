"""
Neo4j Graph Store implementation for citation networks and concept relationships.

Handles storage and querying of the scientific literature graph_store including
citations, author collaborations, and concept hierarchies.
"""

import logging
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
from datetime import datetime

from knowledge_fabric.schemas import Document, Author, Venue, Citation
from configurators.GraphDBConfig import GraphDBConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Neo4j implementation of the graph_store store for scientific literature.
    
    Manages citation networks, author collaboration graphs, and
    concept hierarchies with optimized Cypher queries for retrieval.
    """
    
    def __init__(self, config: GraphDBConfig):
        self.config = config
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
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
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
    
    async def store_document(self, document: Document):
        """
        Store a document and its relationships in the graph_store database.
        
        Creates nodes for the document, authors, venue, and citation relationships.
        """
        async with self.driver.session() as session:
            try:
                # Start transaction
                async with session.begin_transaction() as tx:
                    # Create document node
                    await self._create_document_node(tx, document)
                    
                    # Create author nodes and relationships
                    for author in document.authors:
                        await self._create_author_node(tx, author)
                        await self._create_authorship_relationship(tx, document.id, author.id)
                    
                    # Create venue node and relationship
                    if document.venue:
                        await self._create_venue_node(tx, document.venue)
                        await self._create_publication_relationship(tx, document.id, document.venue.id)
                    
                    # Create citation relationships
                    for citation in document.citations:
                        await self._create_citation_relationship(tx, citation)
                    
                logger.debug(f"Stored document {document.id} in graph_store database")
                
            except Exception as e:
                logger.error(f"Failed to store document {document.id}: {e}")
                raise
    
    async def _create_document_node(self, tx, document: Document):
        """Create a document node in the graph_store."""
        query = '''
        MERGE (d:Document {id: $id})
        SET d.title = $title,
            d.abstract = $abstract,
            d.document_type = $document_type,
            d.publication_date = $publication_date,
            d.doi = $doi,
            d.arxiv_id = $arxiv_id,
            d.url = $url,
            d.language = $language,
            d.keywords = $keywords,
            d.subjects = $subjects,
            d.source = $source,
            d.ingested_at = $ingested_at,
            d.last_updated = $last_updated
        '''
        
        await tx.run(query, {
            "id": document.id,
            "title": document.title,
            "abstract": document.abstract,
            "document_type": document.document_type.value if document.document_type else None,
            "publication_date": document.publication_date.isoformat() if document.publication_date else None,
            "doi": document.doi,
            "arxiv_id": document.arxiv_id,
            "url": document.url,
            "language": document.language,
            "keywords": document.keywords,
            "subjects": document.subjects,
            "source": document.source,
            "ingested_at": document.ingested_at.isoformat(),
            "last_updated": document.last_updated.isoformat()
        })
    
    async def _create_author_node(self, tx, author: Author):
        """Create an author node in the graph_store."""
        query = '''
        MERGE (a:Author {id: $id})
        SET a.name = $name,
            a.orcid = $orcid,
            a.affiliation = $affiliation,
            a.h_index = $h_index
        '''
        
        await tx.run(query, {
            "id": author.id,
            "name": author.name,
            "orcid": author.orcid,
            "affiliation": author.affiliation,
            "h_index": author.h_index
        })
    
    async def _create_venue_node(self, tx, venue: Venue):
        """Create a venue node in the graph_store."""
        query = '''
        MERGE (v:Venue {id: $id})
        SET v.name = $name,
            v.venue_type = $venue_type,
            v.issn = $issn,
            v.impact_factor = $impact_factor,
            v.publisher = $publisher
        '''
        
        await tx.run(query, {
            "id": venue.id,
            "name": venue.name,
            "venue_type": venue.venue_type.value if venue.venue_type else None,
            "issn": venue.issn,
            "impact_factor": venue.impact_factor,
            "publisher": venue.publisher
        })
    
    async def _create_authorship_relationship(self, tx, document_id: str, author_id: str):
        """Create authorship relationship between document and author."""
        query = '''
        MATCH (d:Document {id: $document_id})
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
        MATCH (d:Document {id: $document_id})
        MATCH (v:Venue {id: $venue_id})
        MERGE (d)-[:PUBLISHED_IN]->(v)
        '''
        
        await tx.run(query, {
            "document_id": document_id,
            "venue_id": venue_id
        })
    
    async def _create_citation_relationship(self, tx, citation: Citation):
        """Create citation relationship between documents."""
        query = '''
        MATCH (citing:Document {id: $citing_id})
        MATCH (cited:Document {id: $cited_id})
        MERGE (citing)-[c:CITES {
            context: $context,
            intent: $intent,
            confidence: $confidence,
            created_at: $created_at
        }]->(cited)
        '''
        
        await tx.run(query, {
            "citing_id": citation.citing_paper_id,
            "cited_id": citation.cited_paper_id,
            "context": citation.context,
            "intent": citation.intent,
            "confidence": citation.confidence,
            "created_at": citation.created_at.isoformat()
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
        MATCH path = (center:Document {{id: $document_id}})
                    -[:CITES*1..{depth}]-(connected:Document)
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
        MATCH (target:Document {id: $document_id})-[:CITES]->(cited:Document)
        MATCH (similar:Document)-[:CITES]->(cited)
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
                    -[:AUTHORED]->(:Document)<-[:AUTHORED]-(collaborator:Author)
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