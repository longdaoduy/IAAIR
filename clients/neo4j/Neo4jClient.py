import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from neo4j import AsyncGraphDatabase, AsyncDriver
import enum

from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.nodes.Venue import Venue
from models.schemas.nodes.Institution import Institution
from models.schemas.nodes.Figure import Figure
from models.schemas.nodes.Table import Table
from models.configurators.GraphDBConfig import GraphDBConfig

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Optimized Neo4j Client for scientific literature graphs.
    Uses batch UNWIND operations and dynamic property mapping for high performance.
    Provides both async (ingestions) and sync (query) interfaces.
    """

    def __init__(self):
        self.config = GraphDBConfig()
        self.driver: Optional[AsyncDriver] = None
        self._sync_driver = None  # Synchronous driver for query execution

    async def connect(self):
        """Establish async connection and verify schema."""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size
            )
            await self.driver.verify_connectivity()
            await self._initialize_schema()
            logger.info("Successfully connected to Neo4j and initialized schema.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def connect_sync(self):
        """Establish synchronous connection for query execution."""
        from neo4j import GraphDatabase
        try:
            self._sync_driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size
            )
            logger.info("✅ Neo4j sync driver connected (for queries)")
        except Exception as e:
            logger.error(f"❌ Failed to connect sync Neo4j driver: {e}")
            raise
    
    def _ensure_sync_driver(self):
        """Lazily initialize the sync driver if not already connected."""
        if self._sync_driver is None:
            self.connect_sync()
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query synchronously and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        self._ensure_sync_driver()
        
        try:
            with self._sync_driver.session() as session:
                result = session.run(query, parameters or {})
                records = []
                for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        if hasattr(value, '_properties'):
                            record_dict[key] = dict(value._properties)
                            record_dict[key]['_labels'] = list(value.labels) if hasattr(value, 'labels') else []
                            record_dict[key]['_id'] = value.id if hasattr(value, 'id') else None
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                logger.info(f"📊 Query returned {len(records)} records")
                return records
                
        except Exception as e:
            logger.error(f"❌ Cypher query error: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and node/relationship counts."""
        stats_queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'paper_count': "MATCH (p:Paper) RETURN count(p) as count",
            'author_count': "MATCH (a:Author) RETURN count(a) as count",
            'venue_count': "MATCH (v:Venue) RETURN count(v) as count",
            'citation_count': "MATCH ()-[:CITES]->() RETURN count(*) as count",
            'authorship_count': "MATCH ()-[:AUTHORED]->() RETURN count(*) as count",
            'publication_count': "MATCH ()-[:PUBLISHED_IN]->() RETURN count(*) as count"
        }
        
        stats = {}
        for stat_name, query in stats_queries.items():
            try:
                result = self.execute_query(query)
                stats[stat_name] = result[0]['count'] if result else 0
            except Exception as e:
                logger.warning(f"⚠️ Failed to get {stat_name}: {e}")
                stats[stat_name] = 0
        return stats

    async def _initialize_schema(self):
        """Sets up constraints (enforce uniqueness) and indexes (speed up search)."""
        queries = [
            # Constraints
            "CREATE CONSTRAINT FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (a:Author) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (v:Venue) REQUIRE v.id IS UNIQUE",
            "CREATE CONSTRAINT FOR (i:Institution) REQUIRE i.id IS UNIQUE",
            # Indexes
            "CREATE INDEX FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX FOR (p:Paper) ON (p.title)",
            "CREATE INDEX FOR (f:Figure) ON (f.paper_id)"
        ]
        async with self.driver.session() as session:
            for q in queries:
                try:
                    await session.run(q)
                except Exception:
                    pass  # Ignore if already exists

    async def close(self):
        if self.driver:
            await self.driver.close()
            self.driver = None
        if self._sync_driver:
            self._sync_driver.close()
            self._sync_driver = None
            logger.info("Neo4j sync driver closed")

    # --- Core Ingestion Logic (Atomic & Batched) ---

    async def store_paper_with_content(self, paper: Paper, authors: List[Author] = None,
                                       venue: Venue = None, citations: List[str] = None,
                                       institutions: List[Institution] = None,
                                       figures: List[Figure] = None, tables: List[Table] = None):
        """Orchestrates the atomic ingestions of a paper and all related entities."""
        async with self.driver.session() as session:
            try:
                await session.execute_write(
                    self._ingest_transaction,
                    paper, authors, venue, citations, institutions, figures, tables)
            except Exception as e:
                logger.error(f"Ingestion failed for paper {paper.id}: {e}")
                raise

    @staticmethod
    async def _ingest_transaction(tx, paper, authors, venue, citations, institutions, figures, tables):
        """The internal transaction logic using Cypher UNWIND for bulk processing."""

        # 1. Upsert Paper
        p_props = Neo4jClient._prepare_props(paper)
        await tx.run("""
            MERGE (p:Paper {id: $id})
            SET p += $props, p.last_updated = datetime()
        """, id=paper.id, props=p_props)

        # 2. Batch Authors & Authorship
        if authors:
            a_list = [Neo4jClient._prepare_props(a) for a in authors]
            await tx.run("""
                UNWIND $list AS auth
                MERGE (a:Author {id: auth.id})
                SET a += auth, a.last_updated = datetime()
                WITH a
                MATCH (p:Paper {id: $pid})
                MERGE (a)-[:AUTHORED]->(p)
            """, list=a_list, pid=paper.id)

        # 3. Venue & Publication
        if venue:
            v_props = Neo4jClient._prepare_props(venue)
            await tx.run("""
                MERGE (v:Venue {id: $id})
                SET v += $props, v.last_updated = datetime()
                WITH v
                MATCH (p:Paper {id: $pid})
                MERGE (p)-[:PUBLISHED_IN]->(v)
            """, id=venue.id, props=v_props, pid=paper.id)

        # 4. Batch Citations (Creates stubs for cited papers not yet in DB)
        if citations:
            await tx.run("""
                UNWIND $cites AS cited_id
                MERGE (cited:Paper {id: cited_id})
                WITH cited
                MATCH (p:Paper {id: $pid})
                MERGE (p)-[:CITES]->(cited)
            """, cites=citations, pid=paper.id)

        # 5. Batch Institutions
        if institutions:
            i_list = [Neo4jClient._prepare_props(i) for i in institutions]
            await tx.run("""
                UNWIND $list AS inst
                MERGE (i:Institution {id: inst.id})
                SET i += inst, i.last_updated = datetime()
                WITH i
                MATCH (p:Paper {id: $pid})
                MERGE (p)-[:ASSOCIATED_WITH]->(i)
            """, list=i_list, pid=paper.id)

        # 6. Batch Figures
        if figures:
            f_list = [Neo4jClient._prepare_props(f) for f in figures]
            await tx.run("""
                UNWIND $list AS fig
                MERGE (f:Figure {id: fig.id})
                SET f += fig, f.last_updated = datetime()
                WITH f
                MATCH (p:Paper {id: $pid})
                MERGE (p)-[:CONTAINS_FIGURE]->(f)
            """, list=f_list, pid=paper.id)

        if tables:
            t_list = [Neo4jClient._prepare_props(t) for t in tables]
            await tx.run("""
                UNWIND $list AS tab
                MERGE (t:Table {id: tab.id})
                SET t += tab, t.last_updated = datetime()
                WITH t
                MATCH (p:Paper {id: $pid})
                MERGE (p)-[:CONTAINS_TABLE]->(t)
            """, list=t_list, pid=paper.id)

    # --- Utilities ---

    @staticmethod
    def _prepare_props(obj: Any) -> Dict:
        """Helper to convert objects/Pydantic models to Neo4j-friendly dicts."""
        data = obj.dict() if hasattr(obj, 'dict') else vars(obj)
        clean = {}
        
        # Fields to exclude from Neo4j storage (embeddings are stored in Milvus)
        excluded_fields = {'image_embedding', 'description_embedding'}
        
        for k, v in data.items():
            # Skip embedding fields
            if k in excluded_fields:
                continue
                
            # Handle Enums (like VenueType)
            if isinstance(v, enum.Enum):
                clean[k] = v.value
            # Handle Datetimes
            elif isinstance(v, datetime):
                clean[k] = v.isoformat()
            # Handle complex structures
            elif isinstance(v, (dict, list)):
                clean[k] = json.dumps(v)
            # Handle everything else (if not None)
            elif v is not None:
                clean[k] = v
        return clean
