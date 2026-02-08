"""
Graph Query Handler for Neo4j Cypher queries.

This module provides functionality to execute complex Cypher queries
against the Neo4j graph database, enabling graph-based paper retrievals
and relationship exploration.
"""

from typing import Dict, List, Optional, Any
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
from models.configurators.GraphDBConfig import GraphDBConfig


class GraphQueryHandler:
    """Handler for executing Cypher queries against Neo4j graph database."""
    
    def __init__(self, config: Optional[GraphDBConfig] = None):
        """Initialize the graph query handler.
        
        Args:
            config: Neo4j configuration. If None, loads from environment.
        """
        self.config = config or GraphDBConfig.from_env()
        self.driver = None
        self.logger = logging.getLogger(__name__)
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            self.logger.info("✅ Connected to Neo4j database")
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.driver:
            self.driver.close()
            self.logger.info("🔌 Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            raise RuntimeError("Database connection not established")
        
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                records = []
                for record in result:
                    # Convert neo4j.Record to dictionary
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Handle Neo4j node/relationship objects
                        if hasattr(value, '_properties'):
                            record_dict[key] = dict(value._properties)
                            record_dict[key]['_labels'] = list(value.labels) if hasattr(value, 'labels') else []
                            record_dict[key]['_id'] = value.id if hasattr(value, 'id') else None
                        else:
                            record_dict[key] = value
                    records.append(record_dict)
                
                self.logger.info(f"📊 Query executed successfully, returned {len(records)} records")
                return records
                
        except ClientError as e:
            self.logger.error(f"❌ Cypher query error: {e}")
            raise
        except ServiceUnavailable as e:
            self.logger.error(f"❌ Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error executing query: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and node counts.
        
        Returns:
            Dictionary with database statistics
        """
        stats_queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'total_relationships': "MATCH ()-[r]->() RETURN count(r) as count",
            'paper_count': "MATCH (p:Paper) RETURN count(p) as count",
            'author_count': "MATCH (a:Author) RETURN count(a) as count",
            'venue_count': "MATCH (v:Venue) RETURN count(v) as count",
            'citation_count': "MATCH ()-[c:CitedBy]->() RETURN count(c) as count",
            'authorship_count': "MATCH ()-[a:Authored]->() RETURN count(a) as count",
            'publication_count': "MATCH ()-[p:PublishedIn]->() RETURN count(p) as count"
        }
        
        stats = {}
        for stat_name, query in stats_queries.items():
            try:
                result = self.execute_query(query)
                stats[stat_name] = result[0]['count'] if result else 0
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to get {stat_name}: {e}")
                stats[stat_name] = 0
        
        return stats