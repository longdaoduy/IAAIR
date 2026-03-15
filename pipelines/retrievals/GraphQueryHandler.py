"""
Graph Query Handler for Neo4j Cypher queries.

This module delegates query execution to Neo4jClient, providing a
backward-compatible interface for the retrieval pipeline.
"""

from typing import Dict, List, Optional
import logging
from clients.neo4j.Neo4jClient import Neo4jClient

class GraphQueryHandler:
    """Handler that delegates Cypher query execution to Neo4jClient."""
    
    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        """Initialize the neo4j query handler.
        
        Args:
            neo4j_client: Shared Neo4jClient instance. If None, creates a new one.
        """
        self.logger = logging.getLogger(__name__)

        self._client = neo4j_client
        # Ensure the sync driver is ready
        self.logger.info("✅ GraphQueryHandler initialized (delegates to Neo4jClient)")
    
    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query via Neo4jClient.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        return self._client.execute_query(query, parameters)