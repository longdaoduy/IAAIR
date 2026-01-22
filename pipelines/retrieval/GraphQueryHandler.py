"""
Graph Query Handler for Neo4j Cypher queries.

This module provides functionality to execute complex Cypher queries
against the Neo4j graph database, enabling graph-based paper retrieval
and relationship exploration.
"""

from typing import Dict, List, Optional, Any, Union
import logging
from neo4j import GraphDatabase, Record
from neo4j.exceptions import ServiceUnavailable, ClientError
import json
from datetime import datetime

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


class CypherQueryBuilder:
    """Builder class for constructing common Cypher queries."""
    
    @staticmethod
    def find_papers_by_author(author_name: str, limit: int = 10) -> str:
        """Build query to find papers by author name.
        
        Args:
            author_name: Name of the author to search for
            limit: Maximum number of papers to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (a:Author)-[:Authored]->(p:Paper)
        WHERE toLower(a.name) CONTAINS toLower($author_name)
        RETURN a.name as author_name, 
               p.title as paper_title,
               p.id as paper_id,
               p.doi as doi,
               p.publication_date as publication_date,
               p.cited_by_count as cited_by_count
        ORDER BY p.cited_by_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_papers_citing_paper(paper_id: str, limit: int = 10) -> str:
        """Build query to find papers that cite a specific paper.
        
        Args:
            paper_id: ID of the paper to find citations for
            limit: Maximum number of citing papers to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (cited:Paper {{id: $paper_id}})<-[:CitedBy]-(citing:Paper)
        RETURN citing.title as citing_paper_title,
               citing.id as citing_paper_id,
               citing.doi as citing_doi,
               citing.publication_date as citing_date,
               citing.cited_by_count as citing_paper_citations,
               cited.title as cited_paper_title
        ORDER BY citing.cited_by_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_papers_cited_by_paper(paper_id: str, limit: int = 10) -> str:
        """Build query to find papers cited by a specific paper.
        
        Args:
            paper_id: ID of the paper to find citations from
            limit: Maximum number of cited papers to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (citing:Paper {{id: $paper_id}})-[:CitedBy]->(cited:Paper)
        RETURN cited.title as cited_paper_title,
               cited.id as cited_paper_id,
               cited.doi as cited_doi,
               cited.publication_date as cited_date,
               cited.cited_by_count as cited_paper_citations,
               citing.title as citing_paper_title
        ORDER BY cited.cited_by_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_coauthors(author_name: str, limit: int = 10) -> str:
        """Build query to find coauthors of a specific author.
        
        Args:
            author_name: Name of the author to find coauthors for
            limit: Maximum number of coauthors to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (a1:Author)-[:Authored]->(p:Paper)<-[:Authored]-(a2:Author)
        WHERE toLower(a1.name) CONTAINS toLower($author_name) AND a1 <> a2
        RETURN a2.name as coauthor_name,
               a2.id as coauthor_id,
               count(p) as collaboration_count,
               collect(p.title)[0..3] as sample_papers
        ORDER BY collaboration_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_papers_in_venue(venue_name: str, limit: int = 10) -> str:
        """Build query to find papers published in a specific venue.
        
        Args:
            venue_name: Name of the venue to search for
            limit: Maximum number of papers to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Paper)-[:PublishedIn]->(v:Venue)
        WHERE toLower(v.name) CONTAINS toLower($venue_name)
        RETURN p.title as paper_title,
               p.id as paper_id,
               p.doi as doi,
               p.publication_date as publication_date,
               p.cited_by_count as cited_by_count,
               v.name as venue_name,
               v.type as venue_type
        ORDER BY p.cited_by_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_most_cited_papers(limit: int = 10, min_citations: int = 0) -> str:
        """Build query to find most cited papers.
        
        Args:
            limit: Maximum number of papers to return
            min_citations: Minimum citation count threshold
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Paper)
        WHERE p.cited_by_count >= {min_citations}
        RETURN p.title as paper_title,
               p.id as paper_id,
               p.doi as doi,
               p.publication_date as publication_date,
               p.cited_by_count as cited_by_count,
               p.abstract as abstract
        ORDER BY p.cited_by_count DESC
        LIMIT {limit}
        """
    
    @staticmethod
    def find_author_collaboration_network(author_name: str, depth: int = 2, limit: int = 20) -> str:
        """Build query to find author collaboration network.
        
        Args:
            author_name: Name of the central author
            depth: Network depth (1 = direct collaborators, 2 = collaborators of collaborators)
            limit: Maximum number of nodes to return
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH path = (a1:Author)-[:Authored*1..{depth}]-(a2:Author)
        WHERE toLower(a1.name) CONTAINS toLower($author_name) AND a1 <> a2
        WITH a1, a2, length(path) as distance
        RETURN a1.name as central_author,
               a2.name as connected_author,
               a2.id as connected_author_id,
               distance,
               CASE distance 
                 WHEN 2 THEN 'direct_collaborator'
                 WHEN 4 THEN 'collaborator_of_collaborator'
                 ELSE 'distant_collaborator'
               END as relationship_type
        ORDER BY distance, a2.name
        LIMIT {limit}
        """
    
    @staticmethod
    def find_research_trends_by_year(start_year: int, end_year: int) -> str:
        """Build query to analyze research trends by publication year.
        
        Args:
            start_year: Starting year for analysis
            end_year: Ending year for analysis
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH (p:Paper)-[:PublishedIn]->(v:Venue)
        WHERE p.publication_date >= '{start_year}-01-01' 
          AND p.publication_date <= '{end_year}-12-31'
        WITH p, v, toInteger(split(p.publication_date, '-')[0]) as year
        RETURN year,
               count(p) as paper_count,
               avg(p.cited_by_count) as avg_citations,
               collect(DISTINCT v.type)[0..5] as venue_types,
               collect(p.title)[0..3] as sample_titles
        ORDER BY year
        """
    
    @staticmethod
    def find_citation_path(source_paper_id: str, target_paper_id: str, max_depth: int = 3) -> str:
        """Build query to find citation paths between two papers.
        
        Args:
            source_paper_id: ID of the source paper
            target_paper_id: ID of the target paper
            max_depth: Maximum path length to search
            
        Returns:
            Cypher query string
        """
        return f"""
        MATCH path = (source:Paper {{id: $source_paper_id}})-[:CitedBy*1..{max_depth}]->(target:Paper {{id: $target_paper_id}})
        WITH path, length(path) as path_length
        RETURN [node in nodes(path) | {{title: node.title, id: node.id, citations: node.cited_by_count}}] as citation_path,
               path_length,
               [rel in relationships(path) | type(rel)] as relationship_types
        ORDER BY path_length
        LIMIT 5
        """