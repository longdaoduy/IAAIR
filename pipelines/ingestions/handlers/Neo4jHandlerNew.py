"""
Neo4j Handler for paper ingestion pipeline.

This module provides a high-level interface for uploading papers to Neo4j
and retrieving paper information.
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime

from clients.graph_store.Neo4jClient import Neo4jClient
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.nodes.Venue import Venue

logger = logging.getLogger(__name__)


class Neo4jHandler:
    """Handler for Neo4j database operations in the ingestion pipeline."""
    
    def __init__(self):
        """Initialize Neo4j handler."""
        self.neo4j_client: Optional[Neo4jClient] = None

    async def _initialize_neo4j_client(self) -> Optional[Neo4jClient]:
        """Initialize Neo4j client connection."""
        try:
            if not self.neo4j_client:
                self.neo4j_client = Neo4jClient()
                await self.neo4j_client.connect()
            return self.neo4j_client
        except Exception as e:
            print(f"❌ Failed to initialize Neo4j client: {e}")
            self.neo4j_client = None
            return None

    async def upload_papers_to_neo4j(self, papers_data: List[Dict]) -> bool:
        """
        Upload papers data to Neo4j database.
        
        Args:
            papers_data: List of paper data dictionaries from fetch_papers()
            
        Returns:
            bool: Success status
        """
        if not papers_data:
            print("No papers data to upload")
            return False

        print(f"Starting upload of {len(papers_data)} papers to Neo4j...")

        try:
            # Initialize Neo4j client
            client = await self._initialize_neo4j_client()
            if not client:
                return False

            # Keep track of uploaded entities
            uploaded_papers = 0
            uploaded_authors = 0
            created_citations = 0

            for i, paper_data in enumerate(papers_data):
                if i % 10 == 0:
                    print(f"Processing paper {i + 1}/{len(papers_data)}...")

                try:
                    # Use the Neo4j client's store_paper method
                    paper = paper_data["paper"]
                    authors = paper_data["authors"]
                    citations = [cid for cid in paper_data["citations"]
                                 if self._paper_exists_in_dataset(cid, papers_data)]

                    # Store paper with all its relationships
                    await client.store_paper(
                        paper=paper,
                        authors=authors,
                        venue=None,  # No venue data from OpenAlex for now
                        citations=citations
                    )

                    uploaded_papers += 1
                    uploaded_authors += len(authors)
                    created_citations += len(citations)

                except Exception as e:
                    print(f"Error processing paper {paper.id}: {e}")
                    continue

            print(f"\n=== Neo4j Upload Summary ===")
            print(f"Papers uploaded: {uploaded_papers}")
            print(f"Authors uploaded: {uploaded_authors}")
            print(f"Citation relationships created: {created_citations}")
            print("✅ Successfully uploaded papers to Neo4j!")

            return True

        except Exception as e:
            print(f"❌ Failed to upload papers to Neo4j: {e}")
            return False

        finally:
            if self.neo4j_client:
                await self.neo4j_client.close()
                self.neo4j_client = None

    def upload_papers_to_neo4j_sync(self, papers_data: List[Dict]) -> bool:
        """
        Synchronous wrapper for uploading papers to Neo4j.
        
        Args:
            papers_data: List of paper data dictionaries from fetch_papers()
            
        Returns:
            bool: Success status
        """
        return asyncio.run(self.upload_papers_to_neo4j(papers_data))

    def _paper_exists_in_dataset(self, paper_id: str, papers_data: List[Dict]) -> bool:
        """Check if a paper ID exists in the current dataset."""
        return any(pd["paper"].id == paper_id for pd in papers_data)

    async def get_papers_by_ids(self, paper_ids: List[str]) -> List[Dict]:
        """
        Retrieve detailed information for papers by their IDs.
        
        Args:
            paper_ids: List of paper IDs to retrieve
            
        Returns:
            List of paper dictionaries with detailed information
        """
        if not paper_ids:
            return []

        try:
            # Initialize Neo4j client
            client = await self._initialize_neo4j_client()
            if not client:
                return []

            # Use the Neo4jClient's get_papers_by_ids method
            papers = await client.get_papers_by_ids(paper_ids)
            return papers

        except Exception as e:
            logger.error(f"Failed to get papers by IDs: {e}")
            return []

    def get_papers_by_ids_sync(self, paper_ids: List[str]) -> List[Dict]:
        """Synchronous wrapper for getting papers by IDs."""
        return asyncio.run(self.get_papers_by_ids(paper_ids))

    async def close(self):
        """Close the Neo4j client connection."""
        if self.neo4j_client:
            await self.neo4j_client.close()
            self.neo4j_client = None