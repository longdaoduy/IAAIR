from typing import Dict, List, Optional
import asyncio

from models.schemas.nodes import Paper, Author, Venue
from clients.graph_store.Neo4jClient import Neo4jClient


class Neo4jHandler:
    """Handler for Neo4j database operations related to paper ingestion."""
    
    def __init__(self):
        self.neo4j_client = None

    async def _initialize_neo4j_client(self):
        """Initialize and connect to Neo4j client."""
        if self.neo4j_client is None:
            self.neo4j_client = Neo4jClient()
            await self.neo4j_client.connect()
        return self.neo4j_client

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
                        venue=paper_data.get("venue"),
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
        """Retrieve detailed paper information by IDs from Neo4j.
        
        Args:
            paper_ids: List of paper IDs to retrieve
            
        Returns:
            List of paper details with authors, venue, and citation info
        """
        if not paper_ids:
            return []
        
        try:
            # Initialize Neo4j client
            client = await self._initialize_neo4j_client()
            
            papers_data = []
            
            async with client.session() as session:
                for paper_id in paper_ids:
                    # Query to get paper with authors, venue, and citations
                    query = """
                    MATCH (p:Paper {id: $paper_id})
                    OPTIONAL MATCH (p)<-[:AUTHORED]-(a:Author)
                    OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
                    OPTIONAL MATCH (p)-[:CITES]->(cited:Paper)
                    OPTIONAL MATCH (citing:Paper)-[:CITES]->(p)
                    RETURN p,
                           collect(DISTINCT a) as authors,
                           v as venue,
                           collect(DISTINCT cited.id) as citations,
                           collect(DISTINCT citing.id) as cited_by
                    """
                    
                    result = await session.run(query, paper_id=paper_id)
                    record = await result.single()
                    
                    if record:
                        paper_node = record["p"]
                        authors = record["authors"] or []
                        venue = record["venue"]
                        citations = record["citations"] or []
                        cited_by = record["cited_by"] or []
                        
                        # Format paper data
                        paper_data = {
                            "id": paper_node["id"],
                            "title": paper_node.get("title"),
                            "abstract": paper_node.get("abstract"),
                            "doi": paper_node.get("doi"),
                            "publication_date": paper_node.get("publication_date"),
                            "pmid": paper_node.get("pmid"),
                            "arxiv_id": paper_node.get("arxiv_id"),
                            "source": paper_node.get("source"),
                            "metadata": paper_node.get("metadata", {}),
                            "cited_by_count": len(cited_by),
                            "authors": [
                                {
                                    "id": author["id"],
                                    "name": author["name"],
                                    "orcid": author.get("orcid"),
                                    "metadata": author.get("metadata", {})
                                } for author in authors
                            ],
                            "venue": {
                                "id": venue["id"],
                                "name": venue["name"],
                                "type": venue.get("type"),
                                "issn": venue.get("issn"),
                                "publisher": venue.get("publisher")
                            } if venue else None,
                            "citations": citations,
                            "cited_by": cited_by
                        }
                        
                        papers_data.append(paper_data)
            
            return papers_data
            
        except Exception as e:
            print(f"❌ Failed to retrieve papers from Neo4j: {e}")
            return []
        
        finally:
            if self.neo4j_client:
                await self.neo4j_client.close()
                self.neo4j_client = None
    
    def get_papers_by_ids_sync(self, paper_ids: List[str]) -> List[Dict]:
        """Synchronous wrapper for getting papers by IDs."""
        return asyncio.run(self.get_papers_by_ids(paper_ids))

    async def close(self):
        """Close the Neo4j client connection."""
        if self.neo4j_client:
            await self.neo4j_client.close()
            self.neo4j_client = None