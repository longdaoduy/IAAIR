from typing import Dict, List

from clients.graph.Neo4jClient import Neo4jClient
from models.schemas.nodes import Institution, Figure, Table


class GraphNeo4jHandler:
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
            uploaded_figures = 0
            uploaded_tables = 0
            uploaded_institutions = 0

            for i, paper_data in enumerate(papers_data):
                if i % 10 == 0:
                    print(f"Processing paper {i + 1}/{len(papers_data)}...")

                try:
                    # Extract data
                    paper = paper_data["paper"]
                    authors = paper_data["authors"]
                    citations = [cid for cid in paper_data["citations"]
                                 if self._paper_exists_in_dataset(cid, papers_data)]
                    venue = paper_data.get("venue")
                    
                    # Extract new entities if available
                    institutions = paper_data.get("institutions", [])
                    figures = paper_data.get("figures", [])
                    tables = paper_data.get("tables", [])

                    # Store paper with all its relationships and content
                    await client.store_paper_with_content(
                        paper=paper,
                        authors=authors,
                        venue=venue,
                        citations=citations,
                        institutions=institutions,
                        figures=figures,
                        tables=tables
                    )

                    uploaded_papers += 1
                    uploaded_authors += len(authors)
                    created_citations += len(citations)
                    uploaded_institutions += len(institutions)
                    uploaded_figures += len(figures)
                    uploaded_tables += len(tables)

                except Exception as e:
                    print(f"Error processing paper {paper.id}: {e}")
                    continue

            print(f"\n=== Neo4j Upload Summary ===")
            print(f"Papers uploaded: {uploaded_papers}")
            print(f"Authors uploaded: {uploaded_authors}")
            print(f"Institutions uploaded: {uploaded_institutions}")
            print(f"Figures uploaded: {uploaded_figures}")
            print(f"Tables uploaded: {uploaded_tables}")
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
            print(f"Failed to get papers by IDs: {e}")
            return []

    async def close(self):
        """Close the Neo4j client connection."""
        if self.neo4j_client:
            await self.neo4j_client.close()
            self.neo4j_client = None