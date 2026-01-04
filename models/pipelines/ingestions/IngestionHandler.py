from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio
from models.configurators.GraphDBConfig import GraphDBConfig
from clients.graph_store.Neo4jClient import Neo4jClient
from clients.semantic_scholar.SemanticScholarClient import SemanticScholarClient
from clients.semantic_scholar.OpenAlexClient import OpenAlexClient


class IngestionHandler():
    def __init__(self):
        self.nums_papers_to_pull = 1
        self.neo4j_client = None
        self.semantic_scholar_client = SemanticScholarClient()
        self.openalex_client = OpenAlexClient()

    async def _initialize_neo4j_client(self):
        """Initialize and connect to Neo4j client."""
        if self.neo4j_client is None:
            self.neo4j_client = Neo4jClient()
            await self.neo4j_client.connect()
        return self.neo4j_client

    def test_api_connection(self):
        """Test the API connection with a simple request"""
        return self.openalex_client.test_connection()

    def fetch_papers(self, count: int = 1000, filters: Dict = None) -> List[Dict]:
        """Fetch papers from OpenAlex API.
        
        Args:
            count: Number of papers to fetch (500-1000)
            filters: Additional filters for the API request
            
        Returns:
            List of dictionaries containing paper data with authors and citations
        """
        return self.openalex_client.fetch_papers(count, filters)

    def save_papers_to_json(self, papers_data: List[Dict], filename: str = "openalex_papers.json"):
        """Save fetched papers data to a JSON file."""
        # Convert dataclass objects to dictionaries for JSON serialization
        json_data = []

        for paper_data in papers_data:
            json_paper = {
                "paper": {
                    "id": paper_data["paper"].id,
                    "title": paper_data["paper"].title,
                    "abstract": paper_data["paper"].abstract,
                    "publication_date": paper_data["paper"].publication_date.isoformat() if paper_data[
                        "paper"].publication_date else None,
                    "doi": paper_data["paper"].doi,
                    "metadata": paper_data["paper"].metadata,
                    "venue": paper_data["paper"].venue,
                    "ingested_at": paper_data["paper"].ingested_at.isoformat()
                },
                "authors": [
                    {
                        "id": author.id,
                        "name": author.name,
                        "orcid": author.orcid
                    } for author in paper_data["authors"]
                ],
                "citations": paper_data["citations"],
                "cited_by_count": paper_data["cited_by_count"]
            }
            json_data.append(json_paper)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(json_data)} papers to {filename}")

    def pull_OpenAlex_Paper(self, count: int = None, filters: Dict = None, save_to_file: bool = True,
                            upload_to_neo4j: bool = False) -> List[Dict]:
        """Main method to ingest papers from OpenAlex.
        
        Args:
            count: Number of papers to fetch (500-1000), defaults to nums_papers_to_pull
            filters: Additional filters for the API request
            save_to_file: Whether to save results to JSON file
            upload_to_neo4j: Whether to upload results to Neo4j database
            
        Returns:
            List of paper data with authors and citations
        """
        if count is None:
            count = self.nums_papers_to_pull

        print(f"Starting paper ingestion from OpenAlex (target: {count} papers)")

        # # Test API connection first
        # if not self.test_api_connection():
        #     print("API connection failed. Aborting ingestion.")
        #     return []

        # Fetch papers
        papers_data = self.fetch_papers(count, filters)

        # Save to file if requested
        if save_to_file and papers_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openalex_papers_{timestamp}.json"
            self.save_papers_to_json(papers_data, filename)

        # Upload to Neo4j if requested
        if upload_to_neo4j and papers_data:
            print("\nUploading papers to Neo4j...")
            success = self.upload_papers_to_neo4j_sync(papers_data)
            if success:
                print("✅ Papers successfully uploaded to Neo4j!")
            else:
                print("❌ Failed to upload papers to Neo4j")

        # Print summary
        print(f"\n=== Ingestion Summary ===")
        print(f"Papers fetched: {len(papers_data)}")
        if papers_data:
            total_authors = sum(len(pd['authors']) for pd in papers_data)
            total_citations = sum(len(pd['citations']) for pd in papers_data)
            print(f"Authors extracted: {total_authors}")
            print(f"Citations extracted: {total_citations}")
            print(f"Average citations per paper: {total_citations / len(papers_data):.1f}")

        return papers_data

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

    def _paper_exists_in_dataset(self, paper_id: str, papers_data: List[Dict]) -> bool:
        """Check if a paper ID exists in the current dataset."""
        return any(pd["paper"].id == paper_id for pd in papers_data)

    def upload_papers_to_neo4j_sync(self, papers_data: List[Dict]) -> bool:
        """
        Synchronous wrapper for uploading papers to Neo4j.
        
        Args:
            papers_data: List of paper data dictionaries from fetch_papers()
            
        Returns:
            bool: Success status
        """
        return asyncio.run(self.upload_papers_to_neo4j(papers_data))
    
    def enrich_papers_with_semantic_scholar(self, papers_data: List[Dict], save_to_file: bool = True) -> List[Dict]:
        """
        Enrich papers data with abstracts and additional information from Semantic Scholar.
        
        Args:
            papers_data: List of paper data dictionaries from OpenAlex
            save_to_file: Whether to save the enriched results to a file
            
        Returns:
            List of enriched paper data
        """
        print(f"Starting paper enrichment with Semantic Scholar...")
        
        # Use the Semantic Scholar client to enrich papers
        enriched_papers = self.semantic_scholar_client.enrich_papers_with_abstracts(papers_data)
        
        # Save to file if requested
        if save_to_file and enriched_papers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enriched_openalex_papers_{timestamp}.json"
            self._save_enriched_papers_to_json(enriched_papers, filename)
        
        return enriched_papers
    
    def _save_enriched_papers_to_json(self, papers_data: List[Dict], filename: str):
        """Save enriched papers data to a JSON file."""
        # Convert dataclass objects to dictionaries for JSON serialization
        json_data = []

        for paper_data in papers_data:
            json_paper = {
                "paper": {
                    "id": paper_data["paper"].id,
                    "title": paper_data["paper"].title,
                    "abstract": paper_data["paper"].abstract,
                    "publication_date": paper_data["paper"].publication_date.isoformat() if paper_data["paper"].publication_date else None,
                    "doi": paper_data["paper"].doi,
                    "ingested_at": paper_data["paper"].ingested_at.isoformat(),
                    "metadata": paper_data["paper"].metadata
                },
                "authors": [
                    {
                        "id": author.id,
                        "name": author.name,
                        "orcid": author.orcid
                    } for author in paper_data["authors"]
                ],
                "citations": paper_data["citations"],
                "cited_by_count": paper_data["cited_by_count"]
            }
            
            # Add Semantic Scholar enrichment data if available
            if "semantic_scholar" in paper_data:
                json_paper["semantic_scholar"] = paper_data["semantic_scholar"]
            
            json_data.append(json_paper)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(json_data)} enriched papers to {filename}")
    
    def load_papers_from_json(self, filename: str) -> List[Dict]:
        """
        Load papers data from JSON file and convert back to proper format.
        
        Args:
            filename: JSON file containing papers data
            
        Returns:
            List of paper data dictionaries
        """
        print(f"Loading papers data from {filename}...")
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            papers_data = []
            for item in json_data:
                # Reconstruct Paper object
                paper_data = item["paper"]
                paper = Paper(
                    id=paper_data["id"],
                    title=paper_data["title"],
                    abstract=paper_data["abstract"],
                    publication_date=datetime.fromisoformat(paper_data["publication_date"]) if paper_data["publication_date"] else None,
                    doi=paper_data["doi"],
                    source=paper_data["source"],
                    ingested_at=datetime.fromisoformat(paper_data["ingested_at"]),
                    last_updated=datetime.now()
                )
                
                # Reconstruct Author objects
                authors = []
                for author_data in item["authors"]:
                    author = Author(
                        id=author_data["id"],
                        name=author_data["name"],
                        orcid=author_data["orcid"]
                    )
                    authors.append(author)
                
                # Reconstruct full paper data structure
                reconstructed_data = {
                    "paper": paper,
                    "authors": authors,
                    "citations": item["citations"],
                    "cited_by_count": item["cited_by_count"]
                }
                
                # Add Semantic Scholar data if available
                if "semantic_scholar" in item:
                    reconstructed_data["semantic_scholar"] = item["semantic_scholar"]
                
                papers_data.append(reconstructed_data)
            
            print(f"Loaded {len(papers_data)} papers from JSON file")
            return papers_data
            
        except Exception as e:
            print(f"Error loading papers from {filename}: {e}")
            return []
