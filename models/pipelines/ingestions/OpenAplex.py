import requests
import time
from urllib.parse import urljoin
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import asyncio

from models.configurators.OpenAlexConfig import OpenAlexConfig
from models.configurators.GraphDBConfig import GraphDBConfig
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.edges.CitedBy import CitedBy
from clients.graph_store.Neo4jClient import Neo4jClient

class OpenAplex():
    def __init__(self):
        self.config = OpenAlexConfig()
        self.graph_config = GraphDBConfig()
        self.nums_papers_to_pull = 1
        self.neo4j_client = None

    def make_openalex_request(self, endpoint: str, params: Dict = None, delay: bool = True) -> Dict:
        """
        Make a request to the OpenAlex API with error handling and rate limiting.

        Args:
            endpoint: API endpoint (e.g., 'works', 'authors', 'institutions')
            params: Query parameters
            delay: Whether to add delay for rate limiting

        Returns:
            JSON response as dictionary
        """
        if params is None:
            params = {}

        # Build URL
        url = urljoin(self.config.BASE_URL, endpoint)

        try:
            # Rate limiting
            if delay:
                time.sleep(self.config.REQUEST_DELAY)

            # Make request
            response = requests.get(url, headers=self.config.HEADERS, params=params)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            return None

    def test_api_connection(self):
        """Test the API connection with a simple request"""
        print("Testing OpenAlex API connection...")

        response = self.make_openalex_request("works", {"filter": "publication_year:2023", "per-page": 1})

        if response and "results" in response:
            print("✅ API connection successful!")
            print(f"Total works in 2023: {response['meta']['count']:,}")
            return True
        else:
            print("❌ API connection failed!")
            return False

    def _extract_paper_data(self, work: Dict) -> Optional[Paper]:
        """Extract paper data from OpenAlex work response."""
        try:
            # Extract basic paper information
            title = work.get('title', '').strip()
            if not title:
                return None
                
            abstract = work.get('abstract', '') or ''
            doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None
            
            # Extract publication year
            pub_date = None
            if work.get('publication_year'):
                try:
                    pub_date = datetime(work['publication_year'], 1, 1)
                except:
                    pass
            
            # Create Paper object
            paper = Paper(
                id=work.get('id', '').replace('https://openalex.org/', ''),
                title=title,
                abstract=abstract,
                publication_date=pub_date,
                doi=doi,
                source='OpenAlex'
            )
            
            return paper
            
        except Exception as e:
            print(f"Error extracting paper data: {e}")
            return None
    
    def _extract_authors(self, work: Dict) -> List[Author]:
        """Extract author information from OpenAlex work response."""
        authors = []
        
        for authorship in work.get('authorships', []):
            author_data = authorship.get('author', {})
            
            if not author_data.get('display_name'):
                continue
                
            author = Author(
                id=author_data.get('id', '').replace('https://openalex.org/', ''),
                name=author_data.get('display_name', ''),
                orcid=author_data.get('orcid', '').replace('https://orcid.org/', '') if author_data.get('orcid') else None
            )
            
            authors.append(author)
            
        return authors
    
    def _extract_citations(self, work: Dict) -> List[str]:
        """Extract citation information (referenced works) from OpenAlex work response."""
        citations = []
        
        for ref in work.get('referenced_works', []):
            if ref:
                # Clean the OpenAlex ID
                citation_id = ref.replace('https://openalex.org/', '')
                citations.append(citation_id)
                
        return citations
    
    def fetch_papers(self, count: int = 1000, filters: Dict = None) -> List[Dict]:
        """Fetch papers from OpenAlex API.
        
        Args:
            count: Number of papers to fetch (500-1000)
            filters: Additional filters for the API request
            
        Returns:
            List of dictionaries containing paper data with authors and citations
        """
            
        papers_data = []
        per_page = min(200, count)  # OpenAlex max is 200 per page
        pages_needed = (count + per_page - 1) // per_page
        
        base_params = {
            "per-page": per_page,
            "filter": "has_doi:true,publication_year:2023",
            "select": "id,title,abstract,publication_year,doi,authorships,referenced_works,cited_by_count"
        }
        
        # Add custom filters if provided
        if filters:
            for key, value in filters.items():
                if key == "filter":
                    base_params[key] = f"{base_params[key]},{value}"
                else:
                    base_params[key] = value
        
        print(f"Fetching {count} papers from OpenAlex...")
        
        for page in range(1, pages_needed + 1):
            params = dict()
            params["page"] = page
            
            print(f"Fetching page {page}/{pages_needed}...")
            
            response = self.make_openalex_request("works", params)
            
            if not response or "results" not in response:
                print(f"Failed to fetch page {page}")
                continue
                
            for work in response["results"]:
                if len(papers_data) >= count:
                    break
                    
                # Extract paper data
                paper = self._extract_paper_data(work)
                if not paper:
                    continue
                
                # Extract authors
                authors = self._extract_authors(work)
                
                # Extract citations
                citations = self._extract_citations(work)
                
                # Store all data together
                paper_data = {
                    "paper": paper,
                    "authors": authors,
                    "citations": citations,
                    "cited_by_count": work.get('cited_by_count', 0)
                }
                
                papers_data.append(paper_data)
                
            if len(papers_data) >= count:
                break
        
        print(f"Successfully fetched {len(papers_data)} papers")
        return papers_data
    
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
                    "publication_date": paper_data["paper"].publication_date.isoformat() if paper_data["paper"].publication_date else None,
                    "doi": paper_data["paper"].doi,
                    "source": paper_data["paper"].source,
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
    
    def pull_OpenAlex_Paper(self, count: int = None, filters: Dict = None, save_to_file: bool = True, upload_to_neo4j: bool = False) -> List[Dict]:
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
        
        # Test API connection first
        if not self.test_api_connection():
            print("API connection failed. Aborting ingestion.")
            return []
        
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
            print(f"Average citations per paper: {total_citations/len(papers_data):.1f}")
        
        return papers_data
    
    async def _initialize_neo4j_client(self):
        """Initialize and connect to Neo4j client."""
        if self.neo4j_client is None:
            self.neo4j_client = Neo4jClient(self.graph_config)
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
                    print(f"Processing paper {i+1}/{len(papers_data)}...")
                
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