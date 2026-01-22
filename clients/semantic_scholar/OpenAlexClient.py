"""
OpenAlex API client for fetching academic papers and related data.

This module provides functionality to interact with OpenAlex API
and extract paper, author, and citation information.
"""

import requests
import time
from urllib.parse import urljoin
from typing import Dict, List, Optional, Any
from datetime import datetime
from tqdm import tqdm

from models.configurators.OpenAlexConfig import OpenAlexConfig
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author
from models.schemas.nodes.Venue import Venue, VenueType


class OpenAlexClient:
    """Client for interacting with OpenAlex API."""

    def __init__(self):
        self.config = OpenAlexConfig()

    def make_request(self, endpoint: str, params: Dict = None, delay: bool = True) -> Dict:
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

    def test_connection(self) -> bool:
        """Test the API connection with a simple request"""
        print("Testing OpenAlex API connection...")

        response = self.make_request("works", {"filter": "publication_year:2023", "per-page": 1})

        if response and "results" in response:
            print("✅ API connection successful!")
            print(f"Total works in 2023: {response['meta']['count']:,}")
            return True
        else:
            print("❌ API connection failed!")
            return False

    def extract_paper_data(self, work: Dict) -> Optional[Paper]:
        """Extract paper data from OpenAlex work response."""
        try:
            # Extract basic paper information
            title = work.get('title', '').strip()
            if not title:
                return None

            abstract = work.get('abstract', '') or ''
            doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None

            # Extract PMID from external IDs
            pmid = None
            external_ids = work.get('ids', {})
            if external_ids and external_ids.get('pmid'):
                pmid = external_ids.get('pmid').replace('https://pubmed.ncbi.nlm.nih.gov/', '')

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
                pmid=pmid,
                source="OpenAlex",
                metadata={
                    'openalex_id': work.get('id', '').replace('https://openalex.org/', ''),
                }
            )

            return paper

        except Exception as e:
            print(f"Error extracting paper data: {e}")
            return None

    def extract_authors(self, work: Dict) -> List[Author]:
        """Extract author information from OpenAlex work response."""
        authors = []

        for authorship in work.get('authorships', []):
            author_data = authorship.get('author', {})

            if not author_data.get('display_name'):
                continue

            author = Author(
                id=(author_data.get('id') or '').replace('https://openalex.org/', ''),
                name=author_data.get('display_name', ''),
                orcid=author_data.get('orcid', '').replace('https://orcid.org/', '') if author_data.get(
                    'orcid') else None,
                metadata={
                    'openalex_id': (author_data.get('id') or '').replace('https://openalex.org/', '')
                }
            )

            authors.append(author)

        return authors

    def extract_citations(self, work: Dict) -> List[str]:
        """Extract citation information (referenced works) from OpenAlex work response."""
        citations = []

        for ref in work.get('referenced_works', []):
            if ref:
                # Clean the OpenAlex ID
                citation_id = ref.replace('https://openalex.org/', '')
                citations.append(citation_id)

        return citations

    def extract_venue(self, work: Dict) -> Optional[Venue]:
        """Extract venue information from OpenAlex work response."""
        try:
            host_venue = work.get('primary_location', {}) or work.get('best_oa_location', {})
            if not host_venue:
                # Try locations array
                locations = work.get('locations', [])
                if locations:
                    host_venue = locations[0]  # Use first location

            if not host_venue or not host_venue.get('source'):
                return None

            source = host_venue['source']
            venue_name = source.get('display_name', '').strip()
            if not venue_name:
                return None

            # Determine venue type based on OpenAlex type
            type = VenueType.JOURNAL  # default
            openalex_type = source.get('type', '').lower()

            if 'conference' in openalex_type or 'proceedings' in openalex_type:
                type = VenueType.CONFERENCE
            elif 'journal' in openalex_type:
                type = VenueType.JOURNAL
            elif 'repository' in openalex_type and 'arxiv' in venue_name.lower():
                type = VenueType.ARXIV
            elif 'book' in openalex_type:
                type = VenueType.BOOK

            venue = Venue(
                id=source.get('id', '').replace('https://openalex.org/', ''),
                name=venue_name,
                type=type,
                issn=source.get('issn_l') or (source.get('issn', [None])[0] if source.get('issn') else None),
                publisher=source.get('host_organization_name'),
                metadata={
                    'openalex_id': source.get('id', '').replace('https://openalex.org/', ''),
                }
            )

            return venue

        except Exception as e:
            print(f"Error extracting venue data: {e}")
            return None

    def fetch_papers(self, count: int = 1000, filters: Dict = None) -> List[Dict]:
        """
        Fetch papers from OpenAlex API.
        
        Args:
            count: Number of papers to fetch
            filters: Additional filters for the API request
            
        Returns:
            List of dictionaries containing paper data with authors and citations
        """
        papers_data = []
        per_page = min(200, count)  # OpenAlex max is 200 per page
        pages_needed = 2

        base_params = {
            "per-page": per_page,
            "filter": "has_doi:true",
            "select": "id,title,abstract,publication_year,doi,ids,authorships,referenced_works,cited_by_count,primary_location,best_oa_location,locations"
        }

        # Add custom filters if provided
        if filters:
            for key, value in filters.items():
                if key == "filter":
                    base_params[key] = f"{base_params[key]},{value}"
                else:
                    base_params[key] = value

        print(f"Fetching {count} papers from OpenAlex...")

        for page in tqdm(range(1, pages_needed + 1)):
            params = dict()
            params["page"] = page

            print(f"Fetching page {page}/{pages_needed}...")

            response = self.make_request("works", params)

            if not response or "results" not in response:
                print(f"Failed to fetch page {page}")
                continue

            for work in response["results"]:
                if len(papers_data) >= count:
                    break

                # Extract paper data
                paper = self.extract_paper_data(work)
                if not paper:
                    continue

                # Extract authors
                authors = self.extract_authors(work)

                # Extract citations
                citations = self.extract_citations(work)

                # Extract venue
                venue = self.extract_venue(work)

                # Store all data together
                paper_data = {
                    "paper": paper,
                    "authors": authors,
                    "citations": citations,
                    "venue": venue,
                    "cited_by_count": work.get('cited_by_count', 0)
                }

                papers_data.append(paper_data)

            if len(papers_data) >= count:
                break

        print(f"Successfully fetched {len(papers_data)} papers")
        return papers_data
