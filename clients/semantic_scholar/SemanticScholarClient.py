"""
Semantic Scholar API client for enriching paper data.

This module provides functionality to fetch additional paper information
from Semantic Scholar API, particularly abstracts that may be missing
from OpenAlex data.
"""

import requests
import time
import json
from typing import Dict, List, Optional
from utils.similarity import compute_confidence
from models.configurators.SemanticScholarConfig import SemanticScholarConfig


class SemanticScholarClient:
    """Client for interacting with Semantic Scholar API."""
    
    def __init__(self):
        self.config = SemanticScholarConfig()
    
    def make_request(self, endpoint: str, params: Dict = None, delay: bool = True) -> Dict:
        """
        Make a request to the Semantic Scholar API with error handling and rate limiting.

        Args:
            endpoint: API endpoint (e.g., 'paper/search', 'paper/batch')
            params: Query parameters
            delay: Whether to add delay for rate limiting

        Returns:
            JSON response as dictionary
        """
        if params is None:
            params = {}

        # Build URL - handle both full URLs and endpoint paths
        if endpoint.startswith('http'):
            url = endpoint
        else:
            # Ensure proper URL construction with /v1/ in the path
            base_url = self.config.BASE_URL.rstrip('/')
            endpoint = endpoint.lstrip('/')
            url = f"{base_url}/{endpoint}"

        # Prepare headers with correct API key format
        headers = {
            "x-api-key": self.config.API_KEY,
            "Accept": "application/json",
            "User-Agent": "KnowledgeFabric/1.0"
        }

        try:
            # Rate limiting
            if delay:
                time.sleep(self.config.request_delay)

            # For paper search, manually construct URL to preserve spaces in query
            if "paper/search" in endpoint and "query" in params:
                query = params["query"]
                other_params = {k: v for k, v in params.items() if k != "query"}
                
                # Build query string manually without encoding the query
                param_parts = [f"query={query}"]
                for key, value in other_params.items():
                    param_parts.append(f"{key}={value}")
                
                full_url = f"{url}?{'&'.join(param_parts)}"
                print(full_url)
                response = requests.get(full_url, headers=headers)
            else:
                # Make regular request for other endpoints
                response = requests.get(url, headers=headers, params=params)
            
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"Status code: {e.response.status_code}")
                print(f"Response text: {e.response.text[:200]}...")
                if e.response.status_code == 429:
                    print("Rate limit exceeded. Waiting longer...")
                    time.sleep(10)  # Wait 10 seconds on rate limit
            return None

    def get_paper_by_title(self, title: str, fields: List[str] = None, limit: int = 1) -> Optional[Dict]:
        """
        Search for a paper by title using Semantic Scholar search.
        
        Args:
            title: Title of the paper (used as-is without processing)
            fields: List of fields to retrieve
            limit: Maximum number of results (default 1)
            
        Returns:
            First matching paper data dictionary or None if not found
        """
        if not title or len(title.strip()) < 10:
            return None
            
        if fields is None:
            fields = ["paperId", "title", "abstract", "year", "authors", "citationCount","externalIds"]
        
        # Use the title exactly as provided, no processing
        params = {
            "query": title,
            "limit": limit,
            "fields": ",".join(fields)
        }
        
        try:
            response = self.make_request("paper/search", params)
            if response and "data" in response and len(response["data"]) > 0:
                # Return the first result without similarity checking
                return response["data"][0] if response["data"][0] else None
            
            return None
            
        except Exception as e:
            print(f"Error searching paper by title '{title[:50]}...': {e}")
            return None
    
    def enrich_papers_with_abstracts(self, papers_data: List[Dict], save_progress: bool = True) -> List[Dict]:
        """
        Enrich papers data with abstracts from Semantic Scholar.
        
        Args:
            papers_data: List of paper data from OpenAlex
            save_progress: Whether to save progress periodically
            
        Returns:
            Enhanced papers data with abstracts from Semantic Scholar
        """
        print(f"Starting abstract enrichment for {len(papers_data)} papers...")
        
        enriched_papers = []
        enriched_count = 0
        failed_count = 0
        
        for i, paper_data in enumerate(papers_data):
            if i % 10 == 0:
                print(f"Processing paper {i+1}/{len(papers_data)} (enriched: {enriched_count}, failed: {failed_count})")
            
            paper = paper_data["paper"]
            enhanced_paper_data = paper_data.copy()  # Start with original data
            
            # Skip if already has abstract
            if paper.abstract and paper.abstract.strip():
                enriched_papers.append(enhanced_paper_data)
                continue
            
            # Try to fetch from Semantic Scholar
            s2_paper = None
            
            # First try by DOI if available
            # if paper.doi:
            #     s2_paper = self.get_paper_by_doi(paper.doi)
            
            # If DOI search failed, try by title
            if not s2_paper and paper.title:
                s2_paper = self.get_paper_by_title(paper.title)
            
            # Enrich the paper if we found it in Semantic Scholar
            if s2_paper and s2_paper.get("abstract"):
                # Update the abstract
                paper.abstract = s2_paper["abstract"]
                enhanced_paper_data["paper"] = paper
                
                # Store additional Semantic Scholar metadata
                semantic_scholar_data = {
                    "semantic_scholar_id": s2_paper.get("paperId"),
                    "citationCount": s2_paper.get("citationCount", 0),
                    "title": s2_paper.get("title"),
                    "doi": s2_paper.get("externalIds").get("DOI"),
                    "mag": s2_paper.get("externalIds").get("MAG"),
                    "pmid": s2_paper.get("externalIds").get("PMID"),
                }

                enhanced_paper_data["paper"].metadata.update(semantic_scholar_data)
                enhanced_paper_data["paper"].metadata["confidence"] = compute_confidence(paper_data["paper"],semantic_scholar_data)

                enriched_count += 1
                print(f"  ✅ Enriched: {paper.title[:60]}...")
                
            else:
                failed_count += 1
                print(f"  ❌ No abstract found for: {paper.title[:60]}...")
                print(f"  ❌ No abstract found for: {paper.title[:60]}...")

            enriched_papers.append(enhanced_paper_data)
            
            # Save progress every 50 papers
            if save_progress and (i + 1) % 50 == 0:
                self._save_progress(enriched_papers, f"enriched_papers_progress_{i+1}.json")
        
        print(f"\n=== Abstract Enrichment Summary ===")
        print(f"Total papers processed: {len(papers_data)}")
        print(f"Papers enriched with abstracts: {enriched_count}")
        print(f"Papers without abstracts found: {failed_count}")
        print(f"Success rate: {(enriched_count / len(papers_data) * 100):.1f}%")
        
        return enriched_papers
    
    def _save_progress(self, papers_data: List[Dict], filename: str):
        """Save progress to JSON file."""
        try:
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
                
                # Add Semantic Scholar data if available
                if "semantic_scholar" in paper_data:
                    json_paper["semantic_scholar"] = paper_data["semantic_scholar"]
                
                json_data.append(json_paper)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"Progress saved to {filename}")
            
        except Exception as e:
            print(f"Error saving progress: {e}")
