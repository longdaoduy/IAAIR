"""
OpenAlex API client for fetching academic papers and related data.

This module provides functionality to interact with OpenAlex API
and extract paper, author, and citation information.
"""

import json
import os
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
from models.schemas.nodes.Institution import Institution


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

    @staticmethod
    def extract_paper_data(work: Dict) -> Optional[Paper]:
        """Extract paper data from OpenAlex work response."""
        try:
            # Check if work is None or not a dict
            if not work or not isinstance(work, dict):
                return None
                
            # Extract basic paper information
            title = work.get('title', '').strip()
            if not title:
                return None

            abstract = work.get('abstract', '') or ''
            doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None

            # Extract PMID from external IDs
            pmid = None
            external_ids = work.get('ids', {})
            if external_ids and isinstance(external_ids, dict) and external_ids.get('pmid'):
                pmid = external_ids.get('pmid').replace('https://pubmed.ncbi.nlm.nih.gov/', '')

            # Extract publication year
            pub_date = None
            if work.get('publication_year'):
                try:
                    pub_date = datetime(work['publication_year'], 1, 1)
                except:
                    pass

            # Extract PDF URL from primary_location or best_oa_location
            pdf_url = None
            primary_location = work.get('primary_location', {})
            if primary_location and isinstance(primary_location, dict) and primary_location.get('pdf_url'):
                pdf_url = primary_location.get('pdf_url')
            else:
                best_oa_location = work.get('best_oa_location', {})
                if best_oa_location and isinstance(best_oa_location, dict) and best_oa_location.get('pdf_url'):
                    pdf_url = best_oa_location.get('pdf_url')
                else:
                    # Also check content_urls for direct PDF access
                    content_urls = work.get('content_urls', {})
                    if content_urls and isinstance(content_urls, dict) and content_urls.get('pdf'):
                        pdf_url = content_urls.get('pdf')

            # Create Paper object
            paper = Paper(
                id=work.get('id', '').replace('https://openalex.org/', ''),
                title=title,
                abstract=abstract,
                publication_date=pub_date,
                doi=doi,
                pmid=pmid,
                pdf_url=pdf_url,
                cited_by_count=work.get('cited_by_count', 0) or 0,
                source="OpenAlex",
                metadata={
                    'openalex_id': work.get('id', '').replace('https://openalex.org/', ''),
                }
            )

            return paper

        except Exception as e:
            print(f"Error extracting paper data: {e}")
            return None

    @staticmethod
    def extract_authors(work: Dict) -> List[Author]:
        """Extract author information from OpenAlex work response."""
        authors = []
        
        # Check if work is None or not a dict
        if not work or not isinstance(work, dict):
            return authors

        for authorship in work.get('authorships', []):
            author_data = authorship.get('author', {})

            if not author_data.get('display_name'):
                continue
            id = (author_data.get('id') or '').replace('https://openalex.org/', '')
            if not id:
                continue

            author = Author(
                id=id,
                name=author_data.get('display_name', ''),
                orcid=author_data.get('orcid', '').replace('https://orcid.org/', '') if author_data.get(
                    'orcid') else None,
                metadata={
                    'openalex_id': (author_data.get('id') or '').replace('https://openalex.org/', '')
                }
            )

            authors.append(author)

        return authors

    @staticmethod
    def extract_citations(work: Dict) -> List[str]:
        """Extract citation information (referenced works) from OpenAlex work response."""
        citations = []
        
        # Check if work is None or not a dict
        if not work or not isinstance(work, dict):
            return citations

        for ref_work in work.get('referenced_works', []):
            if ref_work:
                # Clean the OpenAlex ID
                citation_id = ref_work.replace('https://openalex.org/', '')
                citations.append(citation_id)

        return citations

    @staticmethod
    def extract_venue(work: Dict) -> Optional[Venue]:
        """Extract venue information from OpenAlex work response."""
        try:
            # Check if work is None or not a dict
            if not work or not isinstance(work, dict):
                return None
                
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

    @staticmethod
    def extract_institutions(work: Dict) -> List[Institution]:
        """Extract institution information from OpenAlex work response for authors with null IDs."""
        institutions = []
        seen_institutions = set()  # To avoid duplicates
        
        # Check if work is None or not a dict
        if not work or not isinstance(work, dict):
            return institutions

        for authorship in work.get('authorships', []):
            author_data = authorship.get('author', {})

            # Only process authors with null IDs
            if author_data.get('id') is None:
                # Extract institutions associated with this author
                for institution_data in authorship.get('institutions', []):
                    institution_id = institution_data.get('id', '').replace('https://openalex.org/', '')

                    # Skip if we've already processed this institution
                    if institution_id in seen_institutions or not institution_id:
                        continue

                    seen_institutions.add(institution_id)

                    # Extract country from country_code
                    country = None
                    country_code = institution_data.get('country_code')
                    if country_code:
                        # Map common country codes to full names
                        country_mapping = {
                            'JP': 'Japan',
                            'US': 'United States',
                            'GB': 'United Kingdom',
                            'DE': 'Germany',
                            'FR': 'France',
                            'CN': 'China',
                            'CA': 'Canada',
                            'AU': 'Australia',
                            'IT': 'Italy',
                            'ES': 'Spain',
                            'NL': 'Netherlands',
                            'SE': 'Sweden',
                            'CH': 'Switzerland',
                            'KR': 'South Korea',
                            'IN': 'India',
                            'BR': 'Brazil'
                        }
                        country = country_mapping.get(country_code, country_code)

                    institution = Institution(
                        id=institution_id,
                        name=institution_data.get('display_name', ''),
                        country=country,
                        type=institution_data.get('type', 'unknown'),
                        metadata={
                            'openalex_id': institution_data.get('id', ''),
                            'ror': institution_data.get('ror', ''),
                            'country_code': country_code,
                            'lineage': institution_data.get('lineage', []),
                            'associated_author': author_data.get('display_name', '')
                        }
                    )

                    institutions.append(institution)

        return institutions

    # Cursor state file for resuming pagination
    CURSOR_STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "openalex_cursor.json")

    def _load_cursor_state(self) -> Optional[Dict]:
        """Load the saved cursor state from disk."""
        try:
            path = os.path.normpath(self.CURSOR_STATE_FILE)
            if os.path.exists(path):
                with open(path, 'r') as f:
                    state = json.load(f)
                print(f"📄 Loaded cursor state: page {state.get('total_pages_fetched', 0)} pages, "
                      f"{state.get('total_papers_fetched', 0)} papers fetched so far")
                return state
        except Exception as e:
            print(f"⚠️ Failed to load cursor state: {e}")
        return None

    def _save_cursor_state(self, state: Dict):
        """Save the cursor state to disk."""
        try:
            path = os.path.normpath(self.CURSOR_STATE_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            print(f"💾 Saved cursor state (cursor={state.get('cursor', 'N/A')}, "
                  f"total={state.get('total_papers_fetched', 0)} papers)")
        except Exception as e:
            print(f"⚠️ Failed to save cursor state: {e}")

    def _clear_cursor_state(self):
        """Clear the saved cursor state."""
        try:
            path = os.path.normpath(self.CURSOR_STATE_FILE)
            if os.path.exists(path):
                os.remove(path)
                print("🗑️ Cleared cursor state")
        except Exception as e:
            print(f"⚠️ Failed to clear cursor state: {e}")

    def fetch_papers(self, count: int = 1000, filters: Dict = None, resume: bool = True) -> List[Dict]:
        """
        Fetch papers from OpenAlex API using cursor-based pagination.
        
        Supports resuming from the last saved cursor position so that
        subsequent calls continue fetching new papers instead of
        re-fetching from the beginning.
        
        Args:
            count: Number of papers to fetch in this batch
            filters: Additional filters for the API request
            resume: If True, continue from the last saved cursor position
            
        Returns:
            List of dictionaries containing paper data with authors and citations
        """
        papers_data = []
        per_page = 25  # OpenAlex default, max 200

        base_params = {
            "filter": "has_doi:true,has_pdf_url:true",
            "select": "id,title,publication_year,doi,ids,authorships,referenced_works,cited_by_count,primary_location,best_oa_location,locations,content_urls",
            "per-page": per_page
        }

        # Add custom filters if provided
        # OpenAlex filters must go inside the "filter" query parameter as
        # comma-separated key:value pairs (e.g. "has_doi:true,has_pdf:true").
        # Only recognized top-level params (select, sort, search, q, mailto,
        # per-page, sample, seed, group_by) are kept separate.
        ALLOWED_TOP_LEVEL = {
            "select", "sort", "search", "q", "mailto",
            "per-page", "per_page", "sample", "seed",
            "group_by", "group-by", "group_bys", "group-bys",
            "format", "data_version", "data-version"
        }
        if filters:
            for key, value in filters.items():
                if key == "filter":
                    # Merge into the existing filter string
                    base_params["filter"] = f"{base_params['filter']},{value}"
                elif key in ALLOWED_TOP_LEVEL:
                    base_params[key] = value
                else:
                    # Treat as an OpenAlex filter condition (e.g. has_pdf:true)
                    base_params["filter"] = f"{base_params['filter']},{key}:{value}"

        # Load cursor state for resuming
        cursor = "*"  # OpenAlex initial cursor
        total_papers_fetched_before = 0
        total_pages_fetched_before = 0

        if resume:
            saved_state = self._load_cursor_state()
            if saved_state and saved_state.get("cursor"):
                # Check if filters match — only resume if same query
                saved_filter = saved_state.get("filter", "")
                current_filter = base_params.get("filter", "")
                if saved_filter == current_filter:
                    cursor = saved_state["cursor"]
                    total_papers_fetched_before = saved_state.get("total_papers_fetched", 0)
                    total_pages_fetched_before = saved_state.get("total_pages_fetched", 0)
                    print(f"▶️ Resuming from cursor (already fetched {total_papers_fetched_before} papers across {total_pages_fetched_before} pages)")
                else:
                    print(f"🔄 Filters changed — starting fresh (old: '{saved_filter}', new: '{current_filter}')")
                    self._clear_cursor_state()

        pages_needed = (count + per_page - 1) // per_page  # ceiling division
        print(f"Fetching {count} papers from OpenAlex (~{pages_needed} pages)...")

        pages_fetched = 0
        for page_num in tqdm(range(1, pages_needed + 1)):
            params = dict(base_params)
            params["cursor"] = cursor

            print(f"Fetching page {page_num}/{pages_needed} (global page {total_pages_fetched_before + page_num})...")

            response = self.make_request("works", params)

            if not response or "results" not in response:
                print(f"Failed to fetch page {page_num}")
                break

            results = response["results"]
            if not results:
                print("No more results from OpenAlex — reached end of data")
                break

            for work in results:
                if len(papers_data) >= count:
                    break

                # Skip None or invalid work objects
                if not work or not isinstance(work, dict):
                    print("Skipping invalid work object (None or not dict)")
                    continue

                # Extract paper data
                try:
                    paper = self.extract_paper_data(work)
                    if not paper:
                        continue

                    # Extract authors
                    authors = self.extract_authors(work)

                    # Extract citations
                    citations = self.extract_citations(work)

                    # Extract venue
                    venue = self.extract_venue(work)

                    # Extract institutions for authors with null IDs
                    institutions = self.extract_institutions(work)

                    # Store all data together
                    paper_data = {
                        "paper": paper,
                        "authors": authors,
                        "citations": citations,
                        "venue": venue,
                        "institutions": institutions,
                        "cited_by_count": work.get('cited_by_count', 0)
                    }

                    papers_data.append(paper_data)
                
                except Exception as e:
                    print(f"Error processing work: {e}")
                    continue

            pages_fetched += 1

            # Update cursor for next page
            meta = response.get("meta", {})
            next_cursor = meta.get("next_cursor")
            if not next_cursor:
                print("No next cursor — reached end of OpenAlex results")
                break
            cursor = next_cursor

            # Save cursor state after each successful page
            self._save_cursor_state({
                "cursor": cursor,
                "filter": base_params.get("filter", ""),
                "total_papers_fetched": total_papers_fetched_before + len(papers_data),
                "total_pages_fetched": total_pages_fetched_before + pages_fetched,
                "last_updated": datetime.now().isoformat(),
                "per_page": per_page
            })

            if len(papers_data) >= count:
                break

        print(f"✅ Successfully fetched {len(papers_data)} papers in this batch "
              f"(total across all sessions: {total_papers_fetched_before + len(papers_data)})")
        return papers_data
