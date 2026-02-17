from typing import Dict, List
from datetime import datetime
import json
from models.schemas.nodes import Paper, Author, Venue,VenueType
from clients.metadata.SemanticScholarClient import SemanticScholarClient
from clients.metadata.OpenAlexClient import OpenAlexClient

class IngestionHandler():
    def __init__(self):
        self.semantic_scholar_client = SemanticScholarClient()
        self.openalex_client = OpenAlexClient()

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
                    "doi": paper_data["paper"].doi,
                    "publication_date": paper_data["paper"].publication_date.isoformat() if paper_data[
                        "paper"].publication_date else None,
                    "source": paper_data["paper"].source,
                    "metadata": paper_data["paper"].metadata,
                    "pmid": paper_data["paper"].pmid,
                    "ingested_at": paper_data["paper"].ingested_at.isoformat()
                },
                "venue": {
                    "id": paper_data["venue"].id,
                    "name": paper_data["venue"].name,
                    "type": paper_data["venue"].type.name,
                    "issn": paper_data["venue"].issn,
                    "impact_factor": paper_data["venue"].impact_factor,
                    "publisher": paper_data["venue"].publisher,
                    "metadata": paper_data["venue"].metadata,
                } if paper_data.get("venue") else {},
                "authors": [
                    {
                        "id": author.id,
                        "name": author.name,
                        "orcid": author.orcid,
                        "metadata": author.metadata,
                    } for author in paper_data["authors"]
                ],
                "citations": paper_data["citations"],
                "cited_by_count": paper_data["cited_by_count"]
            }
            json_data.append(json_paper)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(json_data)} papers to {filename}")

    def pull_open_alex_paper(self, count: int, filters: Dict = None, save_to_file: bool = True) -> List[Dict]:
        """Main method to ingest papers from OpenAlex.
        
        Args:
            count: Number of papers to fetch (500-1000), defaults to nums_papers_to_pull
            filters: Additional filters for the API request
            save_to_file: Whether to save results to JSON file

        Returns:
            List of paper data with authors and citations
        """

        print(f"Starting paper ingestion from OpenAlex (target: {count} papers)")

        # Fetch papers
        papers_data = self.openalex_client.fetch_papers(count, filters)

        # Save to file if requested
        if save_to_file and papers_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"openalex_papers_{timestamp}.json"
            self.save_papers_to_json(papers_data, filename)

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
            self.save_papers_to_json(enriched_papers, filename)

        return enriched_papers

    def load_papers_from_json(self, filename: str) -> List[Dict]:
        """
        Load papers data from JSON file and reconstruct full domain objects.
        """
        print(f"Loading papers data from {filename}...")

        try:
            with open(filename, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            papers_data = []

            for item in json_data:
                # ---------- Paper ----------
                p = item["paper"]
                paper = Paper(
                    id=p["id"],
                    title=p["title"],
                    abstract=p["abstract"],
                    publication_date=(
                        datetime.fromisoformat(p["publication_date"])
                        if p["publication_date"] else None
                    ),
                    doi=p["doi"],
                    pmid=p.get("pmid"),
                    source=p.get("source"),
                    metadata=p.get("metadata", {}),
                    ingested_at=datetime.fromisoformat(p["ingested_at"]),
                    last_updated=datetime.now(),
                )

                # ---------- Venue ----------
                v = item.get("venue")
                venue = None
                if v:
                    venue_type = (
                        VenueType[v["type"]] if v.get("type") else None
                    )

                    venue = Venue(
                        id=v["id"],
                        name=v["name"],
                        type=venue_type,
                        issn=v.get("issn"),
                        impact_factor=v.get("impact_factor"),
                        publisher=v.get("publisher"),
                        metadata=v.get("metadata", {}),
                    )

                # ---------- Authors ----------
                authors = []
                for a in item.get("authors", []):
                    authors.append(
                        Author(
                            id=a["id"],
                            name=a["name"],
                            orcid=a.get("orcid"),
                            metadata=a.get("metadata", {}),
                        )
                    )

                # ---------- Final reconstructed structure ----------
                reconstructed_data = {
                    "paper": paper,
                    "venue": venue,
                    "authors": authors,
                    "citations": item.get("citations", []),
                    "cited_by_count": item.get("cited_by_count", 0),
                }

                papers_data.append(reconstructed_data)

            print(f"Loaded {len(papers_data)} papers from JSON file")
            return papers_data

        except Exception as e:
            print(f"Error loading papers from {filename}: {e}")
            return []