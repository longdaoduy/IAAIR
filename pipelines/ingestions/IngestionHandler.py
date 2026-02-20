from typing import Dict, List
from datetime import datetime
import json
from models.schemas.nodes import Paper, Author, Venue, VenueType, Institution, Figure, Table
from clients.metadata.SemanticScholarClient import SemanticScholarClient
from clients.metadata.OpenAlexClient import OpenAlexClient
from pipelines.ingestions.PDFProcessingHandler import PDFProcessingHandler
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.DeepseekClient import DeepseekClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient


class IngestionHandler():
    def __init__(self):
        self.semantic_scholar_client = SemanticScholarClient()
        self.openalex_client = OpenAlexClient()
        self.clip_client = CLIPClient()
        self.scibert_client = SciBERTClient()
        self.milvus_client = MilvusClient()
        self.pdf_handler = PDFProcessingHandler(
            self.clip_client, 
            self.scibert_client, 
            self.milvus_client
        )

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
                    "pdf_url": getattr(paper_data["paper"], 'pdf_url', None),
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
                "institutions": [
                    {
                        "id": inst.id,
                        "name": inst.name,
                        "country": inst.country,
                        "city": inst.city,
                        "type": inst.type,
                        "website": inst.website,
                        "metadata": inst.metadata,
                    } for inst in paper_data.get("institutions", [])
                ],
                "figures": [
                    {
                        "id": fig.id,
                        "paper_id": fig.paper_id,
                        "figure_number": fig.figure_number,
                        "description": fig.description,
                        "caption": fig.caption,
                        "page_number": fig.page_number,
                        "image_path": fig.image_path,
                        "image_embedding": fig.image_embedding,
                        "description_embedding": fig.description_embedding,
                    } for fig in paper_data.get("figures", [])
                ],
                "tables": [
                    {
                        "id": tbl.id,
                        "paper_id": tbl.paper_id,
                        "table_number": tbl.table_number,
                        "description": tbl.description,
                        "caption": tbl.caption,
                        "page_number": tbl.page_number,
                        "headers": tbl.headers,
                        "rows": tbl.rows,
                        "image_path": tbl.image_path,
                        "image_embedding": tbl.image_embedding,
                        "description_embedding": tbl.description_embedding,
                    } for tbl in paper_data.get("tables", [])
                ],
                "citations": paper_data["citations"],
                "cited_by_count": paper_data["cited_by_count"]
            }
            json_data.append(json_paper)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(json_data)} papers to {filename}")

    def pull_open_alex_paper(self, count: int, filters: Dict = None, save_to_file: bool = True,
                             process_pdfs: bool = False) -> List[Dict]:
        """Main method to ingest papers from OpenAlex.
        
        Args:
            count: Number of papers to fetch (500-1000), defaults to nums_papers_to_pull
            filters: Additional filters for the API request
            save_to_file: Whether to save results to JSON file
            process_pdfs: Whether to process PDFs and extract figures/tables

        Returns:
            List of paper data with authors, citations, and optionally figures/tables
        """

        print(f"Starting paper ingestion from OpenAlex (target: {count} papers)")

        # Fetch papers with PDF URLs
        papers_data = self.openalex_client.fetch_papers(count, filters)

        # Process PDFs if requested
        if process_pdfs and papers_data:
            print(f"Processing PDFs for {len(papers_data)} papers...")
            papers_data = self._process_papers_with_pdfs(papers_data)

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
            total_institutions = sum(len(pd.get('institutions', [])) for pd in papers_data)
            total_figures = sum(len(pd.get('figures', [])) for pd in papers_data)
            total_tables = sum(len(pd.get('tables', [])) for pd in papers_data)
            print(f"Authors extracted: {total_authors}")
            print(f"Institutions extracted: {total_institutions}")
            print(f"Citations extracted: {total_citations}")
            print(f"Figures extracted: {total_figures}")
            print(f"Tables extracted: {total_tables}")
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

    def process_pdfs_and_extract_content(self, papers_data: List[Dict], save_to_file: bool = True) -> List[Dict]:
        """
        Process PDFs for papers and extract figures and tables with embeddings.
        
        Args:
            papers_data: List of paper data dictionaries
            save_to_file: Whether to save the results to a file
            
        Returns:
            List of enriched paper data with figures and tables
        """
        print(f"Starting PDF processing and content extraction for {len(papers_data)} papers...")

        processed_papers = []
        total_figures = 0
        total_tables = 0

        for i, paper_data in enumerate(papers_data):
            print(f"Processing paper {i + 1}/{len(papers_data)}: {paper_data['paper'].id}")

            try:
                paper = paper_data["paper"]

                # Initialize empty lists for new content (keep existing institutions from OpenAlex)
                figures = []
                tables = []
                institutions = paper_data.get("institutions", [])  # Use institutions from OpenAlex

                # Process PDF if URL is available
                if hasattr(paper, 'pdf_url') and paper.pdf_url:
                    try:
                        # Extract figures and tables from PDF
                        pdf_figures, pdf_tables = self.pdf_handler.process_paper_pdf(paper)
                        figures.extend(pdf_figures)
                        tables.extend(pdf_tables)

                    except Exception as e:
                        print(f"Error processing PDF for paper {paper.id}: {e}")

                # Create enhanced paper data
                enhanced_paper_data = {
                    **paper_data,
                    "institutions": institutions,  # Keep existing institutions from OpenAlex
                    "figures": figures,
                    "tables": tables
                }

                processed_papers.append(enhanced_paper_data)
                total_figures += len(figures)
                total_tables += len(tables)

            except Exception as e:
                print(f"Error processing paper {paper_data['paper'].id}: {e}")
                # Still add the paper without extracted content
                processed_papers.append({
                    **paper_data,
                    "institutions": paper_data.get("institutions", []),  # Keep existing institutions
                    "figures": [],
                    "tables": []
                })

        # Save to file if requested
        if save_to_file and processed_papers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"processed_papers_with_content_{timestamp}.json"
            self.save_papers_to_json(processed_papers, filename)

        # Print summary
        print(f"\n=== PDF Processing Summary ===")
        print(f"Papers processed: {len(processed_papers)}")
        print(f"Total figures extracted: {total_figures}")
        print(f"Total tables extracted: {total_tables}")
        print(f"Average figures per paper: {total_figures / len(processed_papers):.1f}")
        print(f"Average tables per paper: {total_tables / len(processed_papers):.1f}")

        return processed_papers

    def _process_papers_with_pdfs(self, papers_data: List[Dict]) -> List[Dict]:
        """Helper method to process papers with PDF content extraction."""
        processed_papers = []
        total_figures = 0
        total_tables = 0

        for i, paper_data in enumerate(papers_data):
            print(f"Processing paper {i + 1}/{len(papers_data)}: {paper_data['paper'].id}")

            try:
                paper = paper_data["paper"]

                # Initialize empty lists for new content (keep existing institutions from OpenAlex)
                figures = []
                tables = []
                institutions = paper_data.get("institutions", [])  # Use institutions from OpenAlex

                # Process PDF if URL is available
                if hasattr(paper, 'pdf_url') and paper.pdf_url:
                    try:
                        # Extract figures and tables from PDF
                        pdf_figures, pdf_tables = self.pdf_handler.process_paper_pdf(paper)
                        figures.extend(pdf_figures)
                        tables.extend(pdf_tables)

                    except Exception as e:
                        print(f"Error processing PDF for paper {paper.id}: {e}")

                # Create enhanced paper data
                enhanced_paper_data = {
                    **paper_data,
                    "institutions": institutions,  # Keep existing institutions from OpenAlex
                    "figures": figures,
                    "tables": tables
                }

                processed_papers.append(enhanced_paper_data)
                total_figures += len(figures)
                total_tables += len(tables)

            except Exception as e:
                print(f"Error processing paper {paper_data['paper'].id}: {e}")
                # Still add the paper without extracted content
                processed_papers.append({
                    **paper_data,
                    "institutions": paper_data.get("institutions", []),  # Keep existing institutions
                    "figures": [],
                    "tables": []
                })

        print(f"\n=== PDF Processing Results ===")
        print(f"Total figures extracted: {total_figures}")
        print(f"Total tables extracted: {total_tables}")
        print(f"Average figures per paper: {total_figures / len(processed_papers):.1f}")
        print(f"Average tables per paper: {total_tables / len(processed_papers):.1f}")

        return processed_papers

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
                    pdf_url=p.get("pdf_url"),
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

                # ---------- Institutions ----------
                institutions = []
                for i in item.get("institutions", []):
                    institutions.append(
                        Institution(
                            id=i["id"],
                            name=i["name"],
                            country=i.get("country"),
                            city=i.get("city"),
                            type=i.get("type"),
                            website=i.get("website"),
                            metadata=i.get("metadata", {}),
                        )
                    )

                # ---------- Figures ----------
                figures = []
                for f in item.get("figures", []):
                    figures.append(
                        Figure(
                            id=f["id"],
                            paper_id=f["paper_id"],
                            figure_number=f["figure_number"],
                            description=f.get("description"),
                            caption=f.get("caption"),
                            page_number=f.get("page_number"),
                            image_path=f.get("image_path"),
                            image_embedding=f.get("image_embedding"),
                            description_embedding=f.get("description_embedding"),
                        )
                    )

                # ---------- Tables ----------
                tables = []
                for t in item.get("tables", []):
                    tables.append(
                        Table(
                            id=t["id"],
                            paper_id=t["paper_id"],
                            table_number=t["table_number"],
                            description=t.get("description"),
                            caption=t.get("caption"),
                            page_number=t.get("page_number"),
                            headers=t.get("headers"),
                            rows=t.get("rows"),
                            image_path=t.get("image_path"),
                            image_embedding=t.get("image_embedding"),
                            description_embedding=t.get("description_embedding"),
                        )
                    )

                # ---------- Final reconstructed structure ----------
                reconstructed_data = {
                    "paper": paper,
                    "venue": venue,
                    "authors": authors,
                    "institutions": institutions,
                    "figures": figures,
                    "tables": tables,
                    "citations": item.get("citations", []),
                    "cited_by_count": item.get("cited_by_count", 0),
                }

                papers_data.append(reconstructed_data)

            print(f"Loaded {len(papers_data)} papers from JSON file")
            return papers_data

        except Exception as e:
            print(f"Error loading papers from {filename}: {e}")
            return []
