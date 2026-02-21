"""
Test the enhanced paper ingestion pipeline with PDF processing and visual content extraction.
"""

import asyncio
from pipelines.ingestions.IngestionHandler import IngestionHandler

async def test_enhanced_ingestion():
    """Test the enhanced ingestion pipeline."""
    
    # Initialize the ingestion handler
    ingestion_handler = IngestionHandler()
    
    # Test with a small batch of papers, including PDF processing
    print("Testing enhanced paper ingestion with PDF processing...")
    
    # Fetch papers with PDF processing enabled
    papers_data = ingestion_handler.pull_open_alex_paper(
        count=5,  # Small batch for testing
        filters={"has_pdf": True},  # Only papers with PDFs
        save_to_file=True,
        process_pdfs=True  # Enable PDF processing
    )
    
    # Show results
    if papers_data:
        print(f"\n=== Enhanced Ingestion Results ===")
        for i, paper_data in enumerate(papers_data[:2]):  # Show first 2 papers
            paper = paper_data["paper"]
            institutions = paper_data.get("institutions", [])
            figures = paper_data.get("figures", [])
            tables = paper_data.get("tables", [])
            
            print(f"\nPaper {i+1}: {paper.title}")
            print(f"  PDF URL: {paper.pdf_url}")
            print(f"  Authors: {len(paper_data['authors'])}")
            print(f"  Institutions: {len(institutions)}")
            print(f"  Figures extracted: {len(figures)}")
            print(f"  Tables extracted: {len(tables)}")
            
            if institutions:
                print("  Institution names:")
                for inst in institutions[:3]:  # Show first 3
                    print(f"    - {inst.name}")
            
            if figures:
                print("  Figure descriptions:")
                for fig in figures[:2]:  # Show first 2
                    print(f"    - Figure {fig.figure_number}: {fig.description or 'No description'}")
            
            if tables:
                print("  Table descriptions:")
                for tbl in tables[:2]:  # Show first 2
                    print(f"    - Table {tbl.table_number}: {tbl.description or 'No description'}")
    
    print("\n✅ Enhanced ingestion test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_ingestion())