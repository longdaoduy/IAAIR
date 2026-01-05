#!/usr/bin/env python3
"""
Test script for OpenAlex paper ingestion functionality.

This script demonstrates how to use the IngestionHandler class to fetch
500-1000 papers from OpenAlex with title, year, authors, DOI, and citations.
"""

from models.pipelines.ingestions.IngestionHandler import IngestionHandler

def main():
    """Test the OpenAlex paper ingestion functionality."""
    print("=== OpenAlex Paper Ingestion Test ===\n")
    
    # Initialize the OpenAlex ingestion client
    openalex = IngestionHandler()
    
    # Test with a smaller number first (1 paper for quick testing)
    print("Testing with 1 paper first...\n")
    test_papers = openalex.pull_OpenAlex_Paper(count=5, save_to_file=True)
    
    if test_papers:
        print(f"✅ Successfully fetched {len(test_papers)} papers")
        
        # Show sample data
        if len(test_papers) > 0:
            sample = test_papers[0]
            print("\n=== Sample Paper Data ===")
            print(f"Title: {sample['paper'].title}")
            print(f"Source: {sample['paper'].source}")
            print(f"Year: {sample['paper'].publication_date.year if sample['paper'].publication_date else 'Unknown'}")
            print(f"DOI: {sample['paper'].doi}")
            print(f"Abstract: {sample['paper'].abstract[:200] if sample['paper'].abstract else 'No abstract'}...")
            print(f"Authors: {[author.name for author in sample['authors']]}")
            print(f"Citations: {len(sample['citations'])} referenced works")

        # Test Semantic Scholar enrichment
        print("\n" + "="*50)
        print("Testing Semantic Scholar enrichment...")
        
        enriched_papers = openalex.enrich_papers_with_semantic_scholar(test_papers)
        
        if enriched_papers:
            sample_enriched = enriched_papers[0]
            print(f"\n=== Enriched Paper Data ===")
            print(f"Title: {sample_enriched['paper'].title}")
            print(f"Abstract (after enrichment): {sample_enriched['paper'].abstract[:200] if sample_enriched['paper'].abstract else 'Still no abstract'}...")
            
            if "semantic_scholar" in sample_enriched:
                print(f"Semantic Scholar ID: {sample_enriched['semantic_scholar'].get('paperId', 'N/A')}")
                print(f"Semantic Scholar Citation Count: {sample_enriched['semantic_scholar'].get('citationCount', 'N/A')}")
        
        # # Now fetch the full 1000 papers and enrich them
        # print("\n" + "="*50)
        # print("Now fetching 100 papers and enriching with Semantic Scholar (this may take several minutes)...")
        # full_papers = openalex.pull_OpenAlex_Paper(count=100, save_to_file=True)
        #
        # if full_papers:
        #     print(f"✅ Successfully fetched {len(full_papers)} papers")
        #     
        #     # Enrich with Semantic Scholar
        #     enriched_full_papers = openalex.enrich_papers_with_semantic_scholar(full_papers)
        #     print(f"✅ Successfully completed enrichment of {len(enriched_full_papers)} papers")
        # else:
        #     print("❌ Full ingestion failed")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()