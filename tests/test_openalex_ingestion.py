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
    
    # Test with a smaller number first (50 papers for quick testing)
    print("Testing with 1 papers first...\n")
    test_papers = openalex.pull_OpenAlex_Paper(count=5, save_to_file=True)
    
    if test_papers:
        print(f"✅ Successfully fetched {len(test_papers)} papers")
        
        # Show sample data
        if len(test_papers) > 0:
            sample = test_papers[0]
            print("\n=== Sample Paper Data ===")
            print(f"Title: {sample['paper'].title}")
            print(f"Year: {sample['paper'].publication_date.year if sample['paper'].publication_date else 'Unknown'}")
            print(f"DOI: {sample['paper'].doi}")
            print(f"Authors: {[author.name for author in sample['authors']]}")
            print(f"Citations: {len(sample['citations'])} referenced works")
        
        # # Now fetch the full 1000 papers
        # print("\n" + "="*50)
        # print("Now fetching 1000 papers (this may take a few minutes)...")
        # full_papers = openalex.pull_OpenAlex_Paper(count=1000, save_to_file=True)
        #
        # if full_papers:
        #     print(f"✅ Successfully completed full ingestion of {len(full_papers)} papers")
        # else:
        #     print("❌ Full ingestion failed")
    else:
        print("❌ Test failed")

if __name__ == "__main__":
    main()