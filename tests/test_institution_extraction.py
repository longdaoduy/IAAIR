"""
Test the enhanced institution extraction from OpenAlex data.
"""

import json
from clients.metadata.OpenAlexClient import OpenAlexClient
from pipelines.ingestions.IngestionHandler import IngestionHandler

def test_institution_extraction():
    """Test institution extraction from OpenAlex data."""
    
    print("Testing institution extraction from OpenAlex...")
    
    # Initialize the OpenAlex client
    openalex_client = OpenAlexClient()
    
    # Test with a small batch of papers
    print("Fetching papers from OpenAlex...")
    papers_data = openalex_client.fetch_papers(count=5)
    
    if papers_data:
        print(f"\n=== Institution Extraction Results ===")
        for i, paper_data in enumerate(papers_data[:3]):  # Show first 3 papers
            paper = paper_data["paper"]
            authors = paper_data.get("authors", [])
            institutions = paper_data.get("institutions", [])
            
            print(f"\nPaper {i+1}: {paper.title}")
            print(f"  Authors: {len(authors)}")
            
            # Show authors with null IDs
            null_id_authors = [a for a in authors if not a.id]
            if null_id_authors:
                print("  Authors with null IDs:")
                for author in null_id_authors:
                    print(f"    - {author.name}")
            
            print(f"  Institutions extracted: {len(institutions)}")
            if institutions:
                print("  Institution details:")
                for inst in institutions:
                    print(f"    - {inst.name} ({inst.country or 'Unknown country'})")
                    print(f"      ID: {inst.id}")
                    print(f"      Type: {inst.type}")
                    if inst.metadata.get('associated_author'):
                        print(f"      Associated with author: {inst.metadata['associated_author']}")
                    print()
    
    print("\n✅ Institution extraction test completed!")

def test_example_data():
    """Test with the provided example data structure."""
    print("\n=== Testing with Example Data ===")
    
    # Simulate the OpenAlex response structure from the user's example
    example_work = {
        "id": "https://openalex.org/W3038568908",
        "authorships": [
            {
                "author": {
                    "id": "https://openalex.org/A5039600762",
                    "display_name": "M. Shoji"
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I199525922",
                        "display_name": "National Institutes of Natural Sciences",
                        "country_code": "JP",
                        "type": "facility"
                    }
                ]
            },
            {
                "author": {
                    "id": None,  # This author has null ID
                    "display_name": "LHD Experiment Group"
                },
                "institutions": [
                    {
                        "id": "https://openalex.org/I199525922",
                        "display_name": "National Institutes of Natural Sciences",
                        "country_code": "JP",
                        "type": "facility"
                    },
                    {
                        "id": "https://openalex.org/I4210108322",
                        "display_name": "National Institute for Fusion Science",
                        "country_code": "JP",
                        "type": "facility"
                    }
                ]
            }
        ]
    }
    
    # Test the extract_institutions method
    institutions = OpenAlexClient.extract_institutions(example_work)
    
    print(f"Institutions extracted for null ID authors: {len(institutions)}")
    for inst in institutions:
        print(f"  - {inst.name}")
        print(f"    ID: {inst.id}")
        print(f"    Country: {inst.country}")
        print(f"    Type: {inst.type}")
        print(f"    Associated author: {inst.metadata.get('associated_author')}")
        print()

if __name__ == "__main__":
    # Test with example data first
    test_example_data()
    
    # Then test with real API data
    # test_institution_extraction()  # Uncomment to test with real API