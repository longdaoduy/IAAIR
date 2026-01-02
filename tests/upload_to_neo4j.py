#!/usr/bin/env python3
"""
Script to upload OpenAlex papers data to Neo4j database.

This script loads papers data from the JSON file created by OpenAplex
and uploads it to the Neo4j graph database.
"""

import json
import asyncio
from datetime import datetime
from models.pipelines.ingestions.OpenAplex import OpenAplex
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Author import Author

def load_papers_from_json(filename: str):
    """Load papers data from JSON file and convert back to proper format."""
    print(f"Loading papers data from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    papers_data = []
    for item in json_data:
        # Reconstruct Paper object
        paper_data = item["paper"]
        paper = Paper(
            id=paper_data["id"],
            title=paper_data["title"],
            abstract=paper_data["abstract"],
            publication_date=datetime.fromisoformat(paper_data["publication_date"]) if paper_data["publication_date"] else None,
            doi=paper_data["doi"],
            source=paper_data["source"],
            ingested_at=datetime.fromisoformat(paper_data["ingested_at"]),
            last_updated=datetime.now()
        )
        
        # Reconstruct Author objects
        authors = []
        for author_data in item["authors"]:
            author = Author(
                id=author_data["id"],
                name=author_data["name"],
                orcid=author_data["orcid"]
            )
            authors.append(author)
        
        # Reconstruct full paper data structure
        reconstructed_data = {
            "paper": paper,
            "authors": authors,
            "citations": item["citations"],
            "cited_by_count": item["cited_by_count"]
        }
        
        papers_data.append(reconstructed_data)
    
    print(f"Loaded {len(papers_data)} papers from JSON file")
    return papers_data

async def main():
    """Main function to upload papers to Neo4j."""
    # Initialize OpenAplex instance
    openalex = OpenAplex()
    
    # Load papers from JSON file (assuming it exists)
    json_filename = "/home/dnhoa/IAAIR/IAAIR/tests/openalex_papers_20260101_093658.json"
    
    try:
        papers_data = load_papers_from_json(json_filename)
        
        # Upload to Neo4j
        success = await openalex.upload_papers_to_neo4j(papers_data)
        
        if success:
            print("✅ Papers successfully uploaded to Neo4j!")
        else:
            print("❌ Failed to upload papers to Neo4j")
            
    except FileNotFoundError:
        print(f"❌ JSON file '{json_filename}' not found.")
        print("Please run the OpenAlex ingestion first or update the filename.")
    except Exception as e:
        print(f"❌ Error: {e}")

def main_sync():
    """Synchronous wrapper for the main function."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()