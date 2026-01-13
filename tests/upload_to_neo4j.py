#!/usr/bin/env python3
"""
Script to upload OpenAlex papers data to Neo4j database.

This script loads papers data from the JSON file created by IngestionHandler
and uploads it to the Neo4j graph database.
"""

import asyncio
from models.pipelines.handlers.IngestionHandler import IngestionHandler

async def main():
    """Main function to upload papers to Neo4j with all relationships."""
    # Initialize IngestionHandler instance
    ingestion_handler = IngestionHandler()
    
    # Load papers from JSON file using IngestionHandler's method
    json_filename = "enriched_openalex_papers_20260105_231437.json"
    
    try:
        # Use IngestionHandler's load_papers_from_json method
        papers_data = ingestion_handler.load_papers_from_json(json_filename)
        
        if not papers_data:
            print("❌ No papers data loaded from file")
            return
        
        # Initialize Neo4j client
        client = await ingestion_handler._initialize_neo4j_client()
        
        print(f"Starting upload of {len(papers_data)} papers to Neo4j...")
        
        # Keep track of uploaded entities
        uploaded_papers = 0
        uploaded_authors = 0
        uploaded_venues = 0
        created_citations = 0
        created_publications = 0
        
        for i, paper_data in enumerate(papers_data):
            if i % 10 == 0:
                print(f"Processing paper {i + 1}/{len(papers_data)}...")
            
            try:
                paper = paper_data["paper"]
                authors = paper_data.get("authors", [])
                venue = paper_data.get("venue")
                citations = paper_data.get("citations", [])
                
                # Store paper with all its relationships
                await client.store_paper(
                    paper=paper,
                    authors=authors,
                    venue=venue,
                    citations=citations
                )
                
                uploaded_papers += 1
                uploaded_authors += len(authors)
                if venue:
                    uploaded_venues += 1
                    created_publications += 1
                created_citations += len(citations)
                
            except Exception as e:
                print(f"Error processing paper {paper_data['paper'].id}: {e}")
                continue
        
        print(f"\n=== Neo4j Upload Summary ===")
        print(f"Papers uploaded: {uploaded_papers}")
        print(f"Authors uploaded: {uploaded_authors}")
        print(f"Venues uploaded: {uploaded_venues}")
        print(f"Citation relationships created: {created_citations}")
        print(f"Publishing relationships created: {created_publications}")
        print("✅ Successfully uploaded papers with all relationships to Neo4j!")
        
    except FileNotFoundError:
        print(f"❌ JSON file '{json_filename}' not found.")
        print("Please run the OpenAlex ingestion first or update the filename.")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Clean up Neo4j connection
        if ingestion_handler.neo4j_client:
            await ingestion_handler.neo4j_client.close()
            ingestion_handler.neo4j_client = None

def main_sync():
    """Synchronous wrapper for the main function."""
    asyncio.run(main())

if __name__ == "__main__":
    main_sync()