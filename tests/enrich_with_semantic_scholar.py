#!/usr/bin/env python3
"""
Script to enrich OpenAlex papers with abstracts from Semantic Scholar.

This script loads papers data from the JSON file created by OpenAlex ingestion
and enriches it with abstracts and additional metadata from Semantic Scholar API.
"""

import os
from models.pipelines.handlers.IngestionHandler import IngestionHandler


def main():
    """Main function to enrich OpenAlex papers with Semantic Scholar data."""
    # Initialize IngestionHandler
    handler = IngestionHandler()
    
    # Look for the most recent OpenAlex papers file
    json_filename = "openalex_papers_20260106_122219.json"
    
    # Check if file exists
    if not os.path.exists(json_filename):
        print(f"❌ JSON file '{json_filename}' not found.")
        
        # List available JSON files
        json_files = [f for f in os.listdir('..') if f.startswith('openalex_papers_') and f.endswith('.json')]
        if json_files:
            print("\nAvailable OpenAlex JSON files:")
            for i, file in enumerate(json_files, 1):
                print(f"{i}. {file}")
            
            # Ask user to select a file
            try:
                choice = int(input(f"\nSelect a file (1-{len(json_files)}): ")) - 1
                if 0 <= choice < len(json_files):
                    json_filename = json_files[choice]
                else:
                    print("Invalid selection. Exiting.")
                    return
            except ValueError:
                print("Invalid input. Exiting.")
                return
        else:
            print("No OpenAlex JSON files found in current directory.")
            print("Please run the OpenAlex ingestion first.")
            return
    
    try:
        print(f"Using file: {json_filename}")
        
        # Load papers from JSON
        papers_data = handler.load_papers_from_json(json_filename)
        
        if not papers_data:
            print("❌ Failed to load papers data")
            return
        
        # Count papers without abstracts
        no_abstract_count = sum(1 for pd in papers_data 
                               if not pd["paper"].abstract or not pd["paper"].abstract.strip())
        
        print(f"\n=== Papers Analysis ===")
        print(f"Total papers loaded: {len(papers_data)}")
        print(f"Papers without abstracts: {no_abstract_count}")
        print(f"Papers with abstracts: {len(papers_data) - no_abstract_count}")
        
        if no_abstract_count == 0:
            print("✅ All papers already have abstracts!")
            return
        
        # Ask user confirmation
        print(f"\nThis will attempt to fetch abstracts for {no_abstract_count} papers from Semantic Scholar.")
        print("This process may take several minutes depending on API rate limits.")
        
        confirm = input("Continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Operation cancelled.")
            return
        
        # Enrich papers with Semantic Scholar data
        print("\n" + "="*60)
        enriched_papers = handler.enrich_papers_with_semantic_scholar(
            papers_data, 
            save_to_file=True
        )
        
        # Final summary
        newly_enriched = sum(1 for pd in enriched_papers 
                           if "semantic_scholar" in pd)
        
        print(f"\n=== Final Summary ===")
        print(f"Total papers processed: {len(enriched_papers)}")
        print(f"Papers enriched with Semantic Scholar data: {newly_enriched}")
        print(f"✅ Enrichment completed successfully!")
        
        # Ask if user wants to upload to Neo4j
        upload_choice = input("\nWould you like to upload the enriched data to Neo4j? (y/N): ").lower().strip()
        if upload_choice in ['y', 'yes']:
            print("\nUploading enriched papers to Neo4j...")
            success = handler.upload_papers_to_neo4j_sync(enriched_papers)
            if success:
                print("✅ Enriched papers successfully uploaded to Neo4j!")
            else:
                print("❌ Failed to upload enriched papers to Neo4j")
        
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
    except Exception as e:
        print(f"❌ Error during enrichment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()