#!/usr/bin/env python3
"""
Complete Zilliz Pipeline Test.

This script demonstrates the complete end-to-end pipeline:
1. Load papers from JSON files
2. Generate SciBERT embeddings
3. Upload to Zilliz Cloud
4. Verify upload with search queries

Run this script to test the entire knowledge ingestion pipeline.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.ingestions.handlers import IngestionHandler
from pipelines.ingestions.handlers.EmbeddingHandler import EmbeddingHandler
from pipelines.ingestions.handlers import MilvusClient


class ZillizPipelineTest:
    """Complete pipeline test for Zilliz knowledge ingestion."""
    
    def __init__(self):
        """Initialize the pipeline test."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = []
        
        # File paths for intermediate results
        self.input_papers_file = None
        self.embeddings_file = None
        
        print("Starting Zilliz Pipeline Test")
        
    def step_1_load_papers(self, paper_count: int = 3) -> bool:
        """Step 1: Load or fetch papers from available sources.
        
        Args:
            paper_count: Number of papers to process
            
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STEP 1: Loading Papers")
        print("="*60)
        
        try:
            # First try to find existing enriched papers
            import glob
            
            enriched_files = sorted(glob.glob("enriched_openalex_papers_*.json"), 
                                  key=lambda x: os.path.getmtime(x), reverse=True)
            
            if enriched_files:
                self.input_papers_file = enriched_files[0]
                print(f"Using existing enriched papers: {self.input_papers_file}")
                
                # Verify file content
                with open(self.input_papers_file, "r", encoding="utf-8") as f:
                    papers_data = json.load(f)
                
                print(f"✅ Loaded {len(papers_data)} enriched papers")
                return True
            
            # If no enriched papers, try regular openalex papers
            openalex_files = sorted(glob.glob("openalex_papers_*.json"), 
                                  key=lambda x: os.path.getmtime(x), reverse=True)
            
            if openalex_files:
                self.input_papers_file = openalex_files[0]
                print(f"Using existing OpenAlex papers: {self.input_papers_file}")
                
                # Verify file content
                with open(self.input_papers_file, "r", encoding="utf-8") as f:
                    papers_data = json.load(f)
                
                print(f"✅ Loaded {len(papers_data)} OpenAlex papers")
                return True
            
            # If no existing files, fetch new papers
            print(f"No existing papers found. Fetching {paper_count} new papers from OpenAlex...")
            
            ingestion_handler = IngestionHandler()
            papers = ingestion_handler.pull_OpenAlex_Paper(count=paper_count, save_to_file=True)
            
            if not papers:
                print("❌ No papers fetched from OpenAlex")
                return False
            
            # Find the generated file
            openalex_files = sorted(glob.glob("openalex_papers_*.json"), 
                                  key=lambda x: os.path.getmtime(x), reverse=True)
            if openalex_files:
                self.input_papers_file = openalex_files[0]
            
            # Try to enrich with Semantic Scholar
            print("Attempting to enrich with Semantic Scholar abstracts...")
            try:
                enriched_papers = ingestion_handler.enrich_papers_with_semantic_scholar(papers)
                if enriched_papers:
                    # Find the enriched file
                    enriched_files = sorted(glob.glob("enriched_openalex_papers_*.json"), 
                                          key=lambda x: os.path.getmtime(x), reverse=True)
                    if enriched_files:
                        self.input_papers_file = enriched_files[0]
                        print(f"✅ Successfully enriched and saved to: {self.input_papers_file}")
                    else:
                        print("✅ Enrichment completed, using original papers file")
                else:
                    print("Enrichment failed, using original papers file")
            except Exception as e:
                print(f"Enrichment failed: {e}, using original papers file")
            
            print(f"✅ Successfully prepared {len(papers)} papers")
            print(f"Input file: {self.input_papers_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Step 1 failed: {e}")
            return False
    
    def step_2_generate_embeddings(self) -> bool:
        """Step 2: Generate SciBERT embeddings.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STEP 2: Generating SciBERT Embeddings")
        print("="*60)
        
        try:
            if not self.input_papers_file or not os.path.exists(self.input_papers_file):
                print("❌ No input papers file found from Step 1")
                return False
            
            # Initialize SciBERT service
            embedding_service = EmbeddingHandler()
            
            print("Processing papers for embedding generation...")
            self.embeddings_file = embedding_service.process_papers(
                input_file=self.input_papers_file
            )
            
            if not self.embeddings_file or not os.path.exists(self.embeddings_file):
                print("❌ Embeddings file was not created")
                return False
            
            # Load and verify embeddings
            with open(self.embeddings_file, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)
            
            print(f"✅ Successfully generated embeddings")
            print(f"Embeddings file: {self.embeddings_file}")
            print(f"Total embeddings: {len(embeddings_data)}")
            if embeddings_data:
                print(f"Embedding dimension: {embeddings_data[0]['embedding_dim']}")
                print(f"Sample paper: {embeddings_data[0]['title'][:100]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Step 2 failed: {e}")
            return False
    
    def step_3_upload_to_zilliz(self) -> bool:
        """Step 3: Upload embeddings to Zilliz Cloud.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STEP 3: Uploading to Zilliz Cloud")
        print("="*60)
        
        try:
            if not self.embeddings_file or not os.path.exists(self.embeddings_file):
                print("❌ No embeddings file found from Step 2")
                return False
            
            # Initialize Zilliz service
            zilliz_service = MilvusClient()
            
            # Connect to Zilliz
            if not zilliz_service.connect():
                print("❌ Failed to connect to Zilliz Cloud")
                return False
            
            # Upload embeddings
            print("Uploading embeddings to Zilliz Cloud...")
            success = zilliz_service.upload_embeddings(
                embedding_file=self.embeddings_file,
                batch_size=100
            )
            
            if not success:
                print("❌ Upload to Zilliz failed")
                zilliz_service.disconnect()
                return False
            
            # Verify upload
            if zilliz_service.verify_upload():
                print("✅ Upload verified successfully")
            else:
                print("Upload completed but verification had issues")
            
            # Disconnect
            zilliz_service.disconnect()
            
            return True
            
        except Exception as e:
            print(f"❌ Step 3 failed: {e}")
            return False
    
    def step_4_test_search(self) -> bool:
        """Step 4: Test search functionality in Zilliz.
        
        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*60)
        print("STEP 4: Testing Search in Zilliz")
        print("="*60)
        
        try:
            from pymilvus import connections, Collection
            from models.configurators.VectorDBConfig import VectorDBConfig
            
            # Initialize connection
            config = VectorDBConfig.from_env()
            zilliz_service = MilvusClient()
            
            if not zilliz_service.connect():
                print("❌ Failed to connect to Zilliz for search test")
                return False
            
            # Get collection
            collection = Collection(config.collection_name)
            collection.load()
            
            # Test search with a sample query
            if not self.embeddings_file or not os.path.exists(self.embeddings_file):
                print("❌ No embeddings file available for search test")
                zilliz_service.disconnect()
                return False
            
            # Load a sample embedding for search
            with open(self.embeddings_file, "r", encoding="utf-8") as f:
                embeddings_data = json.load(f)
            
            if not embeddings_data:
                print("❌ No embeddings available for search test")
                zilliz_service.disconnect()
                return False
            
            # Use the first embedding as query
            query_embedding = embeddings_data[0]["embedding"]
            query_title = embeddings_data[0]["title"]
            
            print(f"Testing search with query: {query_title[:100]}...")
            
            search_params = {
                "metric_type": config.metric_type,
                "params": {"nprobe": 10}
            }
            
            # Perform search
            results = collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["paper_id", "title", "embedding_source"]
            )
            
            if results and len(results[0]) > 0:
                print("✅ Search test successful!")
                print("Top search results:")
                
                for i, hit in enumerate(results[0]):
                    print(f"   {i+1}. Score: {hit.score:.4f}")
                    print(f"      Paper ID: {hit.entity.get('paper_id')}")
                    print(f"      Title: {hit.entity.get('title', '')[:100]}...")
                    print(f"      Source: {hit.entity.get('embedding_source')}")
                    print()
                
                success = True
            else:
                print("Search returned no results")
                success = False
            
            zilliz_service.disconnect()
            return success
            
        except Exception as e:
            print(f"❌ Step 4 failed: {e}")
            return False
    
    def cleanup_files(self, keep_embeddings: bool = True):
        """Clean up intermediate files.
        
        Args:
            keep_embeddings: Whether to keep the final embeddings file
        """
        print("\nCleaning up intermediate files...")
        
        files_to_clean = []
        
        # Don't clean input papers - user might want to reuse them
        
        if not keep_embeddings and self.embeddings_file and os.path.exists(self.embeddings_file):
            files_to_clean.append(self.embeddings_file)
        
        for file_path in files_to_clean:
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Could not remove {file_path}: {e}")
    
    def run_complete_pipeline(self, paper_count: int = 3, cleanup: bool = False) -> bool:
        """Run the complete pipeline.
        
        Args:
            paper_count: Number of papers to process (if fetching new papers)
            cleanup: Whether to clean up intermediate files
            
        Returns:
            True if all steps successful, False otherwise
        """
        start_time = time.time()
        
        print("Starting Complete Zilliz Pipeline Test")
        print(f"Processing up to {paper_count} papers")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        steps = [
            ("Load Papers", lambda: self.step_1_load_papers(paper_count)),
            ("Generate SciBERT Embeddings", self.step_2_generate_embeddings),
            ("Upload to Zilliz Cloud", self.step_3_upload_to_zilliz),
            ("Test Search Functionality", self.step_4_test_search),
        ]
        
        success_count = 0
        
        for i, (step_name, step_func) in enumerate(steps, 1):
            print(f"\nExecuting Step {i}/{len(steps)}: {step_name}")
            
            try:
                if step_func():
                    success_count += 1
                    print(f"✅ Step {i} completed successfully")
                else:
                    print(f"❌ Step {i} failed")
                    break
            except Exception as e:
                print(f"❌ Step {i} failed with exception: {e}")
                break
        
        # Final summary
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Total execution time: {elapsed_time:.2f} seconds")
        print(f"✅ Successful steps: {success_count}/{len(steps)}")
        print(f"Success rate: {success_count/len(steps)*100:.1f}%")
        
        if success_count == len(steps):
            print("✅ All steps completed successfully!")
            print(f"Final embeddings file: {self.embeddings_file}")
            print("Your Zilliz knowledge ingestion pipeline is working correctly.")
        else:
            print("❌ Pipeline completed with failures")
            print("Check the error messages above for debugging information.")
        
        # Cleanup if requested
        if cleanup:
            self.cleanup_files(keep_embeddings=True)
        
        return success_count == len(steps)
    
    def run_individual_step(self, step_number: int, paper_count: int = 3) -> bool:
        """Run a specific step only.
        
        Args:
            step_number: Step number to run (1-4)
            paper_count: Number of papers for step 1
            
        Returns:
            True if step successful, False otherwise
        """
        step_functions = {
            1: lambda: self.step_1_load_papers(paper_count),
            2: self.step_2_generate_embeddings,
            3: self.step_3_upload_to_zilliz,
            4: self.step_4_test_search,
        }
        
        if step_number not in step_functions:
            print(f"❌ Invalid step number: {step_number}")
            return False
        
        # For steps 2-4, we need to find existing files
        if step_number > 1:
            self._find_existing_files()
        
        return step_functions[step_number]()
    
    def _find_existing_files(self):
        """Find existing files from previous steps."""
        import glob
        
        if not self.input_papers_file:
            # Try to find papers file
            enriched_files = sorted(glob.glob("enriched_openalex_papers_*.json"), 
                                  key=lambda x: os.path.getmtime(x), reverse=True)
            if enriched_files:
                self.input_papers_file = enriched_files[0]
            else:
                openalex_files = sorted(glob.glob("openalex_papers_*.json"), 
                                      key=lambda x: os.path.getmtime(x), reverse=True)
                if openalex_files:
                    self.input_papers_file = openalex_files[0]
        
        if not self.embeddings_file:
            # Try to find embeddings file
            embedding_files = sorted(glob.glob("paper_embeddings_scibert_*.json"), 
                                   key=lambda x: os.path.getmtime(x), reverse=True)
            if embedding_files:
                self.embeddings_file = embedding_files[0]


def main():
    """Main function to run the pipeline test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Zilliz Pipeline Test")
    parser.add_argument("--papers", type=int, default=3, help="Number of papers to process")
    parser.add_argument("--cleanup", action="store_true", help="Clean up intermediate files")
    parser.add_argument("--step", type=int, choices=[1,2,3,4], help="Run specific step only")
    
    args = parser.parse_args()
    
    # Initialize pipeline test
    pipeline = ZillizPipelineTest()
    
    try:
        if args.step:
            # Run specific step
            success = pipeline.run_individual_step(args.step, args.papers)
            print(f"Step {args.step} {'✅ PASSED' if success else '❌ FAILED'}")
        else:
            # Run complete pipeline
            success = pipeline.run_complete_pipeline(
                paper_count=args.papers,
                cleanup=args.cleanup
            )
            
            if success:
                print("\n✅ PIPELINE TEST PASSED!")
            else:
                print("\n❌ PIPELINE TEST FAILED!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()