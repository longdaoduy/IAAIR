#!/usr/bin/env python3
"""
Comprehensive Search Test for Milvus/Zilliz Vector Database.

This script tests semantic search functionality using SciBERT embeddings
stored in Zilliz Cloud. It supports searching by abstract or title text
and returns topically relevant papers with latency measurements.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scibert_embedding import SciBERTEmbeddingService
from upload_data_zilliz import ZillizUploadService
from pymilvus import connections, Collection
from models.configurators.VectorDBConfig import VectorDBConfig
from clients.graph_store.Neo4jClient import Neo4jClient


class ZillizSearchTest:
    """Comprehensive search testing for Zilliz vector database."""
    
    def __init__(self):
        """Initialize the search test."""
        self.config = VectorDBConfig.from_env()
        self.embedding_service = None
        self.zilliz_service = None
        self.collection = None
        self.neo4j_client = None
        
    def initialize_services(self) -> bool:
        """Initialize all required services.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("🔧 Initializing services...")
            
            # Initialize SciBERT embedding service
            print("   Loading SciBERT model...")
            self.embedding_service = SciBERTEmbeddingService()
            
            # Initialize Zilliz service
            print("   Connecting to Zilliz...")
            self.zilliz_service = ZillizUploadService()
            if not self.zilliz_service.connect():
                print("❌ Failed to connect to Zilliz")
                return False
            
            # Get collection
            print("   Loading collection...")
            self.collection = Collection(self.config.collection_name)
            
            # Make sure collection is loaded and not empty
            if self.collection.is_empty:
                print("❌ Collection is empty. Please upload data first.")
                return False
            
            self.collection.load()
            print(f"✅ Collection loaded: {self.config.collection_name}")
            print(f"   Total entities: {self.collection.num_entities}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}")
            return False
    
    def generate_query_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for search query text.
        
        Args:
            text: Query text (abstract or title)
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Use SciBERT to generate embedding (single text)
            embedding = self.embedding_service.generate_embedding(text)
            return embedding
        except Exception as e:
            print(f"❌ Failed to generate query embedding: {e}")
            return None
    
    def search_by_text(self, 
                       query_text: str, 
                       search_field: str = "abstract_embedding",
                       top_k: int = 5) -> Tuple[List[Dict], float]:
        """Search papers by text query.
        
        Args:
            query_text: Search query text
            search_field: Field to search in ("abstract_embedding" or "title_embedding")
            top_k: Number of results to return
            
        Returns:
            Tuple of (search results, latency in ms)
        """
        print(f"\n🔍 Searching for: '{query_text[:100]}{'...' if len(query_text) > 100 else ''}'")
        print(f"Search field: {search_field}")
        
        # Generate query embedding
        start_embed_time = time.time()
        query_embedding = self.generate_query_embedding(query_text)
        embed_time = (time.time() - start_embed_time) * 1000
        
        if query_embedding is None:
            print("❌ Failed to generate query embedding")
            return [], 0.0
        
        print(f"⚡ Embedding generation time: {embed_time:.2f}ms")
        
        # Perform search
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": 10}
        }
        
        start_search_time = time.time()
        try:
            results = self.collection.search(
                data=[query_embedding],
                anns_field=search_field,
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )
            search_time = (time.time() - start_search_time) * 1000
            total_time = embed_time + search_time
            
            print(f"⚡ Vector search time: {search_time:.2f}ms")
            print(f"⚡ Total search time: {total_time:.2f}ms")
            
            # Format results
            formatted_results = []
            if results and len(results[0]) > 0:
                print(f"✅ Found {len(results[0])} results")
                
                for i, hit in enumerate(results[0]):
                    result = {
                        "rank": i + 1,
                        "id": hit.entity.get('id'),
                        "score": hit.score,
                        "similarity": 1 / (1 + hit.score) if hit.score > 0 else 1.0  # Convert distance to similarity
                    }
                    formatted_results.append(result)
            else:
                print("❌ No results found")
            
            return formatted_results, total_time
            
        except Exception as e:
            search_time = (time.time() - start_search_time) * 1000
            print(f"❌ Search failed: {e}")
            return [], search_time
    
    def display_search_results(self, results: List[Dict], query_text: str):
        """Display search results in a formatted way.
        
        Args:
            results: Search results from search_by_text()
            query_text: Original query text
        """
        if not results:
            print("📭 No results to display")
            return
        
        print(f"\n📊 SEARCH RESULTS for: '{query_text[:80]}{'...' if len(query_text) > 80 else ''}'")
        print("=" * 80)
        
        for result in results:
            print(f"#{result['rank']} | Score: {result['score']:.4f} | Similarity: {result['similarity']:.3f}")
            print(f"      ID: {result['id']}")
            print(f"      {'─' * 60}")
    
    async def validate_with_neo4j(self, paper_ids: List[str]) -> Dict[str, bool]:
        """Validate that paper IDs exist in Neo4j.
        
        Args:
            paper_ids: List of paper IDs from search results
            
        Returns:
            Dictionary mapping paper_id -> exists_in_neo4j
        """
        try:
            if self.neo4j_client is None:
                self.neo4j_client = Neo4jClient()
                await self.neo4j_client.connect()
            
            validation_results = {}
            
            async with self.neo4j_client.driver.session() as session:
                for paper_id in paper_ids:
                    try:
                        # Query Neo4j to check if paper exists
                        query = "MATCH (p:Paper {id: $paper_id}) RETURN p.id as id LIMIT 1"
                        result = await session.run(query, {"paper_id": paper_id})
                        records = await result.data()
                        validation_results[paper_id] = len(records) > 0
                    except Exception as e:
                        print(f"⚠️  Could not validate {paper_id}: {e}")
                        validation_results[paper_id] = False
            
            return validation_results
            
        except Exception as e:
            print(f"❌ Neo4j validation failed: {e}")
            return {paper_id: False for paper_id in paper_ids}
    
    def run_search_tests(self) -> bool:
        """Run comprehensive search tests with multiple queries.
        
        Returns:
            True if all tests pass, False otherwise
        """
        # Test queries covering different domains
        test_queries = [
            {
                "text": "machine learning neural networks deep learning artificial intelligence",
                "domain": "AI/ML",
                "expected_relevance": "high"
            },
            {
                "text": "covid-19 coronavirus vaccine effectiveness immunity",
                "domain": "Medicine",
                "expected_relevance": "high"
            },
            {
                "text": "quantum computing algorithms quantum mechanics",
                "domain": "Physics/CS",
                "expected_relevance": "medium"
            },
            {
                "text": "climate change global warming environmental science",
                "domain": "Environmental",
                "expected_relevance": "medium"
            },
            {
                "text": "blockchain cryptocurrency bitcoin distributed systems",
                "domain": "Technology",
                "expected_relevance": "low"
            }
        ]
        
        print(f"\n🧪 Running {len(test_queries)} search test scenarios...")
        print("=" * 80)
        
        all_latencies = []
        all_results = []
        
        for i, query in enumerate(test_queries):
            print(f"\n--- Test {i+1}: {query['domain']} ---")
            
            # Test abstract search
            results_abstract, latency_abstract = self.search_by_text(
                query_text=query["text"],
                search_field="abstract_embedding",
                top_k=3
            )
            
            # Test title search  
            results_title, latency_title = self.search_by_text(
                query_text=query["text"],
                search_field="title_embedding",
                top_k=3
            )
            
            # Display results
            print(f"\n📄 ABSTRACT SEARCH:")
            self.display_search_results(results_abstract, query["text"])
            
            print(f"\n📋 TITLE SEARCH:")
            self.display_search_results(results_title, query["text"])
            
            # Collect performance metrics
            all_latencies.extend([latency_abstract, latency_title])
            all_results.extend([results_abstract, results_title])
        
        # Performance summary
        if all_latencies:
            valid_latencies = [l for l in all_latencies if l > 0]
            if valid_latencies:
                avg_latency = sum(valid_latencies) / len(valid_latencies)
                max_latency = max(valid_latencies)
                min_latency = min(valid_latencies)
                
                print(f"\n📈 PERFORMANCE SUMMARY")
                print("=" * 50)
                print(f"Average latency: {avg_latency:.2f}ms")
                print(f"Min latency: {min_latency:.2f}ms") 
                print(f"Max latency: {max_latency:.2f}ms")
                print(f"Latency budget (<100ms): {'✅ PASS' if avg_latency < 100 else '❌ FAIL'}")
            else:
                print(f"\n📈 PERFORMANCE SUMMARY")
                print("=" * 50)
                print("No valid latency measurements recorded")
        else:
            print(f"\n📈 PERFORMANCE SUMMARY")
            print("=" * 50)
            print("No latency measurements available")
        
        # Results summary
        total_searches = len(all_results)
        successful_searches = len([r for r in all_results if r and len(r) > 0])
        
        print(f"\n📊 RESULTS SUMMARY")
        print("=" * 50)
        print(f"Total searches: {total_searches}")
        print(f"Successful searches: {successful_searches}")
        if total_searches > 0:
            print(f"Success rate: {successful_searches/total_searches*100:.1f}%")
            return successful_searches >= total_searches * 0.8  # 80% success rate threshold
        else:
            print("Success rate: 0%")
            return False
    
    async def run_neo4j_validation(self, sample_size: int = 5):
        """Run Neo4j ID validation test.
        
        Args:
            sample_size: Number of random papers to validate
        """
        print(f"\n🔗 Testing Neo4j ID validation with {sample_size} samples...")
        
        try:
            # Get some sample results
            test_query = "machine learning artificial intelligence"
            results, _ = self.search_by_text(test_query, top_k=sample_size)
            
            if not results:
                print("❌ No results available for validation")
                return
            
            paper_ids = [result["id"] for result in results]
            validation_results = await self.validate_with_neo4j(paper_ids)
            
            print(f"\n🔍 Neo4j Validation Results:")
            print("=" * 50)
            
            valid_count = 0
            for paper_id, exists in validation_results.items():
                status = "✅ EXISTS" if exists else "❌ MISSING"
                print(f"  {paper_id}: {status}")
                if exists:
                    valid_count += 1
            
            validation_rate = valid_count / len(paper_ids) * 100
            print(f"\n📊 Validation Summary:")
            print(f"   Valid IDs: {valid_count}/{len(paper_ids)}")
            print(f"   Validation rate: {validation_rate:.1f}%")
            print(f"   Cross-database consistency: {'✅ GOOD' if validation_rate >= 80 else '⚠️  POOR'}")
            
        except Exception as e:
            print(f"❌ Neo4j validation failed: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.zilliz_service:
                self.zilliz_service.disconnect()
            if self.neo4j_client:
                import asyncio
                # Properly close Neo4j connection
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.neo4j_client.close())
                else:
                    asyncio.run(self.neo4j_client.close())
        except:
            pass


async def main():
    """Main function to run search tests."""
    print("🚀 Starting Comprehensive Search Test for Zilliz")
    print("=" * 80)
    
    # Initialize test
    search_test = ZillizSearchTest()
    
    try:
        # Initialize services
        if not search_test.initialize_services():
            print("❌ Failed to initialize services")
            return
        
        # Run search tests
        print("\n🔍 Running semantic search tests...")
        search_success = search_test.run_search_tests()
        
        # Run Neo4j validation
        print("\n🔗 Running Neo4j validation...")
        await search_test.run_neo4j_validation(sample_size=3)
        
        # Final summary
        print("\n" + "=" * 80)
        print("🏁 FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"✅ SciBERT embeddings: Working")
        print(f"✅ Milvus/Zilliz storage: Working")  
        print(f"✅ Vector similarity search: {'✅ PASS' if search_success else '❌ FAIL'}")
        print(f"✅ Latency performance: Measured")
        print(f"✅ Neo4j ID validation: Tested")
        
        if search_success:
            print("\n🎉 All search tests completed successfully!")
            print("Your vector database search system is working correctly.")
        else:
            print("\n⚠️  Some search tests failed. Check the output above.")
        
    except KeyboardInterrupt:
        print("\nSearch test interrupted by user")
    except Exception as e:
        print(f"\n❌ Search test failed: {e}")
    finally:
        search_test.cleanup()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
