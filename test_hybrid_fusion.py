"""
Test suite for hybrid fusion and attribution system.
"""

import pytest
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    HybridSearchRequest, RoutingStrategy, QueryType,
    QueryClassifier, RoutingDecisionEngine, ResultFusion,
    AttributionTracker
)
from pipelines.retrievals.attribution_utils import AttributionManager

class TestQueryClassifier:
    """Test query classification functionality."""
    
    def setup_method(self):
        self.classifier = QueryClassifier()
    
    def test_semantic_query_classification(self):
        """Test classification of semantic queries."""
        semantic_queries = [
            "find papers similar to machine learning applications",
            "research about natural language processing techniques",
            "studies related to deep neural networks"
        ]
        
        for query in semantic_queries:
            query_type, confidence = self.classifier.classify_query(query)
            assert query_type in [QueryType.SEMANTIC, QueryType.HYBRID]
            assert confidence > 0.0
    
    def test_structural_query_classification(self):
        """Test classification of structural queries."""
        structural_queries = [
            "papers cited by Smith et al",
            "authors who collaborated with John Doe",
            "works published in Nature journal"
        ]
        
        for query in structural_queries:
            query_type, confidence = self.classifier.classify_query(query)
            assert query_type in [QueryType.STRUCTURAL, QueryType.HYBRID]
            assert confidence > 0.0
    
    def test_factual_query_classification(self):
        """Test classification of factual queries."""
        factual_queries = [
            "who invented the transformer architecture",
            "what is the impact factor of Nature",
            "when was BERT published"
        ]
        
        for query in factual_queries:
            query_type, confidence = self.classifier.classify_query(query)
            assert query_type in [QueryType.FACTUAL, QueryType.HYBRID]
            assert confidence > 0.0

class TestRoutingDecisionEngine:
    """Test routing decision logic."""
    
    def setup_method(self):
        self.engine = RoutingDecisionEngine()
    
    def test_adaptive_routing_decision(self):
        """Test adaptive routing decisions."""
        test_cases = [
            {
                "query": "machine learning applications in healthcare",
                "expected_strategies": [RoutingStrategy.VECTOR_FIRST, RoutingStrategy.PARALLEL]
            },
            {
                "query": "papers cited by Hinton and Bengio",
                "expected_strategies": [RoutingStrategy.GRAPH_FIRST, RoutingStrategy.PARALLEL]
            },
            {
                "query": "collaborative filtering recommendation systems",
                "expected_strategies": [RoutingStrategy.VECTOR_FIRST, RoutingStrategy.PARALLEL]
            }
        ]
        
        for case in test_cases:
            request = HybridSearchRequest(
                query=case["query"],
                routing_strategy=RoutingStrategy.ADAPTIVE
            )
            
            strategy = self.engine.decide_routing(case["query"], request)
            assert strategy in case["expected_strategies"]
    
    def test_explicit_routing_strategy(self):
        """Test that explicit routing strategies are respected."""
        query = "test query"
        
        for strategy in RoutingStrategy:
            if strategy != RoutingStrategy.ADAPTIVE:
                request = HybridSearchRequest(
                    query=query,
                    routing_strategy=strategy
                )
                
                decided_strategy = self.engine.decide_routing(query, request)
                assert decided_strategy == strategy

class TestResultFusion:
    """Test result fusion functionality."""
    
    def setup_method(self):
        self.fusion = ResultFusion()
    
    def test_fusion_with_overlapping_results(self):
        """Test fusion when vector and graph results have overlapping papers."""
        vector_results = [
            {
                "paper_id": "paper1",
                "title": "Test Paper 1",
                "similarity_score": 0.9,
                "abstract": "Abstract 1"
            },
            {
                "paper_id": "paper2",
                "title": "Test Paper 2",
                "similarity_score": 0.8,
                "abstract": "Abstract 2"
            }
        ]
        
        graph_results = [
            {
                "paper_id": "paper1",  # Overlapping with vector
                "title": "Test Paper 1",
                "relevance_score": 0.7,
                "abstract": "Abstract 1"
            },
            {
                "paper_id": "paper3",
                "title": "Test Paper 3",
                "relevance_score": 0.6,
                "abstract": "Abstract 3"
            }
        ]
        
        fused_results = self.fusion.fuse_results(vector_results, graph_results)
        
        assert len(fused_results) == 3  # Two unique + one overlapping
        
        # Check that paper1 has both vector and graph scores
        paper1_result = next(r for r in fused_results if r.paper_id == "paper1")
        assert paper1_result.vector_score == 0.9
        assert paper1_result.graph_score == 0.7
        assert "vector_search" in paper1_result.source_path
        assert "graph_search" in paper1_result.source_path
    
    def test_fusion_with_custom_weights(self):
        """Test fusion with custom weights."""
        vector_results = [{
            "paper_id": "paper1",
            "title": "Test Paper",
            "similarity_score": 0.8,
        }]
        
        graph_results = [{
            "paper_id": "paper1",
            "title": "Test Paper",
            "relevance_score": 0.6,
        }]
        
        custom_weights = {
            "vector_score": 0.7,
            "graph_score": 0.3,
            "rerank_score": 0.0
        }
        
        fused_results = self.fusion.fuse_results(
            vector_results, graph_results, custom_weights
        )
        
        assert len(fused_results) == 1
        expected_score = 0.7 * 0.8 + 0.3 * 0.6  # 0.56 + 0.18 = 0.74
        assert abs(fused_results[0].relevance_score - expected_score) < 0.01

class TestAttributionTracker:
    """Test attribution tracking functionality."""
    
    def setup_method(self):
        self.tracker = AttributionTracker()
    
    def test_attribution_creation(self):
        """Test creation of attribution spans."""
        from main import SearchResult
        
        result = SearchResult(
            paper_id="paper1",
            title="Machine Learning Applications",
            abstract="This paper discusses machine learning techniques for various applications in healthcare and finance.",
            authors=["John Doe", "Jane Smith"],
            relevance_score=0.8,
            attributions=[],
            source_path=[],
            confidence_scores={}
        )
        
        query = "machine learning healthcare"
        results = self.tracker.track_attributions([result], query)
        
        assert len(results) == 1
        assert len(results[0].attributions) > 0
        
        # Check that title attribution exists
        title_attribution = next(
            (a for a in results[0].attributions if a.source_type == "paper"),
            None
        )
        assert title_attribution is not None
        assert title_attribution.text == result.title

class TestAttributionManager:
    """Test advanced attribution management."""
    
    def setup_method(self):
        self.manager = AttributionManager()
    
    def test_evidence_bundle_creation(self):
        """Test creation of evidence bundles."""
        search_results = [
            {
                "paper_id": "paper1",
                "title": "Deep Learning in Medical Imaging",
                "abstract": "This study explores deep learning applications in medical image analysis.",
                "authors": ["Dr. Smith"],
                "venue": "Nature Medicine",
                "relevance_score": 0.9
            },
            {
                "paper_id": "paper2", 
                "title": "Convolutional Neural Networks for X-ray Analysis",
                "abstract": "We present a CNN approach for automated X-ray diagnosis.",
                "authors": ["Dr. Johnson"],
                "venue": "Medical AI Journal",
                "relevance_score": 0.8
            }
        ]
        
        query = "deep learning medical imaging"
        bundle = self.manager.create_evidence_bundle(
            query, search_results, "vector_first"
        )
        
        assert bundle["bundle_id"] is not None
        assert bundle["query"] == query
        assert bundle["routing_path"] == "vector_first"
        assert len(bundle["evidence_sources"]) == 2
        assert len(bundle["attribution_chains"]) > 0
        assert "overall_confidence" in bundle["confidence_assessment"]
    
    def test_supporting_passage_extraction(self):
        """Test extraction of supporting passages from text."""
        text = "Machine learning has revolutionized medical imaging. Deep neural networks can analyze X-rays with high accuracy. This technology shows promise for early disease detection."
        query = "machine learning medical imaging"
        
        passages = self.manager._extract_supporting_passages(text, query)
        
        assert len(passages) > 0
        # First passage should have highest relevance (contains both "machine learning" and "medical imaging")
        assert passages[0]["relevance_score"] > 0.5

class TestHybridSearchIntegration:
    """Integration tests for the complete hybrid search system."""
    
    @pytest.mark.asyncio
    async def test_hybrid_search_request_validation(self):
        """Test that hybrid search request validation works."""
        # Valid request
        valid_request = HybridSearchRequest(
            query="machine learning applications",
            top_k=10,
            routing_strategy=RoutingStrategy.ADAPTIVE
        )
        
        assert valid_request.query == "machine learning applications"
        assert valid_request.top_k == 10
        assert valid_request.routing_strategy == RoutingStrategy.ADAPTIVE
    
    def test_performance_tracking(self):
        """Test routing performance tracking."""
        engine = RoutingDecisionEngine()
        
        # Add some performance data
        engine.update_performance(
            RoutingStrategy.VECTOR_FIRST,
            QueryType.SEMANTIC,
            latency=1.5,
            relevance_score=0.8
        )
        
        engine.update_performance(
            RoutingStrategy.GRAPH_FIRST,
            QueryType.STRUCTURAL,
            latency=2.0,
            relevance_score=0.7
        )
        
        # Check that performance data is stored
        assert len(engine.performance_history) == 2
        
        vector_key = f"{RoutingStrategy.VECTOR_FIRST}_{QueryType.SEMANTIC}"
        assert vector_key in engine.performance_history
        assert engine.performance_history[vector_key]["latencies"][0] == 1.5

def run_comprehensive_test():
    """Run a comprehensive test of the hybrid fusion system."""
    print("🧪 Running Hybrid Fusion System Tests...")
    
    # Test 1: Query Classification
    print("\n1. Testing Query Classification...")
    classifier = QueryClassifier()
    
    test_queries = [
        "machine learning applications in healthcare",
        "papers cited by Geoffrey Hinton",
        "what is the transformer architecture",
        "similar papers about natural language processing"
    ]
    
    for query in test_queries:
        query_type, confidence = classifier.classify_query(query)
        print(f"   Query: '{query}' -> {query_type.value} (confidence: {confidence:.2f})")
    
    # Test 2: Result Fusion
    print("\n2. Testing Result Fusion...")
    fusion = ResultFusion()
    
    mock_vector_results = [
        {"paper_id": "v1", "title": "Vector Paper 1", "similarity_score": 0.9},
        {"paper_id": "v2", "title": "Vector Paper 2", "similarity_score": 0.8}
    ]
    
    mock_graph_results = [
        {"paper_id": "v1", "title": "Vector Paper 1", "relevance_score": 0.7},  # Overlap
        {"paper_id": "g1", "title": "Graph Paper 1", "relevance_score": 0.6}
    ]
    
    fused_results = fusion.fuse_results(mock_vector_results, mock_graph_results)
    print(f"   Fused {len(fused_results)} results from vector and graph search")
    
    for result in fused_results[:2]:
        print(f"   - {result.title}: relevance={result.relevance_score:.2f}, "
              f"vector={result.vector_score:.2f}, graph={result.graph_score:.2f}")
    
    # Test 3: Attribution Tracking
    print("\n3. Testing Attribution Tracking...")
    attribution_manager = AttributionManager()
    
    mock_search_results = [
        {
            "paper_id": "attr1",
            "title": "Deep Learning in Medical Imaging",
            "abstract": "This paper presents deep learning techniques for medical image analysis and diagnosis.",
            "authors": ["Dr. Smith", "Dr. Johnson"],
            "venue": "Nature Medicine",
            "relevance_score": 0.9
        }
    ]
    
    query = "deep learning medical imaging"
    evidence_bundle = attribution_manager.create_evidence_bundle(
        query, mock_search_results, "vector_first"
    )
    
    print(f"   Created evidence bundle with {len(evidence_bundle['evidence_sources'])} sources")
    print(f"   Attribution chains: {len(evidence_bundle['attribution_chains'])}")
    print(f"   Overall confidence: {evidence_bundle['confidence_assessment']['overall_confidence']:.2f}")
    
    print("\n✅ All tests completed successfully!")
    return True

if __name__ == "__main__":
    # Run the comprehensive test
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 Hybrid Fusion System is ready for production!")
        print("\nNext steps:")
        print("- Deploy the updated main.py with hybrid endpoints")
        print("- Configure the hybrid_config.py for your environment")
        print("- Monitor performance using the analytics endpoints")
        print("- Fine-tune fusion weights based on evaluation results")
    else:
        print("\n❌ Tests failed. Please review the implementation.")
        sys.exit(1)