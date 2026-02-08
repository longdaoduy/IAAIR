from models.entities.retrieval.RoutingStrategy import RoutingStrategy
from models.entities.retrieval.QueryType import QueryType
from models.entities.retrieval.HybridSearchRequest import HybridSearchRequest
from models.engines.QueryClassifier import QueryClassifier
from models.configurators.LlamaConfig import LlamaConfig
import logging
import json
import os
from typing import Optional, Tuple

# ===============================================================================
# HYBRID FUSION SYSTEM
# ===============================================================================

logger = logging.getLogger(__name__)

class RoutingDecisionEngine:
    """Decide optimal routing strategy based on query and system state using Few-Shot Learning with Llama AI."""

    def __init__(self):
        self.query_classifier = QueryClassifier()  # Keep as fallback
        self.performance_history = {}  # Track routing performance
        
        # Initialize Llama for few-shot learning
        self.llama_config = LlamaConfig()
        self.llama_model = self.llama_config.initialize_client()
        self.use_llama = self.llama_model is not None
        
        # Load few-shot learning examples from file
        self.few_shot_examples = self._load_few_shot_examples()

    def decide_routing(self, query: str, request: HybridSearchRequest) -> RoutingStrategy:
        """Decide routing strategy based on query analysis using Few-Shot Learning with Llama AI."""
        routing_result = self._few_shot_route_decision(query, request)

        if routing_result is None:
            logger.warning("Few-shot routing failed, falling back to rule-based routing")
            # Fallback to rule-based routing
            query_type, confidence = self.query_classifier.classify_query(query)
            if query_type == QueryType.STRUCTURAL:
                return RoutingStrategy.GRAPH_FIRST
            elif query_type == QueryType.SEMANTIC:
                return RoutingStrategy.VECTOR_FIRST
            else:
                return RoutingStrategy.PARALLEL
                
        strategy, query_type, confidence = routing_result
        logger.info(
            f"Few-shot learning selected: {strategy.value} (query_type: {query_type}, confidence: {confidence})")
        return strategy

    def _few_shot_route_decision(self, query: str, request: HybridSearchRequest) -> Optional[Tuple[RoutingStrategy, QueryType, float]]:
        """Use few-shot learning with Llama to make intelligent routing decisions."""
        try:
            # Check if Llama is available
            if not self.use_llama or self.llama_model is None:
                logger.warning("Llama model not available for few-shot routing")
                return None
                
            # Get performance context
            performance_context = self._get_performance_context()
            
            # Create few-shot learning prompt
            prompt = self._build_few_shot_prompt(query, performance_context)
            
            # Use the Llama API to generate content
            response_text = self.llama_config.generate_text(
                client=self.llama_model,
                prompt=prompt
            )
            
            if not response_text:
                logger.error("Empty or invalid response from Llama model")
                return None
            
            # Parse the structured response
            return self._parse_few_shot_response(response_text)
                
        except Exception as e:
            logger.error(f"Error in few-shot routing decision: {e}")
            return None
    
    def _load_few_shot_examples(self) -> list:
        """Load few-shot learning examples from JSON file."""
        # Get the path to the data directory relative to the current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        examples_path = os.path.join(project_root, 'data', 'few_shot_examples.json')

        with open(examples_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            examples = data.get('few_shot_examples', [])

        logger.info(f"Loaded {len(examples)} few-shot examples from {examples_path}")

        # Log metadata if available
        if 'metadata' in data:
            metadata = data['metadata']
            logger.info(f"Examples metadata: version={metadata.get('version')}, "
                        f"query_types={metadata.get('query_types')}, "
                        f"routing_strategies={metadata.get('routing_strategies')}")

        return examples

    def _build_few_shot_prompt(self, query: str, performance_context: str) -> str:
        """Build few-shot learning prompt for query classification and routing."""
        
        # Format few-shot examples
        examples_text = ""
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"""
Example {i}:
Query: "{example['query']}"
Analysis:
- Query Type: {example['query_type']}
- Routing Strategy: {example['routing']}
- Confidence: {example['confidence']}
- Reasoning: {example['reasoning']}
"""
        
        prompt = f"""You are an expert system for academic paper search routing. Analyze the query and determine the optimal search strategy based on these examples:

{examples_text}

Performance Context: {performance_context}

Now analyze this new query:
Query: "{query}"

Provide your analysis in this exact format:
Query Type: [SEMANTIC|STRUCTURAL|FACTUAL|HYBRID]
Routing Strategy: [VECTOR_FIRST|GRAPH_FIRST|PARALLEL]  
Confidence: [0.0-1.0]
Reasoning: [Brief explanation of your decision]

Guidelines:
- STRUCTURAL queries involve specific IDs, authors, citations, relationships
- SEMANTIC queries involve conceptual similarity, topics, themes
- FACTUAL queries ask specific factual questions
- HYBRID queries combine multiple aspects
- VECTOR_FIRST: Best for semantic similarity and conceptual searches
- GRAPH_FIRST: Best for exact matches, relationships, and structured data
- PARALLEL: Best for complex queries needing both approaches
- Consider performance history when available
- Confidence should reflect certainty in classification and routing choice
"""
        return prompt
    
    def _parse_few_shot_response(self, response_text: str) -> Optional[Tuple[RoutingStrategy, QueryType, float]]:
        """Parse the structured response from few-shot learning."""
        try:
            lines = response_text.strip().split('\n')
            
            query_type = None
            routing_strategy = None  
            confidence = 0.8  # default
            
            for line in lines:
                line = line.strip()
                if line.startswith('Query Type:'):
                    type_str = line.split(':', 1)[1].strip().upper()
                    if type_str == 'SEMANTIC':
                        query_type = QueryType.SEMANTIC
                    elif type_str == 'STRUCTURAL':
                        query_type = QueryType.STRUCTURAL
                    elif type_str == 'FACTUAL':
                        query_type = QueryType.FACTUAL
                    elif type_str == 'HYBRID':
                        query_type = QueryType.HYBRID
                        
                elif line.startswith('Routing Strategy:'):
                    strategy_str = line.split(':', 1)[1].strip().upper()
                    if strategy_str == 'VECTOR_FIRST':
                        routing_strategy = RoutingStrategy.VECTOR_FIRST
                    elif strategy_str == 'GRAPH_FIRST':
                        routing_strategy = RoutingStrategy.GRAPH_FIRST
                    elif strategy_str == 'PARALLEL':
                        routing_strategy = RoutingStrategy.PARALLEL
                        
                elif line.startswith('Confidence:'):
                    try:
                        confidence = float(line.split(':', 1)[1].strip())
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                    except ValueError:
                        confidence = 0.8  # default
            
            if query_type is not None and routing_strategy is not None:
                return routing_strategy, query_type, confidence
            else:
                logger.warning(f"Could not parse few-shot response: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error parsing few-shot response: {e}")
            return None
    
    def get_query_classification(self, query: str) -> Tuple[QueryType, float]:
        """Get query classification using few-shot learning or fallback to rule-based."""
        if self.use_llama:
            # Create a mock request for few-shot classification
            mock_request = HybridSearchRequest(
                query=query,
                routing_strategy=RoutingStrategy.ADAPTIVE
            )
            
            result = self._few_shot_route_decision(query, mock_request)
            if result:
                _, query_type, confidence = result
                return query_type, confidence
        
        # Fallback to rule-based classifier
        return self.query_classifier.classify_query(query)
    
    def _get_performance_context(self) -> str:
        """Get performance history context for Llama."""
        if not self.performance_history:
            return "No performance history available"
            
        context = []
        for key, metrics in self.performance_history.items():
            if metrics['latencies'] and metrics['relevance_scores']:
                avg_latency = sum(metrics['latencies']) / len(metrics['latencies'])
                avg_relevance = sum(metrics['relevance_scores']) / len(metrics['relevance_scores'])
                context.append(f"{key}: avg_latency={avg_latency:.2f}s, avg_relevance={avg_relevance:.2f}")
                
        return "; ".join(context[:5])  # Limit context size

    def update_performance(self, strategy: RoutingStrategy, query_type: QueryType,
                           latency: float, relevance_score: float):
        """Update performance tracking for adaptive routing."""
        key = f"{strategy}_{query_type}"
        if key not in self.performance_history:
            self.performance_history[key] = {'latencies': [], 'relevance_scores': []}

        self.performance_history[key]['latencies'].append(latency)
        self.performance_history[key]['relevance_scores'].append(relevance_score)

        # Keep only recent history (last 100 queries)
        for metric_list in self.performance_history[key].values():
            if len(metric_list) > 100:
                metric_list.pop(0)