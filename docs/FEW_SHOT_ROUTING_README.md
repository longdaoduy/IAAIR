# Few-Shot Learning for Intelligent Query Routing

## Overview

The `RoutingDecisionEngine` now uses few-shot learning with Google's Gemini AI instead of rule-based query classification. This approach provides more intelligent, context-aware, and adaptable routing decisions for the hybrid search system.

## Key Improvements

### From Rule-Based to Few-Shot Learning

**Before (Rule-Based):**
- Fixed keyword matching
- Simple heuristics
- Limited context understanding
- Static decision boundaries

**After (Few-Shot Learning):**
- Context-aware pattern recognition
- Learning from curated examples
- Natural language understanding
- Adaptive reasoning with explanations

## Architecture

### Few-Shot Learning Components

1. **Training Examples**: 10 carefully curated query-routing examples
2. **Structured Prompts**: Template-based prompts with examples
3. **Response Parsing**: Structured output parsing for reliability
4. **Fallback Mechanism**: Automatic fallback to rule-based classification

### Query Classification Types

```python
QueryType.STRUCTURAL   # Paper IDs, authors, citations, relationships
QueryType.SEMANTIC     # Conceptual similarity, topics, themes  
QueryType.FACTUAL      # Specific factual questions
QueryType.HYBRID       # Complex queries needing multiple approaches
```

### Routing Strategies

```python
RoutingStrategy.VECTOR_FIRST  # Best for semantic similarity
RoutingStrategy.GRAPH_FIRST   # Best for exact matches, relationships
RoutingStrategy.PARALLEL      # Best for complex hybrid queries
```

## Few-Shot Training Examples

### Structural Queries → GRAPH_FIRST

```python
{
    "query": "who is the author of paper have id = W2036113194",
    "query_type": "STRUCTURAL",
    "routing": "GRAPH_FIRST", 
    "confidence": 0.95,
    "reasoning": "Specific paper ID with author lookup - requires graph traversal for exact match"
}
```

### Semantic Queries → VECTOR_FIRST

```python
{
    "query": "papers about machine learning in healthcare",
    "query_type": "SEMANTIC",
    "routing": "VECTOR_FIRST",
    "confidence": 0.90,
    "reasoning": "Conceptual topic search - vector embeddings capture semantic similarity best"
}
```

### Hybrid Queries → PARALLEL

```python
{
    "query": "recent trends and applications of neural networks in NLP",
    "query_type": "HYBRID",
    "routing": "PARALLEL",
    "confidence": 0.85,
    "reasoning": "Complex query combining trends (semantic) with specific domain (structural) - needs both approaches"
}
```

## Implementation Details

### Few-Shot Prompt Structure

```
You are an expert system for academic paper search routing. Analyze the query and determine the optimal search strategy based on these examples:

Example 1:
Query: "who is the author of paper have id = W2036113194"
Analysis:
- Query Type: STRUCTURAL
- Routing Strategy: GRAPH_FIRST
- Confidence: 0.95
- Reasoning: Specific paper ID with author lookup - requires graph traversal for exact match

[... more examples ...]

Now analyze this new query:
Query: "{user_query}"

Provide your analysis in this exact format:
Query Type: [SEMANTIC|STRUCTURAL|FACTUAL|HYBRID]
Routing Strategy: [VECTOR_FIRST|GRAPH_FIRST|PARALLEL]  
Confidence: [0.0-1.0]
Reasoning: [Brief explanation of your decision]
```

### Response Parsing

The system parses structured responses:
```
Query Type: STRUCTURAL
Routing Strategy: GRAPH_FIRST
Confidence: 0.92
Reasoning: Author-specific search requires graph relationships for exact authorship matches
```

## Performance Benefits

### Accuracy Improvements

- **Context Understanding**: Better comprehension of query intent
- **Pattern Recognition**: Learns from examples rather than rigid rules
- **Confidence Scoring**: More nuanced confidence assessments
- **Reasoning**: Provides explanations for routing decisions

### Adaptability

- **New Query Types**: Handles novel query patterns better
- **Domain Adaptation**: Can incorporate domain-specific examples
- **Performance Learning**: Integrates historical performance data
- **Continuous Improvement**: Easy to add new examples

## API Integration

### Backward Compatibility

The API remains unchanged - few-shot learning is internal to the routing engine:

```python
# Same API as before
routing_strategy = routing_engine.decide_routing(query, request)
query_type, confidence = routing_engine.get_query_classification(query)
```

### Enhanced Classification Method

```python
def get_query_classification(self, query: str) -> Tuple[QueryType, float]:
    """Get query classification using few-shot learning or fallback to rule-based."""
    if self.use_gemini:
        # Use few-shot learning
        result = self._few_shot_route_decision(query, request)
        if result:
            _, query_type, confidence = result
            return query_type, confidence
    
    # Fallback to rule-based classifier
    return self.query_classifier.classify_query(query)
```

## Configuration

### Environment Setup

```bash
# Required for few-shot learning
GEMINI_API_KEY=your_api_key_here

# Optional Gemini configuration
GEMINI_MODEL=gemini-1.5-flash
GEMINI_TEMPERATURE=0.1
GEMINI_MAX_TOKENS=1000
```

### Fallback Behavior

- **No API Key**: Automatically uses rule-based `QueryClassifier`
- **API Failure**: Graceful fallback with error logging
- **Parsing Error**: Falls back to rule-based classification
- **No Impact**: Search functionality remains fully operational

## Testing and Validation

### Test Categories

1. **Structural Queries**: Paper IDs, author lookups, citations
2. **Semantic Queries**: Conceptual similarity, topic searches
3. **Hybrid Queries**: Complex multi-part queries
4. **Edge Cases**: Novel query patterns, ambiguous cases

### Running Tests

```bash
# Test few-shot learning capabilities
python3 test_few_shot_routing.py

# Compare approaches
python3 test_few_shot_routing.py --compare
```

### Expected Results

```
Query: who is the author of paper have id = W2036113194
📊 Classification: STRUCTURAL, confidence: 0.95
🎯 Selected strategy: GRAPH_FIRST
✅ Expected: GRAPH_FIRST

Query: papers about machine learning in healthcare  
📊 Classification: SEMANTIC, confidence: 0.90
🎯 Selected strategy: VECTOR_FIRST
✅ Expected: VECTOR_FIRST
```

## Monitoring and Analytics

### Enhanced Logging

```python
logger.info(f"Few-shot learning selected: {strategy.value} (query_type: {query_type}, confidence: {confidence})")
```

### Performance Tracking

- Classification accuracy vs ground truth
- Routing decision effectiveness
- Response parsing success rates
- Fallback usage frequency

## Customization

### Adding New Examples

```python
def _build_few_shot_examples(self) -> list:
    return [
        # ... existing examples ...
        {
            "query": "your new example query",
            "query_type": "SEMANTIC|STRUCTURAL|FACTUAL|HYBRID",
            "routing": "VECTOR_FIRST|GRAPH_FIRST|PARALLEL",
            "confidence": 0.85,
            "reasoning": "Clear explanation of routing decision"
        }
    ]
```

### Domain-Specific Examples

- **Medical Research**: Medical terminology, clinical studies
- **Computer Science**: Technical concepts, algorithms
- **Interdisciplinary**: Cross-domain research patterns

## Troubleshooting

### Common Issues

1. **"Few-shot learning failed"**: Check Gemini API key and network
2. **Unexpected routing**: Validate few-shot examples and prompt structure
3. **Low confidence scores**: Review and improve training examples
4. **Parsing errors**: Check response format consistency

### Debug Information

```python
# Enable debug logging
logging.getLogger('models.engines.RoutingDecisionEngine').setLevel(logging.DEBUG)

# Check few-shot examples
routing_engine = RoutingDecisionEngine()
print(f"Examples loaded: {len(routing_engine.few_shot_examples)}")
```

## Future Enhancements

### Potential Improvements

- **Dynamic Example Selection**: Choose most relevant examples for each query
- **Performance-Based Learning**: Automatically update examples based on results
- **Multi-Modal Examples**: Include query context and user feedback
- **Domain Adaptation**: Specialized example sets for different research domains

### Integration Opportunities

- **User Feedback**: Learn from user routing preferences
- **Query Analytics**: Identify common patterns for new examples
- **A/B Testing**: Compare few-shot vs rule-based performance
- **Continuous Learning**: Update examples based on system performance