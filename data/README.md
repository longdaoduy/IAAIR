# Few-Shot Examples Data Management

## Overview

The few-shot learning examples for query routing are now stored in a structured JSON file instead of being hardcoded. This approach provides better maintainability, version control, and extensibility.

## File Structure

### Examples File: `data/few_shot_examples.json`

```json
{
  "few_shot_examples": [
    {
      "query": "who is the author of paper have id = W2036113194",
      "query_type": "STRUCTURAL",
      "routing": "GRAPH_FIRST",
      "confidence": 0.95,
      "reasoning": "Specific paper ID with author lookup - requires graph traversal for exact match"
    }
  ],
  "metadata": {
    "version": "1.0",
    "created": "2026-02-08",
    "description": "Few-shot learning examples for query routing classification",
    "total_examples": 15,
    "query_types": {
      "STRUCTURAL": 6,
      "SEMANTIC": 6,
      "HYBRID": 3,
      "FACTUAL": 1
    },
    "routing_strategies": {
      "GRAPH_FIRST": 7,
      "VECTOR_FIRST": 6,
      "PARALLEL": 4
    }
  }
}
```

### Example Schema

Each example must contain:
- **query**: The input query string
- **query_type**: One of `STRUCTURAL`, `SEMANTIC`, `FACTUAL`, `HYBRID`
- **routing**: One of `VECTOR_FIRST`, `GRAPH_FIRST`, `PARALLEL`
- **confidence**: Float between 0.0 and 1.0
- **reasoning**: Human-readable explanation for the routing decision

## Management Tools

### CLI Utility: `few_shot_manager.py`

```bash
# View all examples
python few_shot_manager.py view

# Add new example interactively
python few_shot_manager.py add

# Validate examples format and consistency
python few_shot_manager.py validate

# Show statistics only
python few_shot_manager.py stats
```

### Example Usage

```bash
$ python few_shot_manager.py view
================================================================================
📚 Few-Shot Learning Examples (15 total)
================================================================================
Version: 1.0
Created: 2026-02-08
Description: Few-shot learning examples for query routing classification

 1. Query: "who is the author of paper have id = W2036113194"
    Type: STRUCTURAL    Strategy: GRAPH_FIRST    Confidence: 0.95
    Reasoning: Specific paper ID with author lookup - requires graph traversal for exact match

 2. Query: "papers about machine learning in healthcare"
    Type: SEMANTIC      Strategy: VECTOR_FIRST   Confidence: 0.9
    Reasoning: Conceptual topic search - vector embeddings capture semantic similarity best
```

## Loading Mechanism

The `RoutingDecisionEngine` automatically loads examples on initialization:

```python
class RoutingDecisionEngine:
    def __init__(self):
        # ... other initialization ...
        self.few_shot_examples = self._load_few_shot_examples()
    
    def _load_few_shot_examples(self) -> list:
        """Load examples from JSON file with error handling."""
        try:
            # Load from data/few_shot_examples.json
            with open(examples_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('few_shot_examples', [])
        except Exception as e:
            logger.warning(f"Failed to load examples: {e}")
            return self._get_fallback_examples()
```

## Error Handling

### Robust Fallback System

1. **File Not Found**: Uses minimal hardcoded fallback examples
2. **JSON Parse Error**: Falls back to basic examples with error logging
3. **Schema Validation**: Continues with valid examples, logs invalid ones
4. **Runtime Errors**: Graceful degradation to rule-based classification

### Fallback Examples

If the JSON file cannot be loaded, the system uses these minimal examples:
- One STRUCTURAL example (paper ID lookup)
- One SEMANTIC example (topic search)
- One HYBRID example (complex query)

## Best Practices

### Adding New Examples

1. **Diverse Coverage**: Ensure examples cover all query types and routing strategies
2. **High Quality**: Use real-world queries with clear reasoning
3. **Balanced Distribution**: Maintain roughly equal representation across categories
4. **Clear Reasoning**: Provide detailed explanations for routing decisions

### Example Categories

**STRUCTURAL → GRAPH_FIRST**
- Paper ID lookups (`W1234567`, `DOI:...`)
- Author-specific searches
- Citation analysis
- Collaboration networks

**SEMANTIC → VECTOR_FIRST**
- Conceptual similarity (`papers about...`, `similar to...`)
- Topic-based searches
- Thematic research areas
- Broad subject queries

**HYBRID → PARALLEL**
- Multi-part queries combining structure and semantics
- Trend analysis with specific constraints
- Cross-domain research queries
- Complex analytical questions

**FACTUAL → (Strategy varies)**
- Specific factual questions
- Count queries (`how many...`)
- Temporal queries (`when...`, `recent...`)

## Maintenance

### Regular Tasks

1. **Validation**: Run `python few_shot_manager.py validate` regularly
2. **Balance Check**: Monitor distribution across categories
3. **Performance Review**: Analyze routing decisions and update examples
4. **Version Control**: Commit changes to examples file with clear messages

### Quality Assurance

```bash
# Validate before committing changes
python few_shot_manager.py validate

# Check current statistics
python few_shot_manager.py stats

# Test with updated examples
python test_few_shot_routing.py
```

## Integration Testing

### Test Files

- `test_few_shot_routing.py`: Comprehensive routing tests
- `few_shot_manager.py`: Management utility with validation

### Running Tests

```bash
# Test routing with current examples
python test_few_shot_routing.py

# Compare few-shot vs rule-based approaches
python test_few_shot_routing.py --compare

# Validate examples consistency
python few_shot_manager.py validate
```

## Version History

### Version 1.0 (2026-02-08)
- Initial JSON-based examples file
- 15 curated examples across all categories
- Balanced distribution of query types and routing strategies
- Comprehensive metadata tracking

## Future Enhancements

### Planned Features

1. **Dynamic Example Selection**: Choose most relevant examples for each query
2. **Performance-Based Updates**: Automatically improve examples based on routing success
3. **Domain-Specific Examples**: Specialized examples for different research fields
4. **User Feedback Integration**: Learn from user interactions and preferences

### Schema Extensions

```json
{
  "query": "example query",
  "query_type": "STRUCTURAL",
  "routing": "GRAPH_FIRST",
  "confidence": 0.95,
  "reasoning": "explanation",
  "domain": "computer_science",           // Future: domain-specific routing
  "performance_score": 0.87,             // Future: track actual performance
  "user_feedback": "positive",           // Future: incorporate user feedback
  "created_date": "2026-02-08",         // Future: example lifecycle tracking
  "last_used": "2026-02-10"             // Future: usage analytics
}
```