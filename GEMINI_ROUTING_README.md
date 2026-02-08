# Gemini AI Integration for Intelligent Routing

## Overview

The routing decision engine now uses Google's Gemini AI to make intelligent routing decisions for the hybrid search system. This provides more sophisticated query analysis and routing strategy selection compared to the previous rule-based approach.

## Features

- **AI-Powered Routing**: Uses Gemini 1.5 Flash to analyze queries and select optimal routing strategies
- **Performance-Aware**: Incorporates historical performance data into routing decisions
- **Fallback Support**: Automatically falls back to rule-based routing if Gemini is unavailable
- **Configurable**: Supports various Gemini model configurations

## Setup

### 1. Get Gemini API Key

1. Visit [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the API key for configuration

### 2. Configure Environment

Add your Gemini API key to your environment:

```bash
# Option 1: Environment variable
export GEMINI_API_KEY="your_api_key_here"

# Option 2: .env file
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

### 3. Optional Configuration

```bash
# Model selection (default: gemini-1.5-flash)
GEMINI_MODEL=gemini-1.5-flash

# Temperature for response randomness (default: 0.1)
GEMINI_TEMPERATURE=0.1

# Maximum response tokens (default: 1000)
GEMINI_MAX_TOKENS=1000
```

## How It Works

### Routing Strategies

The system selects from three routing strategies:

1. **VECTOR_FIRST**: Optimal for semantic similarity searches
   - "Find papers similar to..."
   - "Papers about..."
   - Conceptual topic searches

2. **GRAPH_FIRST**: Optimal for structural queries
   - Paper ID lookups (W1234567, DOI, etc.)
   - Author relationship queries
   - Citation analysis

3. **PARALLEL**: Optimal for complex queries
   - Multi-part questions
   - Uncertain query types
   - Queries needing both approaches

### AI Decision Process

1. **Query Analysis**: Gemini analyzes the query structure and intent
2. **Context Integration**: Historical performance data is considered
3. **Strategy Selection**: AI selects the optimal routing strategy
4. **Fallback**: If Gemini fails, rule-based classification is used

## Performance Benefits

- **Improved Accuracy**: AI understands query nuances better than rules
- **Adaptive Learning**: Performance history influences future decisions
- **Reduced Latency**: Better routing reduces unnecessary search operations
- **Robust Fallback**: System remains functional without AI

## Testing

Test the Gemini integration:

```bash
python3 test_gemini_routing.py
```

This will show:
- Gemini integration status
- Routing decisions for various query types
- Fallback behavior when needed

## Monitoring

The system logs routing decisions and performance:

```python
# Check if Gemini is active
routing_engine.use_gemini  # True/False

# View performance history
routing_engine.performance_history
```

## Troubleshooting

### Common Issues

1. **"Gemini routing disabled"**
   - Check GEMINI_API_KEY is set correctly
   - Verify API key is valid at Google AI Studio

2. **Import errors**
   - Ensure `pip install google-generativeai` is run
   - Check requirements.txt includes the package

3. **Fallback routing only**
   - Gemini API may be down or rate-limited
   - Check API quotas in Google AI Studio

### Performance Tuning

- Lower `GEMINI_TEMPERATURE` for more consistent routing
- Increase `GEMINI_MAX_TOKENS` for complex query analysis
- Monitor `performance_history` to optimize strategy selection

## Examples

### Paper ID Query
```
Query: "who is the author of paper have id = W2036113194"
Gemini Decision: GRAPH_FIRST
Reasoning: Contains specific paper ID, requires graph traversal
```

### Conceptual Search
```
Query: "find papers similar to machine learning in healthcare"
Gemini Decision: VECTOR_FIRST
Reasoning: Semantic similarity search, conceptual matching needed
```

### Complex Query
```
Query: "papers about deep learning by Geoffrey Hinton and their citations"
Gemini Decision: PARALLEL
Reasoning: Combines conceptual search with relationship analysis
```