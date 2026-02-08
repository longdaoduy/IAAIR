class MockClient:
    def __init__(self):
        pass

    def generate_content(self, prompt):
        # Check if this is a routing decision prompt or a user question prompt
        if "Query Type:" in prompt and "Routing Strategy:" in prompt and "academic paper search routing" in prompt:
            # This is a routing decision prompt - provide routing analysis
            prompt_lower = prompt.lower()

            if 'paper' in prompt_lower and ('id' in prompt_lower or 'w2' in prompt_lower):
                response_text = """Query Type: STRUCTURAL
Routing Strategy: GRAPH_FIRST
Confidence: 0.9
Reasoning: Query contains specific paper ID, best handled by graph search for exact matches."""
            elif 'author' in prompt_lower and ('wrote' in prompt_lower or 'by' in prompt_lower):
                response_text = """Query Type: STRUCTURAL  
Routing Strategy: GRAPH_FIRST
Confidence: 0.8
Reasoning: Query asks about authorship, which requires graph traversal of author-paper relationships."""
            elif any(word in prompt_lower for word in ['concept', 'topic', 'about', 'semantic', 'similarity']):
                response_text = """Query Type: SEMANTIC
Routing Strategy: VECTOR_FIRST
Confidence: 0.8
Reasoning: Query involves conceptual similarity and topics, best handled by vector search."""
            elif 'citation' in prompt_lower or 'reference' in prompt_lower:
                response_text = """Query Type: HYBRID
Routing Strategy: PARALLEL
Confidence: 0.7
Reasoning: Citations involve both semantic content and structural relationships, needs both approaches."""
            else:
                response_text = """Query Type: HYBRID
Routing Strategy: PARALLEL
Confidence: 0.6
Reasoning: General query that may benefit from both vector and graph search approaches."""
        else:
            # This is a user question prompt - provide a helpful answer
            prompt_lower = prompt.lower()

            if "search results:" in prompt_lower and "answer:" in prompt_lower:
                # Extract the question from the prompt
                if "who is the author" in prompt_lower:
                    response_text = """Based on the search results provided, I can help identify the authors of the paper. However, I need to examine the specific search results to provide accurate author information. If the search results contain the paper details, I would list the authors clearly. For example, if this were about the U-Net paper, the primary authors would typically include Olaf Ronneberger and colleagues from the University of Freiburg who developed this influential convolutional network architecture for biomedical image segmentation."""
                elif "what is" in prompt_lower or "about" in prompt_lower:
                    response_text = """Based on the search results, I can provide information about the topic you're asking about. The search results should contain relevant papers and their abstracts that address your question. I'll synthesize the key findings and provide a comprehensive answer based on the available information."""
                elif "how does" in prompt_lower or "how to" in prompt_lower:
                    response_text = """Based on the research papers found, I can explain the methodology or approach you're asking about. The search results should contain technical details and explanations that I can use to provide a clear, step-by-step answer to your question."""
                else:
                    response_text = """Based on the search results provided, I can see relevant papers related to your question. Let me analyze the information and provide a comprehensive answer that addresses your specific query using the findings from these research papers."""
            else:
                response_text = "I'm a helpful research assistant ready to answer your questions based on academic search results."

        return type('Response', (), {'text': response_text})()
