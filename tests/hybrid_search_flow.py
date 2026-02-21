#!/usr/bin/env python3
"""
Hybrid Search System Flow Visualization
Creates a visual diagram of the hybrid fusion architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
colors = {
    'input': '#E8F4FD',
    'routing': '#FFE6CC', 
    'vector': '#D4EDDA',
    'graph': '#F8D7DA',
    'fusion': '#E2E3E5',
    'attribution': '#FFF3CD',
    'output': '#D1ECF1'
}

# Helper function to create boxes
def create_box(ax, xy, width, height, text, color, fontsize=9):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(xy[0] + width/2, xy[1] + height/2, text,
            ha='center', va='center', fontsize=fontsize, wrap=True)

# Helper function to create arrows
def create_arrow(ax, start, end, color='black', style='->', linewidth=2):
    arrow = ConnectionPatch(start, end, "data", "data",
                          arrowstyle=style, shrinkA=5, shrinkB=5,
                          mutation_scale=20, fc=color, ec=color, linewidth=linewidth)
    ax.add_artist(arrow)

# Title
ax.text(8, 11.5, 'IAAIR Hybrid Search System Flow', 
        ha='center', va='center', fontsize=16, weight='bold')

# 1. Input Layer
create_box(ax, (1, 10), 3, 0.8, 'User Query\n"machine learning"', colors['input'], 10)
create_box(ax, (5, 10), 3, 0.8, 'HybridSearchRequest\n- routing_strategy\n- fusion_weights\n- top_k', colors['input'], 8)

# 2. Query Classification & Routing
create_box(ax, (9, 9.5), 3.5, 1, 'QueryClassifier\n- Analyze query type\n- Determine complexity\n- Route strategy', colors['routing'], 9)

# 3. Routing Decision Engine
create_box(ax, (13, 9.5), 2.5, 1, 'RoutingDecision\nEngine\n- VECTOR_FIRST\n- GRAPH_FIRST\n- PARALLEL\n- ADAPTIVE', colors['routing'], 8)

# 4. Search Engines (Parallel Branch)
# Vector Search
create_box(ax, (2, 7.5), 3, 1.2, 'Vector Search\n(Zilliz)\n- Dense embeddings\n- Sparse embeddings\n- Similarity scoring', colors['vector'], 8)

# Graph Search  
create_box(ax, (11, 7.5), 3, 1.2, 'Graph Search\n(Neo4j)\n- Cypher queries\n- Relationship traversal\n- Structural scoring', colors['graph'], 8)

# 5. Result Processing
create_box(ax, (2, 5.5), 3, 1, 'Vector Results\n- similarity_score\n- paper metadata\n- embeddings', colors['vector'], 8)

create_box(ax, (11, 5.5), 3, 1, 'Graph Results\n- relevance_score\n- relationship data\n- structural context', colors['graph'], 8)

# 6. Fusion Engine
create_box(ax, (6.5, 4), 3.5, 1.5, 'ResultFusion Engine\n- Score normalization\n- Weighted combination\n- Duplicate merging\n- Ranking', colors['fusion'], 9)

# 7. Scientific Reranking (Optional)
create_box(ax, (1, 2), 4, 1, 'Scientific Reranker\n- Citation impact\n- Venue prestige\n- Recency score\n- Author authority', colors['fusion'], 8)

# 8. Attribution System (Optional)  
create_box(ax, (11, 2), 4, 1, 'Attribution Tracker\n- Source provenance\n- Supporting passages\n- Confidence scores\n- Evidence bundles', colors['attribution'], 8)

# 9. Final Results
create_box(ax, (6, 0.2), 4, 1, 'HybridSearchResponse\n- Ranked results\n- Fusion statistics\n- Attribution data\n- Performance metrics', colors['output'], 9)

# Draw arrows for main flow
create_arrow(ax, (2.5, 10), (2.5, 9.5))  # Query to routing
create_arrow(ax, (6.5, 10.4), (9, 10))   # Request to classifier
create_arrow(ax, (12.5, 10), (13, 9.5))  # Classifier to decision engine

# Routing to search engines
create_arrow(ax, (10.5, 9.5), (3.5, 8.7))   # To vector search
create_arrow(ax, (12, 9.5), (12.5, 8.7))    # To graph search

# Search results to processing
create_arrow(ax, (3.5, 7.5), (3.5, 6.5))    # Vector results
create_arrow(ax, (12.5, 7.5), (12.5, 6.5))  # Graph results

# Results to fusion
create_arrow(ax, (5, 6), (6.5, 5))          # Vector to fusion
create_arrow(ax, (11, 6), (9.5, 5))         # Graph to fusion

# Fusion to optional components
create_arrow(ax, (7, 4), (3, 3))            # To reranker
create_arrow(ax, (9, 4), (13, 3))           # To attribution

# Final assembly
create_arrow(ax, (3, 2), (6.5, 1.2))        # Reranker to output
create_arrow(ax, (13, 2), (9.5, 1.2))       # Attribution to output
create_arrow(ax, (8.25, 4), (8, 1.2))       # Direct fusion to output

# Add decision points
ax.text(7, 8.5, 'Routing\nStrategy', ha='center', va='center', 
        fontsize=8, style='italic', bbox=dict(boxstyle="round", facecolor='white'))

ax.text(8.25, 3, 'Optional\nProcessing', ha='center', va='center',
        fontsize=8, style='italic', bbox=dict(boxstyle="round", facecolor='white'))

# Add timing annotations
ax.text(0.5, 7, 'Search\nTime:\n~7.5s', ha='center', va='center',
        fontsize=7, bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.7))

ax.text(15.5, 7, 'Fusion\nTime:\n~7.5s', ha='center', va='center', 
        fontsize=7, bbox=dict(boxstyle="round", facecolor='lightgreen', alpha=0.7))

# Legend
legend_elements = [
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['input'], label='Input Layer'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['routing'], label='Routing & Classification'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['vector'], label='Vector Search'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['graph'], label='Graph Search'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['fusion'], label='Fusion & Reranking'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['attribution'], label='Attribution'),
    mpatches.Rectangle((0, 0), 1, 1, facecolor=colors['output'], label='Output')
]

ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.85))

plt.tight_layout()
plt.savefig('/home/dnhoa/IAAIR/IAAIR/hybrid_search_flow.png', dpi=300, bbox_inches='tight')
plt.show()

# Create a second diagram showing the data structures
fig2, ax2 = plt.subplots(1, 1, figsize=(14, 10))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 10)
ax2.axis('off')

ax2.text(7, 9.5, 'Hybrid Search Data Flow & Structures', 
         ha='center', va='center', fontsize=16, weight='bold')

# Input structure
create_box(ax2, (1, 8), 4, 1.2, 'HybridSearchRequest\n{\n  "query": "machine learning",\n  "routing_strategy": "parallel",\n  "fusion_weights": {...},\n  "top_k": 5\n}', colors['input'], 8)

# Vector search data
create_box(ax2, (0.5, 6), 3, 1.5, 'Vector Results\n{\n  "paper_id": "W123",\n  "similarity_score": 1.2,\n  "title": "...",\n  "abstract": "...",\n  "embeddings": [...]\n}', colors['vector'], 7)

# Graph search data
create_box(ax2, (10.5, 6), 3, 1.5, 'Graph Results\n{\n  "paper_id": "W123",\n  "relevance_score": 0.8,\n  "relationships": [...],\n  "cypher_context": "..."\n}', colors['graph'], 7)

# Fusion processing
create_box(ax2, (5, 4.5), 4, 2, 'Fusion Processing\n{\n  "raw_relevance_score": 0.96,\n  "normalized_score": 0.96,\n  "vector_weight": 0.6,\n  "graph_weight": 0.4,\n  "final_score": 0.6\n}', colors['fusion'], 7)

# Final output
create_box(ax2, (4, 1.5), 6, 2, 'SearchResult\n{\n  "paper_id": "W123",\n  "relevance_score": 0.6,\n  "vector_score": 1.0,\n  "graph_score": 0.0,\n  "attributions": [...],\n  "confidence_scores": {...}\n}', colors['output'], 8)

# Arrows for data flow
create_arrow(ax2, (3, 8), (2, 7.5))         # Input to vector
create_arrow(ax2, (4.5, 8.5), (10.5, 7.5)) # Input to graph
create_arrow(ax2, (3.5, 6), (6, 6.5))       # Vector to fusion
create_arrow(ax2, (10.5, 6.8), (8.5, 6.5)) # Graph to fusion
create_arrow(ax2, (7, 4.5), (7, 3.5))       # Fusion to output

# Add score normalization note
ax2.text(7, 0.5, 'Score Normalization: All scores capped at 1.0 to pass validation\nFusion Formula: (vector_weight × vector_score) + (graph_weight × graph_score)', 
         ha='center', va='center', fontsize=9, style='italic',
         bbox=dict(boxstyle="round", facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/dnhoa/IAAIR/IAAIR/hybrid_search_data_flow.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Flow diagrams created successfully!")
print("📁 Files saved:")
print("   - hybrid_search_flow.png")
print("   - hybrid_search_data_flow.png")