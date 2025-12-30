# SNOMED CT Integration for Knowledge Fabric

This module provides comprehensive integration with SNOMED Clinical Terms (SNOMED CT) via the BioPortal API, enabling medical knowledge extraction and semantic annotation capabilities for biomedical documents.

## Overview

SNOMED CT is one of the world's most comprehensive medical terminology systems, containing:
- Over 350,000 active medical concepts
- Hierarchical relationships between concepts  
- Multiple descriptions and synonyms for each concept
- Semantic types for computational reasoning
- Cross-references to other medical vocabularies (UMLS CUI)

## Features

### 1. Concept Search
Search for SNOMED CT concepts by text query with options for:
- Exact vs fuzzy matching
- Including/excluding obsolete terms
- Pagination support
- Detailed concept information

### 2. Text Annotation
Automatically identify medical concepts in free text:
- Named entity recognition for medical terms
- Position information (start/end offsets)
- Context extraction
- Confidence scoring

### 3. Concept Hierarchy Navigation
Explore relationships between medical concepts:
- Parent-child relationships
- Ancestor and descendant traversal
- Semantic relationship types
- Concept properties and attributes

### 4. Document Enhancement
Enhance biomedical documents with medical concept metadata:
- Automatic concept extraction from titles and abstracts
- Semantic type categorization
- Concept definition lookup
- Integration with knowledge graph

## Quick Start

### 1. Get BioPortal API Key

First, register for a free BioPortal account and get your API key:
1. Go to https://bioportal.bioontology.org/
2. Create an account
3. Navigate to "Account" → "API Key"
4. Copy your API key

### 2. Basic Usage

```python
import asyncio
from knowledge_fabric.integrations.snomed_ct import SnomedCTIntegrator

async def main():
    # Initialize integrator
    integrator = SnomedCTIntegrator(api_key="YOUR_API_KEY_HERE")
    
    try:
        # Search for diabetes concepts
        concepts = await integrator.search_concepts("diabetes", max_results=5)
        for concept in concepts:
            print(f"{concept.pref_label} - {concept.definition}")
        
        # Annotate medical text
        text = "Patient presents with type 2 diabetes and hypertension."
        annotations = await integrator.annotate_text(text)
        for ann in annotations:
            print(f"'{ann.text}' -> {ann.concept.pref_label}")
    
    finally:
        await integrator.close()

asyncio.run(main())
```

### 3. Knowledge Fabric Integration

```python
from knowledge_fabric.core import KnowledgeFabric
from models.schemas.schemas import Document, Author

# Configure Knowledge Fabric with BioPortal API key
config = Settings(bioportal_api_key="YOUR_API_KEY_HERE")
kf = KnowledgeFabric(config=config)

# Create sample document
document = Document(
    id="sample_doc",
    title="Diabetes Management in Clinical Practice",
    abstract="This study examines treatment protocols for type 2 diabetes...",
    authors=[Author(name="Dr. Smith")]
)

# Enhance with medical concepts
enhanced_docs = await kf.enhance_with_medical_concepts([document])
medical_concepts = enhanced_docs[0].metadata['medical_concepts']

print(f"Found {len(medical_concepts)} medical concepts:")
for concept in medical_concepts:
    print(f"- {concept['label']} ({concept['id']})")
```

## Configuration

### Environment Variables

```bash
# Required
BIOPORTAL_API_KEY=your_api_key_here

# Optional
BIOPORTAL_BASE_URL=https://data.bioontology.org
BIOPORTAL_RATE_LIMIT=1000  # requests per hour
BIOPORTAL_TIMEOUT=30       # seconds
BIOPORTAL_CACHE_TTL=3600   # seconds
```

### Programmatic Configuration

```python
from knowledge_fabric.integrations.snomed_ct import SnomedCTIntegrator

integrator = SnomedCTIntegrator(
    api_key="your_api_key",
    rate_limit=1000,  # requests per hour
    timeout=30,       # request timeout in seconds
    cache_ttl=3600    # cache time-to-live in seconds
)
```

## API Reference

### SnomedCTIntegrator

Main class for SNOMED CT integration.

#### Methods

##### `search_concepts(query, exact_match=False, include_obsolete=False, max_results=50)`

Search for SNOMED CT concepts.

**Parameters:**
- `query`: Search query text
- `exact_match`: Require exact match (default: False)
- `include_obsolete`: Include obsolete terms (default: False) 
- `max_results`: Maximum number of results (default: 50)

**Returns:** List of `SnomedConcept` objects

##### `annotate_text(text, longest_only=True, whole_words_only=True)`

Annotate text with SNOMED CT concepts.

**Parameters:**
- `text`: Text to annotate
- `longest_only`: Return only longest matches (default: True)
- `whole_words_only`: Match whole words only (default: True)

**Returns:** List of `AnnotationResult` objects

##### `get_concept_details(concept_id)`

Get detailed information for a specific concept.

**Parameters:**
- `concept_id`: SNOMED CT concept identifier

**Returns:** `SnomedConcept` object or None

##### `get_concept_hierarchy(concept_id, direction="children", max_depth=2)`

Get hierarchical relationships for a concept.

**Parameters:**
- `concept_id`: SNOMED CT concept identifier  
- `direction`: "children", "parents", "ancestors", or "descendants"
- `max_depth`: Maximum hierarchy depth (default: 2)

**Returns:** Dictionary mapping relationship types to concept lists

##### `enhance_document_with_medical_concepts(document, knowledge_fabric)`

Enhance a document with SNOMED CT concepts.

**Parameters:**
- `document`: Document to enhance
- `knowledge_fabric`: Knowledge Fabric instance

**Returns:** Enhanced `Document` with medical concepts in metadata

### Data Structures

#### SnomedConcept

Represents a SNOMED CT concept.

**Attributes:**
- `concept_id`: Unique identifier
- `pref_label`: Preferred label/name
- `definition`: Concept definition
- `synonyms`: List of alternative terms
- `semantic_types`: UMLS semantic types
- `cui`: UMLS Concept Unique Identifier
- `parents`: Parent concept IDs
- `children`: Child concept IDs
- `properties`: Additional properties

#### AnnotationResult

Represents a text annotation result.

**Attributes:**
- `text`: Matched text span
- `start_pos`: Start position (0-based)
- `end_pos`: End position
- `concept`: Associated `SnomedConcept`
- `context`: Surrounding text context
- `confidence`: Confidence score

## Examples

### Example 1: Medical Literature Analysis

```python
import asyncio
from knowledge_fabric.integrations.snomed_ct import SnomedCTIntegrator

async def analyze_medical_paper():
    integrator = SnomedCTIntegrator(api_key="YOUR_API_KEY")
    
    # Sample abstract from medical literature
    abstract = """
    This randomized controlled trial examined the efficacy of metformin
    in patients with type 2 diabetes mellitus. Primary outcomes included
    HbA1c reduction and cardiovascular events. Secondary outcomes assessed
    diabetic nephropathy progression and retinal complications.
    """
    
    # Extract medical concepts
    annotations = await integrator.annotate_text(abstract)
    
    print("Medical concepts found:")
    for ann in annotations:
        concept = ann.concept
        print(f"- {ann.text} → {concept.pref_label}")
        if concept.semantic_types:
            print(f"  Types: {', '.join(concept.semantic_types[:2])}")
        if concept.definition:
            print(f"  Definition: {concept.definition[:100]}...")
        print()
    
    await integrator.close()

asyncio.run(analyze_medical_paper())
```

### Example 2: Building Medical Knowledge Graph

```python
import asyncio
from knowledge_fabric.core import KnowledgeFabric
from models.schemas.schemas import Document, Author


async def build_medical_knowledge_graph():
    # Initialize Knowledge Fabric with SNOMED CT
    kf = KnowledgeFabric(config=Settings(bioportal_api_key="YOUR_API_KEY"))

    # Sample medical documents
    documents = [
        Document(
            id="doc1",
            title="Diabetes and Cardiovascular Disease",
            abstract="Study of diabetes complications...",
            authors=[Author(name="Dr. Jones")]
        ),
        Document(
            id="doc2",
            title="Hypertension Management Guidelines",
            abstract="Clinical guidelines for blood pressure control...",
            authors=[Author(name="Dr. Smith")]
        )
    ]

    # Enhance with medical concepts
    enhanced_docs = await kf.enhance_with_medical_concepts(documents)

    # Ingest into knowledge graph_store
    results = await kf.ingest_documents(enhanced_docs)
    print(f"Ingested {results['successful']} documents with medical concepts")

    # Search using medical terminology
    search_results = await kf.search("diabetes management hypertension")
    print(f"Found {len(search_results)} related documents")

    await kf.close()


asyncio.run(build_medical_knowledge_graph())
```

### Example 3: Medical Concept Exploration

```python
import asyncio
from knowledge_fabric.integrations.snomed_ct import SnomedCTIntegrator

async def explore_medical_concepts():
    integrator = SnomedCTIntegrator(api_key="YOUR_API_KEY")
    
    # Start with diabetes
    diabetes_concepts = await integrator.search_concepts("diabetes mellitus", exact_match=True)
    
    if diabetes_concepts:
        diabetes = diabetes_concepts[0]
        print(f"Concept: {diabetes.pref_label}")
        print(f"Definition: {diabetes.definition}")
        print()
        
        # Explore children (subtypes)
        children = await integrator.get_concept_hierarchy(diabetes.concept_id, "children")
        print("Diabetes subtypes:")
        for child in children.get("children", [])[:5]:
            print(f"- {child.pref_label}")
        print()
        
        # Explore parents (broader categories)
        parents = await integrator.get_concept_hierarchy(diabetes.concept_id, "parents")
        print("Broader categories:")
        for parent in parents.get("parents", []):
            print(f"- {parent.pref_label}")
    
    await integrator.close()

asyncio.run(explore_medical_concepts())
```

## Performance and Rate Limiting

### Rate Limiting
- Default: 1000 requests per hour
- Automatic rate limiting with exponential backoff
- Configurable rate limits per application needs

### Caching
- In-memory caching with TTL
- Reduces API calls for repeated queries
- Configurable cache duration (default: 1 hour)

### Optimization Tips
1. Use exact matching when possible for better cache hits
2. Batch document processing to minimize API calls
3. Implement persistent caching for production use
4. Monitor rate limit usage in logs

## Troubleshooting

### Common Issues

**1. API Key Issues**
```
Error: 401 Unauthorized
```
- Verify API key is correct
- Ensure API key is not expired
- Check BioPortal account status

**2. Rate Limit Exceeded**
```
Error: 429 Too Many Requests
```
- Reduce request rate
- Implement longer delays between requests
- Consider upgrading BioPortal account

**3. Concept Not Found**
```
Error: 404 Not Found
```
- Verify concept ID format
- Check if concept is obsolete
- Try alternative search terms

### Debugging

Enable detailed logging:

```python
import logging
logging.getLogger('knowledge_fabric.integrations.snomed_ct').setLevel(logging.DEBUG)
```

## Integration with Other Ontologies

While this module focuses on SNOMED CT, the architecture supports additional biomedical ontologies:

- **Gene Ontology (GO)**: Gene and protein functions
- **Human Phenotype Ontology (HPO)**: Phenotypic abnormalities
- **ChEBI**: Chemical entities
- **MESH**: Medical subject headings
- **NCIT**: Cancer terminology

See `bioportal_config.py` for supported ontologies and configuration.

## Contributing

To extend SNOMED CT integration:

1. **Add new methods**: Extend `SnomedCTIntegrator` class
2. **Support additional ontologies**: Create similar integrator classes
3. **Enhance search**: Improve query processing and ranking
4. **Add semantic reasoning**: Implement relationship inference

## References

- [SNOMED CT Documentation](https://www.snomed.org/snomed-ct)
- [BioPortal API Documentation](https://data.bioontology.org/documentation)
- [UMLS Semantic Types](https://www.nlm.nih.gov/research/umls/META3_current_semantic_types.html)
- [Knowledge Fabric Architecture](../../../README.md)