"""
Biomedical Knowledge Integrations for Knowledge Fabric

This module provides integrations with major biomedical knowledge sources including:
- SNOMED CT: Medical terminology via BioPortal
- PrimeKG: Precision medicine knowledge graph_store
- Gene Ontology: Biological processes, molecular functions
- Human Phenotype Ontology: Phenotypic abnormalities
"""

from .snomed_ct import (
    SnomedCTIntegrator,
    SnomedConcept,
    AnnotationResult,
    initialize_snomed_integration,
    batch_enhance_documents_with_snomed
)

from .primekg import (
    PrimeKGIntegrator,
    download_primekg_data,
    load_primekg_for_knowledge_fabric,
    analyze_primekg_coverage
)

__all__ = [
    # SNOMED CT integration
    'SnomedCTIntegrator',
    'SnomedConcept', 
    'AnnotationResult',
    'initialize_snomed_integration',
    'batch_enhance_documents_with_snomed',
    
    # PrimeKG integration
    'PrimeKGIntegrator',
    'download_primekg_data',
    'load_primekg_for_knowledge_fabric',
    'analyze_primekg_coverage'
]