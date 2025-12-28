"""
Configuration for BioPortal and biomedical ontology integrations
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class BioPortalConfig:
    """Configuration for BioPortal API integration"""
    api_key: str
    base_url: str = "https://data.bioontology.org"
    rate_limit: int = 1000  # requests per hour
    timeout: int = 30
    cache_ttl: int = 3600  # 1 hour


# Supported ontologies in BioPortal
SUPPORTED_ONTOLOGIES = {
    'SNOMEDCT': {
        'name': 'SNOMED Clinical Terms',
        'description': 'Comprehensive clinical terminology',
        'domains': ['clinical', 'medical', 'healthcare'],
        'preferred': True
    },
    'GO': {
        'name': 'Gene Ontology',
        'description': 'Gene and gene product attributes',
        'domains': ['molecular_biology', 'genetics', 'biochemistry'],
        'preferred': True
    },
    'HP': {
        'name': 'Human Phenotype Ontology',
        'description': 'Human phenotypic abnormalities',
        'domains': ['genetics', 'phenotypes', 'diseases'],
        'preferred': True
    },
    'NCIT': {
        'name': 'NCI Thesaurus',
        'description': 'Cancer research terminology',
        'domains': ['oncology', 'cancer', 'biomarkers'],
        'preferred': True
    },
    'CHEBI': {
        'name': 'Chemical Entities of Biological Interest',
        'description': 'Chemical compounds and molecular entities',
        'domains': ['chemistry', 'biochemistry', 'drug_discovery'],
        'preferred': True
    },
    'UBERON': {
        'name': 'Uber-anatomy ontology',
        'description': 'Cross-species anatomy',
        'domains': ['anatomy', 'development', 'comparative_biology'],
        'preferred': False
    },
    'CL': {
        'name': 'Cell Ontology',
        'description': 'Cell types from all organisms',
        'domains': ['cell_biology', 'immunology', 'development'],
        'preferred': False
    },
    'DOID': {
        'name': 'Disease Ontology',
        'description': 'Human disease terminology',
        'domains': ['diseases', 'pathology', 'clinical'],
        'preferred': True
    },
    'FMA': {
        'name': 'Foundational Model of Anatomy',
        'description': 'Human anatomy reference',
        'domains': ['anatomy', 'medical_imaging', 'clinical'],
        'preferred': False
    },
    'MESH': {
        'name': 'Medical Subject Headings',
        'description': 'NLM controlled vocabulary',
        'domains': ['literature', 'indexing', 'search'],
        'preferred': True
    }
}

# Domain-specific ontology recommendations
DOMAIN_ONTOLOGY_MAP = {
    'clinical_medicine': ['SNOMEDCT', 'MESH', 'NCIT', 'DOID'],
    'molecular_biology': ['GO', 'CHEBI', 'CL'],
    'genetics': ['HP', 'GO', 'DOID'],
    'pharmacology': ['CHEBI', 'SNOMEDCT', 'NCIT'],
    'anatomy': ['UBERON', 'FMA', 'SNOMEDCT'],
    'cancer_research': ['NCIT', 'CHEBI', 'SNOMEDCT'],
    'immunology': ['CL', 'GO', 'SNOMEDCT'],
    'neuroscience': ['SNOMEDCT', 'UBERON', 'HP'],
    'pathology': ['DOID', 'SNOMEDCT', 'HP'],
    'biochemistry': ['CHEBI', 'GO', 'SNOMEDCT']
}

# Semantic type mappings for concept classification
SEMANTIC_TYPE_CATEGORIES = {
    'anatomical': [
        'T017',  # Anatomical Structure
        'T023',  # Body Part, Organ, or Organ Component
        'T024',  # Tissue
        'T025',  # Cell
        'T026',  # Cell Component
        'T029',  # Body Location or Region
        'T030',  # Body Space or Junction
    ],
    'clinical': [
        'T033',  # Finding
        'T034',  # Laboratory or Test Result
        'T037',  # Injury or Poisoning
        'T046',  # Pathologic Function
        'T047',  # Disease or Syndrome
        'T048',  # Mental or Behavioral Dysfunction
        'T049',  # Cell or Molecular Dysfunction
        'T050',  # Experimental Model of Disease
        'T184',  # Sign or Symptom
        'T190',  # Anatomical Abnormality
        'T191',  # Neoplastic Process
    ],
    'therapeutic': [
        'T061',  # Therapeutic or Preventive Procedure
        'T074',  # Medical Device
        'T121',  # Pharmacologic Substance
        'T122',  # Biomedical or Dental Material
        'T123',  # Biologically Active Substance
        'T125',  # Hormone
        'T126',  # Enzyme
        'T127',  # Vitamin
        'T129',  # Immunologic Factor
        'T131',  # Hazardous or Poisonous Substance
        'T200',  # Clinical Drug
    ],
    'biological': [
        'T001',  # Organism
        'T002',  # Plant
        'T004',  # Fungus
        'T005',  # Virus
        'T007',  # Bacterium
        'T011',  # Amphibian
        'T012',  # Bird
        'T013',  # Fish
        'T014',  # Reptile
        'T015',  # Mammal
        'T016',  # Human
        'T116',  # Amino Acid, Peptide, or Protein
        'T192',  # Receptor
        'T194',  # Archaeon
    ]
}


def get_ontologies_for_domain(domain: str) -> List[str]:
    """Get recommended ontologies for a research domain"""
    return DOMAIN_ONTOLOGY_MAP.get(domain, ['SNOMEDCT', 'MESH'])


def get_semantic_category(semantic_types: List[str]) -> Optional[str]:
    """Determine semantic category from UMLS semantic types"""
    for category, types in SEMANTIC_TYPE_CATEGORIES.items():
        if any(st in types for st in semantic_types):
            return category
    return None


def is_preferred_ontology(ontology_id: str) -> bool:
    """Check if ontology is marked as preferred"""
    return SUPPORTED_ONTOLOGIES.get(ontology_id, {}).get('preferred', False)