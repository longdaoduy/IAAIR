"""
SNOMED CT Integration Module for Knowledge Fabric

This module provides integration with SNOMED Clinical Terms via BioPortal API,
enabling medical knowledge extraction and annotation capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
from urllib.parse import quote, urlencode
from dataclasses import dataclass, field
from datetime import datetime

from ..core import KnowledgeFabric
from models.schemas.schemas import Document
from ..utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


@dataclass
class SnomedConcept:
    """SNOMED CT concept representation"""
    concept_id: str
    pref_label: str
    definition: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    semantic_types: List[str] = field(default_factory=list)
    cui: Optional[str] = None
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    ontology_source: str = "SNOMEDCT"


@dataclass
class AnnotationResult:
    """SNOMED CT annotation result"""
    text: str
    start_pos: int
    end_pos: int
    concept: SnomedConcept
    context: Optional[str] = None
    confidence: float = 1.0


class SnomedCTIntegrator:
    """
    SNOMED CT integration via BioPortal API
    
    Provides medical terminology extraction, concept lookup, and semantic annotation
    capabilities for biomedical documents in the Knowledge Fabric.
    """
    
    BASE_URL = "https://data.bioontology.org"
    ONTOLOGY_ID = "SNOMEDCT"
    
    def __init__(
        self,
        api_key: str,
        rate_limit: int = 1000,  # requests per hour
        timeout: int = 30,
        cache_ttl: int = 3600  # 1 hour cache
    ):
        """
        Initialize SNOMED CT integrator
        
        Args:
            api_key: BioPortal API key
            rate_limit: Maximum requests per hour
            timeout: Request timeout in seconds
            cache_ttl: Cache time-to-live in seconds
        """
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.rate_limiter = RateLimiter(rate_limit, 3600)  # per hour
        
        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}  # key: (result, timestamp)
        self._cache_ttl = cache_ttl
        
        # Session will be created lazily
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers={
                    'Authorization': f'apikey token={self.api_key}',
                    'Accept': 'application/json',
                    'User-Agent': 'KnowledgeFabric-SNOMED/1.0'
                }
            )
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if datetime.now().timestamp() - timestamp < self._cache_ttl:
                return result
            else:
                del self._cache[key]
        return None
    
    def _set_cache(self, key: str, value: Any):
        """Set cache value with timestamp"""
        self._cache[key] = (value, datetime.now().timestamp())
    
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited API request to BioPortal"""
        await self.rate_limiter.acquire()
        
        # Add API key and default parameters
        params = {
            'apikey': self.api_key,
            'display_context': 'false',  # Reduce response size
            'display_links': 'false',    # Reduce response size
            **params
        }
        
        url = f"{self.BASE_URL}{endpoint}"
        cache_key = f"{endpoint}:{urlencode(sorted(params.items()))}"
        
        # Check cache
        cached = self._get_cached(cache_key)
        if cached:
            logger.debug(f"Cache hit for {endpoint}")
            return cached
        
        session = await self._get_session()
        
        try:
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                result = await response.json()
                
                # Cache successful responses
                self._set_cache(cache_key, result)
                return result
                
        except aiohttp.ClientError as e:
            logger.error(f"BioPortal API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in BioPortal request: {e}")
            raise
    
    async def search_concepts(
        self,
        query: str,
        exact_match: bool = False,
        include_obsolete: bool = False,
        page_size: int = 50,
        max_results: int = 100
    ) -> List[SnomedConcept]:
        """
        Search SNOMED CT concepts by text query
        
        Args:
            query: Search query text
            exact_match: Require exact match
            include_obsolete: Include obsolete terms
            page_size: Results per page
            max_results: Maximum total results
            
        Returns:
            List of matching SNOMED CT concepts
        """
        concepts = []
        page = 1
        
        while len(concepts) < max_results:
            params = {
                'q': query,
                'ontologies': self.ONTOLOGY_ID,
                'require_exact_match': str(exact_match).lower(),
                'also_search_obsolete': str(include_obsolete).lower(),
                'page': page,
                'pagesize': min(page_size, max_results - len(concepts)),
                'include': 'prefLabel,definition,synonym,semanticType,cui,parents'
            }
            
            try:
                response = await self._make_request('/search', params)
                
                if not response.get('collection'):
                    break
                
                for item in response['collection']:
                    concept = self._parse_concept(item)
                    concepts.append(concept)
                
                # Check if we have more pages
                if len(response['collection']) < page_size:
                    break
                    
                page += 1
                
            except Exception as e:
                logger.error(f"Error searching concepts: {e}")
                break
        
        return concepts[:max_results]
    
    async def get_concept_details(self, concept_id: str) -> Optional[SnomedConcept]:
        """
        Get detailed information for a specific SNOMED CT concept
        
        Args:
            concept_id: SNOMED CT concept identifier
            
        Returns:
            Detailed concept information or None if not found
        """
        # URL-encode the concept ID
        encoded_id = quote(concept_id, safe='')
        
        try:
            params = {
                'include': 'prefLabel,definition,synonym,semanticType,cui,parents,children,properties'
            }
            
            response = await self._make_request(
                f'/ontologies/{self.ONTOLOGY_ID}/classes/{encoded_id}',
                params
            )
            
            return self._parse_concept(response)
            
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.warning(f"SNOMED CT concept not found: {concept_id}")
                return None
            raise
        except Exception as e:
            logger.error(f"Error getting concept details: {e}")
            return None
    
    async def annotate_text(
        self,
        text: str,
        longest_only: bool = True,
        whole_words_only: bool = True,
        exclude_numbers: bool = True,
        minimum_match_length: int = 3
    ) -> List[AnnotationResult]:
        """
        Annotate text with SNOMED CT concepts
        
        Args:
            text: Text to annotate
            longest_only: Return only longest matches
            whole_words_only: Match whole words only
            exclude_numbers: Exclude number-only matches
            minimum_match_length: Minimum match length
            
        Returns:
            List of annotation results
        """
        params = {
            'text': text,
            'ontologies': self.ONTOLOGY_ID,
            'longest_only': str(longest_only).lower(),
            'whole_word_only': str(whole_words_only).lower(),
            'exclude_numbers': str(exclude_numbers).lower(),
            'minimum_match_length': minimum_match_length,
            'include': 'prefLabel,definition,synonym,semanticType,cui'
        }
        
        try:
            response = await self._make_request('/annotator', params)
            annotations = []
            
            for annotation in response.get('collection', []):
                # Parse annotation structure
                concept_data = annotation['annotatedClass']
                concept = self._parse_concept(concept_data)
                
                # Extract annotation details
                for annotation_detail in annotation['annotations']:
                    result = AnnotationResult(
                        text=annotation_detail['text'],
                        start_pos=annotation_detail['from'] - 1,  # Convert to 0-based
                        end_pos=annotation_detail['to'],
                        concept=concept,
                        context=text[
                            max(0, annotation_detail['from'] - 50):
                            min(len(text), annotation_detail['to'] + 50)
                        ]
                    )
                    annotations.append(result)
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error annotating text: {e}")
            return []
    
    async def get_concept_hierarchy(
        self,
        concept_id: str,
        direction: str = "children",
        max_depth: int = 2
    ) -> Dict[str, List[SnomedConcept]]:
        """
        Get hierarchical relationships for a concept
        
        Args:
            concept_id: SNOMED CT concept identifier
            direction: "children", "parents", "ancestors", or "descendants"
            max_depth: Maximum hierarchy depth
            
        Returns:
            Dictionary mapping relationship types to concept lists
        """
        encoded_id = quote(concept_id, safe='')
        
        try:
            params = {
                'include': 'prefLabel,definition,synonym'
            }
            
            response = await self._make_request(
                f'/ontologies/{self.ONTOLOGY_ID}/classes/{encoded_id}/{direction}',
                params
            )
            
            concepts = []
            collection = response.get('collection', [])
            if isinstance(collection, list):
                for item in collection:
                    concept = self._parse_concept(item)
                    concepts.append(concept)
            
            return {direction: concepts}
            
        except Exception as e:
            logger.error(f"Error getting concept hierarchy: {e}")
            return {direction: []}
    
    def _parse_concept(self, data: Dict[str, Any]) -> SnomedConcept:
        """Parse BioPortal concept data into SnomedConcept"""
        # Extract basic information
        concept_id = data.get('@id', '')
        pref_label = data.get('prefLabel', '')
        
        # Extract definition (can be string or list)
        definition = None
        if 'definition' in data:
            def_data = data['definition']
            if isinstance(def_data, list) and def_data:
                definition = def_data[0]
            elif isinstance(def_data, str):
                definition = def_data
        
        # Extract synonyms
        synonyms = []
        if 'synonym' in data:
            syn_data = data['synonym']
            if isinstance(syn_data, list):
                synonyms = syn_data
            elif isinstance(syn_data, str):
                synonyms = [syn_data]
        
        # Extract semantic types
        semantic_types = []
        if 'semanticType' in data:
            sem_data = data['semanticType']
            if isinstance(sem_data, list):
                semantic_types = sem_data
            elif isinstance(sem_data, str):
                semantic_types = [sem_data]
        
        # Extract CUI
        cui = None
        if 'cui' in data:
            cui_data = data['cui']
            if isinstance(cui_data, list) and cui_data:
                cui = cui_data[0]
            elif isinstance(cui_data, str):
                cui = cui_data
        
        # Extract parent/child relationships
        parents = []
        if 'parents' in data:
            parent_data = data['parents']
            if isinstance(parent_data, list):
                parents = [p.get('@id', '') for p in parent_data if isinstance(p, dict)]
        
        children = []
        if 'children' in data:
            child_data = data['children']
            if isinstance(child_data, list):
                children = [c.get('@id', '') for c in child_data if isinstance(c, dict)]
        
        # Extract other properties
        properties = {}
        for key, value in data.items():
            if key not in ['@id', '@type', 'prefLabel', 'definition', 'synonym', 
                          'semanticType', 'cui', 'parents', 'children', 'links']:
                properties[key] = value
        
        return SnomedConcept(
            concept_id=concept_id,
            pref_label=pref_label,
            definition=definition,
            synonyms=synonyms,
            semantic_types=semantic_types,
            cui=cui,
            parents=parents,
            children=children,
            properties=properties
        )
    
    async def enhance_document_with_medical_concepts(
        self,
        document: Document,
        knowledge_fabric: KnowledgeFabric
    ) -> Document:
        """
        Enhance a document with SNOMED CT medical concept annotations
        
        Args:
            document: Paper to enhance
            knowledge_fabric: Knowledge Fabric instance
            
        Returns:
            Enhanced document with medical concepts
        """
        logger.info(f"Enhancing document {document.id} with SNOMED CT concepts")
        
        # Combine title and abstract for annotation
        text_to_annotate = f"{document.title}. {document.abstract or ''}"
        
        # Get SNOMED CT annotations
        annotations = await self.annotate_text(text_to_annotate)
        
        # Extract unique concepts
        unique_concepts = {}
        for annotation in annotations:
            concept = annotation.concept
            if concept.concept_id not in unique_concepts:
                unique_concepts[concept.concept_id] = concept
        
        # Add concepts to document metadata
        medical_concepts = []
        for concept in unique_concepts.values():
            concept_data = {
                'id': concept.concept_id,
                'label': concept.pref_label,
                'definition': concept.definition,
                'semantic_types': concept.semantic_types,
                'cui': concept.cui,
                'source': 'SNOMED CT'
            }
            medical_concepts.append(concept_data)
        
        # Update document metadata
        enhanced_metadata = document.metadata.copy()
        enhanced_metadata['medical_concepts'] = medical_concepts
        enhanced_metadata['snomed_annotations'] = len(annotations)
        enhanced_metadata['enhanced_with_snomed'] = datetime.utcnow().isoformat()
        
        # Create enhanced document
        enhanced_document = Document(
            id=document.id,
            title=document.title,
            authors=document.authors,
            abstract=document.abstract,
            content=document.content,
            doi=document.doi,
            url=document.url,
            publication_date=document.publication_date,
            venue=document.venue,
            citations=document.citations,
            embeddings=document.embeddings,
            metadata=enhanced_metadata,
            version=document.version + 1,
            created_at=document.created_at,
            updated_at=datetime.utcnow()
        )
        
        return enhanced_document


# Integration helper functions

async def initialize_snomed_integration(
    knowledge_fabric: KnowledgeFabric,
    bioportal_api_key: str
) -> SnomedCTIntegrator:
    """
    Initialize SNOMED CT integration with Knowledge Fabric
    
    Args:
        knowledge_fabric: Knowledge Fabric instance
        bioportal_api_key: BioPortal API key
        
    Returns:
        Configured SNOMED CT integrator
    """
    integrator = SnomedCTIntegrator(api_key=bioportal_api_key)
    
    logger.info("SNOMED CT integration initialized")
    return integrator


async def batch_enhance_documents_with_snomed(
    documents: List[Document],
    integrator: SnomedCTIntegrator,
    knowledge_fabric: KnowledgeFabric,
    batch_size: int = 10
) -> List[Document]:
    """
    Enhance multiple documents with SNOMED CT concepts in batches
    
    Args:
        documents: Documents to enhance
        integrator: SNOMED CT integrator
        knowledge_fabric: Knowledge Fabric instance
        batch_size: Batch size for processing
        
    Returns:
        Enhanced documents
    """
    enhanced_documents = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        logger.info(f"Processing SNOMED CT enhancement batch {i//batch_size + 1}")
        
        # Process batch concurrently
        tasks = [
            integrator.enhance_document_with_medical_concepts(doc, knowledge_fabric)
            for doc in batch
        ]
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in batch_results:
            if isinstance(result, Exception):
                logger.error(f"Error enhancing document: {result}")
            else:
                enhanced_documents.append(result)
        
        # Small delay between batches to be respectful to the API
        await asyncio.sleep(1)
    
    return enhanced_documents


# Example usage and testing functions

async def example_snomed_search():
    """Example of searching SNOMED CT concepts"""
    # Initialize with your BioPortal API key
    integrator = SnomedCTIntegrator(api_key="YOUR_API_KEY_HERE")
    
    try:
        # Search for diabetes-related concepts
        concepts = await integrator.search_concepts("diabetes", max_results=10)
        
        print(f"Found {len(concepts)} diabetes-related concepts:")
        for concept in concepts[:5]:
            print(f"- {concept.pref_label} ({concept.concept_id})")
            if concept.definition:
                print(f"  Definition: {concept.definition}")
            print()
        
        # Get detailed information for first concept
        if concepts:
            detailed = await integrator.get_concept_details(concepts[0].concept_id)
            if detailed:
                print(f"Detailed info for '{detailed.pref_label}':")
                print(f"  Synonyms: {detailed.synonyms}")
                print(f"  Semantic types: {detailed.semantic_types}")
        
        # Annotate medical text
        text = "Patient presents with type 2 diabetes mellitus and hypertension."
        annotations = await integrator.annotate_text(text)
        
        print(f"\nAnnotations for: '{text}'")
        for annotation in annotations:
            print(f"- '{annotation.text}' -> {annotation.concept.pref_label}")
    
    finally:
        await integrator.close()


if __name__ == "__main__":
    asyncio.run(example_snomed_search())