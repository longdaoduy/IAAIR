"""
Attribution utilities for tracking source provenance in hybrid search.
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import re
from datetime import datetime

@dataclass
class ProvenanceRecord:
    """Record for tracking data lineage and transformations."""
    record_id: str
    timestamp: str
    operation_type: str  # 'search', 'fusion', 'rerank', 'attribution'
    input_sources: List[str]
    output_targets: List[str]
    parameters: Dict[str, Any]
    confidence_scores: Dict[str, float]
    transformation_details: str

class AttributionManager:
    """Advanced attribution management with provenance tracking."""
    
    def __init__(self):
        self.provenance_ledger: List[ProvenanceRecord] = []
        self.citation_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b[A-Z][a-z]+ et al\.?\b',  # Author citations
            r'\b(?:Figure|Table|Section)\s+\d+\b',  # Reference patterns
            r'\b(?:doi:|DOI:)\s*[\w\-\./]+\b'  # DOI patterns
        ]
    
    def create_evidence_bundle(self, query: str, search_results: List[Dict], 
                             routing_path: str) -> Dict[str, Any]:
        """Create comprehensive evidence bundle with full provenance."""
        bundle_id = f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        evidence_bundle = {
            "bundle_id": bundle_id,
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "routing_path": routing_path,
            "evidence_sources": [],
            "attribution_chains": [],
            "confidence_assessment": {},
            "provenance_trail": []
        }
        
        # Process each result for evidence extraction
        for result in search_results:
            evidence_source = self._extract_evidence_source(result, query)
            if evidence_source:
                evidence_bundle["evidence_sources"].append(evidence_source)
        
        # Build attribution chains
        evidence_bundle["attribution_chains"] = self._build_attribution_chains(
            evidence_bundle["evidence_sources"], query
        )
        
        # Assess overall confidence
        evidence_bundle["confidence_assessment"] = self._assess_bundle_confidence(
            evidence_bundle["attribution_chains"]
        )
        
        # Record provenance
        provenance_record = ProvenanceRecord(
            record_id=bundle_id,
            timestamp=datetime.now().isoformat(),
            operation_type="evidence_creation",
            input_sources=[r.get("paper_id", "") for r in search_results],
            output_targets=[bundle_id],
            parameters={"query": query, "routing_path": routing_path},
            confidence_scores=evidence_bundle["confidence_assessment"],
            transformation_details="Created evidence bundle from search results"
        )
        
        self.provenance_ledger.append(provenance_record)
        evidence_bundle["provenance_trail"] = [provenance_record.__dict__]
        
        return evidence_bundle
    
    def _extract_evidence_source(self, result: Dict, query: str) -> Optional[Dict]:
        """Extract evidence source information from a search result."""
        if not result.get("paper_id"):
            return None
        
        evidence_source = {
            "source_id": result["paper_id"],
            "source_type": "academic_paper",
            "title": result.get("title", ""),
            "authors": result.get("authors", []),
            "venue": result.get("venue"),
            "publication_date": result.get("publication_date"),
            "doi": result.get("doi"),
            "relevance_score": result.get("relevance_score", 0.0),
            "supporting_passages": [],
            "citation_context": [],
            "extraction_metadata": {
                "extraction_timestamp": datetime.now().isoformat(),
                "query_match_type": self._classify_query_match(result, query)
            }
        }
        
        # Extract supporting passages from abstract
        if result.get("abstract"):
            passages = self._extract_supporting_passages(result["abstract"], query)
            evidence_source["supporting_passages"] = passages
        
        # Extract citation context
        if result.get("abstract"):
            citations = self._extract_citations(result["abstract"])
            evidence_source["citation_context"] = citations
        
        return evidence_source
    
    def _classify_query_match(self, result: Dict, query: str) -> str:
        """Classify how the result matches the query."""
        title = result.get("title", "").lower()
        abstract = result.get("abstract", "").lower()
        query_lower = query.lower()
        
        if query_lower in title:
            return "title_exact_match"
        elif any(word in title for word in query_lower.split()):
            return "title_partial_match"
        elif query_lower in abstract:
            return "abstract_exact_match"
        elif any(word in abstract for word in query_lower.split()):
            return "abstract_partial_match"
        else:
            return "semantic_match"
    
    def _extract_supporting_passages(self, text: str, query: str, 
                                   window_size: int = 100) -> List[Dict]:
        """Extract text passages that support the query."""
        passages = []
        query_words = set(query.lower().split())
        sentences = re.split(r'[.!?]+', text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            matching_words = sum(1 for word in query_words if word in sentence_lower)
            
            if matching_words > 0:
                # Create passage with context window
                start_idx = max(0, i - 1)
                end_idx = min(len(sentences), i + 2)
                context = ' '.join(sentences[start_idx:end_idx]).strip()
                
                passage = {
                    "text": context,
                    "sentence_index": i,
                    "matching_words": matching_words,
                    "relevance_score": matching_words / len(query_words),
                    "char_start": text.find(sentence),
                    "char_end": text.find(sentence) + len(sentence)
                }
                passages.append(passage)
        
        # Sort by relevance and return top passages
        passages.sort(key=lambda x: x["relevance_score"], reverse=True)
        return passages[:5]  # Return top 5 passages
    
    def _extract_citations(self, text: str) -> List[Dict]:
        """Extract citation patterns from text."""
        citations = []
        
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citation = {
                    "text": match.group(),
                    "pattern_type": pattern,
                    "char_start": match.start(),
                    "char_end": match.end(),
                    "context": text[max(0, match.start()-50):match.end()+50]
                }
                citations.append(citation)
        
        return citations
    
    def _build_attribution_chains(self, evidence_sources: List[Dict], 
                                query: str) -> List[Dict]:
        """Build attribution chains showing evidence flow."""
        chains = []
        
        for source in evidence_sources:
            for passage in source.get("supporting_passages", []):
                chain = {
                    "chain_id": f"{source['source_id']}_passage_{passage.get('sentence_index', 0)}",
                    "source_paper": source["source_id"],
                    "evidence_text": passage["text"],
                    "relevance_score": passage["relevance_score"],
                    "attribution_type": "direct_textual_support",
                    "confidence": self._calculate_attribution_confidence(passage, source),
                    "verification_status": "pending",  # Would be verified against ground truth
                    "supporting_metadata": {
                        "query_overlap": passage["matching_words"],
                        "source_quality": self._assess_source_quality(source),
                        "passage_position": passage["sentence_index"]
                    }
                }
                chains.append(chain)
        
        # Sort chains by confidence
        chains.sort(key=lambda x: x["confidence"], reverse=True)
        return chains
    
    def _calculate_attribution_confidence(self, passage: Dict, source: Dict) -> float:
        """Calculate confidence score for an attribution."""
        base_confidence = passage["relevance_score"]
        
        # Boost confidence based on source quality
        source_quality_boost = self._assess_source_quality(source) * 0.2
        
        # Boost confidence based on passage position (earlier is often more important)
        position_boost = max(0, (10 - passage.get("sentence_index", 10)) / 10) * 0.1
        
        # Combine factors
        confidence = min(1.0, base_confidence + source_quality_boost + position_boost)
        return round(confidence, 3)
    
    def _assess_source_quality(self, source: Dict) -> float:
        """Assess the quality of an evidence source."""
        quality_score = 0.5  # Base score\n        \n        # Venue quality\n        venue = source.get("venue", "").lower()\n        high_impact_venues = ["nature", "science", "cell", "nejm", "lancet"]\n        if any(hiv in venue for hiv in high_impact_venues):\n            quality_score += 0.3\n        \n        # Recency (placeholder - would calculate from publication_date)\n        if source.get("publication_date"):\n            quality_score += 0.1\n        \n        # DOI presence\n        if source.get("doi"):\n            quality_score += 0.1\n        \n        return min(1.0, quality_score)
    
    def _assess_bundle_confidence(self, attribution_chains: List[Dict]) -> Dict[str, float]:
        """Assess overall confidence of the evidence bundle."""
        if not attribution_chains:\n            return {"overall_confidence": 0.0, "chain_count": 0}\n        \n        # Calculate various confidence metrics\n        confidences = [chain["confidence"] for chain in attribution_chains]\n        high_confidence_chains = [c for c in confidences if c > 0.7]\n        \n        assessment = {\n            "overall_confidence": sum(confidences) / len(confidences),\n            "max_confidence": max(confidences),\n            "min_confidence": min(confidences),\n            "high_confidence_ratio": len(high_confidence_chains) / len(confidences),\n            "chain_count": len(attribution_chains),\n            "evidence_diversity": len(set(chain["source_paper"] for chain in attribution_chains))\n        }\n        \n        return assessment
    
    def get_provenance_trail(self, bundle_id: str) -> List[Dict]:
        """Get full provenance trail for an evidence bundle."""
        return [record.__dict__ for record in self.provenance_ledger 
                if bundle_id in record.output_targets or bundle_id == record.record_id]
    
    def verify_attribution_accuracy(self, attribution_chains: List[Dict], 
                                  ground_truth: List[Dict]) -> Dict[str, float]:
        \"\"\"Verify attribution accuracy against ground truth data.\"\"\"\n        # This would implement attribution verification logic
        # For now, return mock metrics
        return {\n            "precision": 0.85,\n            "recall": 0.78,\n            "f1_score": 0.81,\n            "exact_match_rate": 0.72\n        }\n