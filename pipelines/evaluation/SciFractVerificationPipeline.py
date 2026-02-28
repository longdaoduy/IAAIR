"""
SciFact verification pipeline integration for scientific claim verification.

This module provides claim verification capabilities to reduce hallucinations
and improve factual accuracy in scientific RAG systems.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from enum import Enum

from models.entities.retrieval.SearchResult import SearchResult

logger = logging.getLogger(__name__)

class VerificationLabel(str, Enum):
    """SciFact verification labels."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES" 
    NOT_ENOUGH_INFO = "NOT_ENOUGH_INFO"
    DISPUTED = "DISPUTED"

@dataclass
class ScientificClaim:
    """Scientific claim for verification."""
    claim_id: str
    claim_text: str
    domain: Optional[str] = None
    confidence: Optional[float] = None
    source_papers: Optional[List[str]] = None

@dataclass
class VerificationEvidence:
    """Evidence for claim verification."""
    paper_id: str
    title: str
    abstract: str
    relevant_sentences: List[str]
    evidence_score: float  # Strength of evidence (0-1)
    stance: VerificationLabel  # How this evidence relates to claim

@dataclass
class VerificationResult:
    """Result of claim verification."""
    claim: ScientificClaim
    final_label: VerificationLabel
    confidence: float
    evidence_pieces: List[VerificationEvidence]
    reasoning: str
    contradictory_evidence: Optional[List[VerificationEvidence]] = None

@dataclass
class VerificationBenchmark:
    """Benchmark for evaluation."""
    claim: ScientificClaim
    ground_truth_label: VerificationLabel
    ground_truth_evidence: List[str]  # Paper IDs that provide evidence


class SciFractVerificationPipeline:
    """SciFact-style scientific claim verification pipeline."""
    
    def __init__(self, retrieval_client=None, llm_client=None):
        self.retrieval_client = retrieval_client
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        
        # Evidence strength thresholds
        self.strong_evidence_threshold = 0.8
        self.moderate_evidence_threshold = 0.6
        self.weak_evidence_threshold = 0.4
    
    def extract_claims_from_text(self, text: str) -> List[ScientificClaim]:
        """
        Extract verifiable scientific claims from text.
        
        Args:
            text: Input text to extract claims from
            
        Returns:
            List of extracted scientific claims
        """
        # Identify sentences that make factual scientific claims
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        # Patterns that indicate scientific claims
        claim_patterns = [
            r'\b(increase|decrease|reduce|improve|enhance|inhibit|activate|suppress)\b',
            r'\b(study shows?|research demonstrates?|findings suggest)\b',
            r'\b(effective|ineffective|significant|correlation|association)\b',
            r'\b(causes?|leads? to|results? in|due to)\b',
            r'\b\d+%\b|\b\d+\.\d+\b|\bp\s*[<>=]\s*\d+\.\d+',  # Statistical claims
        ]
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains claim patterns
            has_claim_pattern = any(re.search(pattern, sentence, re.IGNORECASE) 
                                  for pattern in claim_patterns)
            
            if has_claim_pattern:
                claim = ScientificClaim(
                    claim_id=f"claim_{i+1:03d}",
                    claim_text=sentence,
                    confidence=0.7  # Default confidence for extracted claims
                )
                claims.append(claim)
        
        self.logger.info(f"Extracted {len(claims)} scientific claims from text")
        return claims
    
    def retrieve_evidence(self, claim: ScientificClaim, top_k: int = 10) -> List[SearchResult]:
        """
        Retrieve relevant papers as evidence for claim verification.
        
        Args:
            claim: Scientific claim to find evidence for
            top_k: Number of papers to retrieve
            
        Returns:
            List of relevant papers
        """
        if not self.retrieval_client:
            self.logger.warning("No retrieval client available for evidence retrieval")
            return []
        
        try:
            # Use claim text as query
            results = self.retrieval_client.search_similar_papers(
                query_text=claim.claim_text,
                top_k=top_k,
                use_hybrid=True
            )
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving evidence for claim {claim.claim_id}: {e}")
            return []
    
    def assess_evidence_strength(self, claim: ScientificClaim, 
                               evidence_paper: SearchResult) -> Tuple[float, VerificationLabel]:
        """
        Assess how strongly a paper supports/refutes a claim.
        
        Args:
            claim: The scientific claim
            evidence_paper: Paper to assess as evidence
            
        Returns:
            Tuple of (evidence_strength, stance)
        """
        # Simplified evidence assessment - in practice would use NLI models
        claim_text = claim.claim_text.lower()
        
        # Check abstract and title for relevant content
        paper_text = ""
        if evidence_paper.title:
            paper_text += evidence_paper.title.lower() + " "
        if evidence_paper.abstract:
            paper_text += evidence_paper.abstract.lower()
        
        # Look for supporting/refuting language patterns
        supporting_patterns = [
            r'\b(confirm|support|demonstrate|show|prove|validate|verify)\b',
            r'\b(positive|beneficial|effective|improvement|increase)\b',
            r'\b(significant|strong|clear|evident)\b'
        ]
        
        refuting_patterns = [
            r'\b(contradict|refute|disprove|oppose|challenge|dispute)\b',
            r'\b(negative|harmful|ineffective|no effect|decrease)\b',
            r'\b(not|lack|absent|insufficient|failed)\b'
        ]
        
        # Count matches
        support_score = sum(1 for pattern in supporting_patterns 
                           if re.search(pattern, paper_text))
        refute_score = sum(1 for pattern in refuting_patterns 
                          if re.search(pattern, paper_text))
        
        # Calculate relevance based on keyword overlap
        claim_words = set(re.findall(r'\b\w+\b', claim_text))
        paper_words = set(re.findall(r'\b\w+\b', paper_text))
        relevance = len(claim_words & paper_words) / len(claim_words | paper_words)
        
        # Determine stance and strength
        if support_score > refute_score and relevance > 0.3:
            stance = VerificationLabel.SUPPORTS
            strength = min(0.8, relevance + 0.1 * support_score)
        elif refute_score > support_score and relevance > 0.3:
            stance = VerificationLabel.REFUTES
            strength = min(0.8, relevance + 0.1 * refute_score)
        elif relevance > 0.2:
            stance = VerificationLabel.NOT_ENOUGH_INFO
            strength = relevance * 0.5
        else:
            stance = VerificationLabel.NOT_ENOUGH_INFO
            strength = 0.1
        
        return strength, stance
    
    def verify_claim(self, claim: ScientificClaim) -> VerificationResult:
        """
        Verify a scientific claim using retrieved evidence.
        
        Args:
            claim: Scientific claim to verify
            
        Returns:
            Verification result with evidence and reasoning
        """
        # Retrieve evidence papers
        evidence_papers = self.retrieve_evidence(claim)
        
        if not evidence_papers:
            return VerificationResult(
                claim=claim,
                final_label=VerificationLabel.NOT_ENOUGH_INFO,
                confidence=0.1,
                evidence_pieces=[],
                reasoning="No relevant evidence found in literature."
            )
        
        # Assess each piece of evidence
        evidence_pieces = []
        for paper in evidence_papers:
            strength, stance = self.assess_evidence_strength(claim, paper)
            
            if strength > self.weak_evidence_threshold:
                evidence = VerificationEvidence(
                    paper_id=paper.paper_id,
                    title=paper.title or "Unknown Title",
                    abstract=paper.abstract or "",
                    relevant_sentences=[],  # Would extract in full implementation
                    evidence_score=strength,
                    stance=stance
                )
                evidence_pieces.append(evidence)
        
        # Aggregate evidence to make final decision
        support_evidence = [e for e in evidence_pieces if e.stance == VerificationLabel.SUPPORTS]
        refute_evidence = [e for e in evidence_pieces if e.stance == VerificationLabel.REFUTES]
        
        support_strength = sum(e.evidence_score for e in support_evidence)
        refute_strength = sum(e.evidence_score for e in refute_evidence)
        
        # Make final decision
        if support_strength > refute_strength and support_strength > 1.0:
            final_label = VerificationLabel.SUPPORTS
            confidence = min(0.9, support_strength / (support_strength + refute_strength + 1))
            reasoning = f"Supported by {len(support_evidence)} pieces of evidence with total strength {support_strength:.2f}"
        elif refute_strength > support_strength and refute_strength > 0.8:
            final_label = VerificationLabel.REFUTES
            confidence = min(0.9, refute_strength / (support_strength + refute_strength + 1))
            reasoning = f"Refuted by {len(refute_evidence)} pieces of evidence with total strength {refute_strength:.2f}"
        elif support_strength > 0.5 and refute_strength > 0.5:
            final_label = VerificationLabel.DISPUTED
            confidence = 0.6
            reasoning = f"Conflicting evidence: {len(support_evidence)} supporting, {len(refute_evidence)} refuting"
        else:
            final_label = VerificationLabel.NOT_ENOUGH_INFO
            confidence = 0.3
            reasoning = f"Insufficient evidence strength (support: {support_strength:.2f}, refute: {refute_strength:.2f})"
        
        return VerificationResult(
            claim=claim,
            final_label=final_label,
            confidence=confidence,
            evidence_pieces=evidence_pieces,
            reasoning=reasoning,
            contradictory_evidence=refute_evidence if support_evidence and refute_evidence else None
        )
    
    def verify_text(self, text: str) -> List[VerificationResult]:
        """
        Verify all claims in a text passage.
        
        Args:
            text: Text containing scientific claims
            
        Returns:
            List of verification results for each claim
        """
        # Extract claims
        claims = self.extract_claims_from_text(text)
        
        # Verify each claim
        results = []
        for claim in claims:
            try:
                result = self.verify_claim(claim)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error verifying claim {claim.claim_id}: {e}")
                continue
        
        return results


class VerificationEvaluator:
    """Evaluate verification pipeline performance."""
    
    def evaluate_verification(self, predictions: List[VerificationResult],
                            benchmarks: List[VerificationBenchmark]) -> Dict[str, float]:
        """
        Evaluate verification pipeline against benchmarks.
        
        Args:
            predictions: Predicted verification results
            benchmarks: Ground truth benchmarks
            
        Returns:
            Evaluation metrics
        """
        if len(predictions) != len(benchmarks):
            self.logger.warning("Mismatch between predictions and benchmarks")
        
        correct = 0
        total = min(len(predictions), len(benchmarks))
        
        # Label-wise metrics
        label_metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0} for label in VerificationLabel}
        
        for pred, bench in zip(predictions[:total], benchmarks[:total]):
            if pred.final_label == bench.ground_truth_label:
                correct += 1
                label_metrics[pred.final_label]['tp'] += 1
            else:
                label_metrics[pred.final_label]['fp'] += 1
                label_metrics[bench.ground_truth_label]['fn'] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate precision, recall, F1 for each label
        metrics = {'accuracy': accuracy}
        
        for label in VerificationLabel:
            tp = label_metrics[label]['tp']
            fp = label_metrics[label]['fp']
            fn = label_metrics[label]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[f'{label.lower()}_precision'] = precision
            metrics[f'{label.lower()}_recall'] = recall
            metrics[f'{label.lower()}_f1'] = f1
        
        return metrics


def create_verification_benchmarks() -> List[VerificationBenchmark]:
    """Create default verification benchmarks."""
    
    benchmarks = [
        VerificationBenchmark(
            claim=ScientificClaim(
                claim_id="verify_001",
                claim_text="CRISPR-Cas9 can edit genes with 100% accuracy",
                domain="biomedical"
            ),
            ground_truth_label=VerificationLabel.REFUTES,
            ground_truth_evidence=["W2963920772"]
        ),
        VerificationBenchmark(
            claim=ScientificClaim(
                claim_id="verify_002", 
                claim_text="Machine learning improves drug discovery efficiency",
                domain="biomedical"
            ),
            ground_truth_label=VerificationLabel.SUPPORTS,
            ground_truth_evidence=["W2748394829", "W2951235503"]
        ),
        VerificationBenchmark(
            claim=ScientificClaim(
                claim_id="verify_003",
                claim_text="Transformers replaced all previous neural network architectures",
                domain="cs"
            ),
            ground_truth_label=VerificationLabel.REFUTES,
            ground_truth_evidence=["W2964069955"]
        )
    ]
    
    return benchmarks