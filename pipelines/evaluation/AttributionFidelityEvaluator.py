"""
Attribution fidelity measurement for scientific RAG system.

This module evaluates the accuracy and quality of source attribution,
measuring exact span matching and citation accuracy.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from difflib import SequenceMatcher

from models.entities.retrieval.SearchResult import SearchResult
from models.entities.retrieval.AttributionSpan import AttributionSpan

logger = logging.getLogger(__name__)

@dataclass
class AttributionGoldStandard:
    """Ground truth attribution for evaluation."""
    query_id: str
    query_text: str
    expected_attributions: List[Dict[str, any]]  # Expected attribution spans
    generated_text: str  # Text that should be attributed
    
@dataclass
class AttributionMetrics:
    """Attribution evaluation metrics."""
    exact_span_match_rate: float
    partial_span_match_rate: float
    citation_coverage: float
    wrong_source_rate: float
    attribution_precision: float
    attribution_recall: float
    average_confidence: float
    high_confidence_rate: float  # Percentage with confidence > 0.8


class AttributionFidelityEvaluator:
    """Evaluate attribution accuracy and fidelity."""
    
    def __init__(self, confidence_threshold: float = 0.8):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def exact_span_match(self, predicted_span: str, ground_truth_span: str, 
                         tolerance: int = 0) -> bool:
        """
        Check if predicted span exactly matches ground truth.
        
        Args:
            predicted_span: The predicted attribution span
            ground_truth_span: Expected attribution span
            tolerance: Number of characters tolerance for boundaries
            
        Returns:
            True if spans match within tolerance
        """
        # Normalize whitespace
        pred_normalized = re.sub(r'\s+', ' ', predicted_span.strip())
        gt_normalized = re.sub(r'\s+', ' ', ground_truth_span.strip())
        
        if tolerance == 0:
            return pred_normalized == gt_normalized
        
        # Check if spans are similar within tolerance
        similarity = SequenceMatcher(None, pred_normalized, gt_normalized).ratio()
        return similarity >= (1.0 - tolerance / max(len(pred_normalized), len(gt_normalized)))
    
    def partial_span_match(self, predicted_span: str, ground_truth_span: str,
                          min_overlap: float = 0.5) -> bool:
        """
        Check if predicted span partially matches ground truth.
        
        Args:
            predicted_span: The predicted attribution span
            ground_truth_span: Expected attribution span  
            min_overlap: Minimum overlap ratio required
            
        Returns:
            True if spans overlap sufficiently
        """
        # Normalize and tokenize
        pred_words = set(re.sub(r'\s+', ' ', predicted_span.strip().lower()).split())
        gt_words = set(re.sub(r'\s+', ' ', ground_truth_span.strip().lower()).split())
        
        if not pred_words or not gt_words:
            return False
            
        # Calculate Jaccard similarity
        intersection = len(pred_words & gt_words)
        union = len(pred_words | gt_words)
        
        overlap_ratio = intersection / union if union > 0 else 0
        return overlap_ratio >= min_overlap
    
    def citation_coverage(self, attributions: List[AttributionSpan], 
                         required_sources: Set[str]) -> float:
        """
        Calculate what percentage of required sources are cited.
        
        Args:
            attributions: List of attribution spans
            required_sources: Set of source IDs that should be cited
            
        Returns:
            Coverage ratio (0.0 to 1.0)
        """
        if not required_sources:
            return 1.0
            
        cited_sources = {attr.source_id for attr in attributions}
        covered = len(cited_sources & required_sources)
        
        return covered / len(required_sources)
    
    def wrong_source_rate(self, attributions: List[AttributionSpan],
                         valid_sources: Set[str]) -> float:
        """
        Calculate percentage of attributions to invalid sources.
        
        Args:
            attributions: List of attribution spans
            valid_sources: Set of valid source IDs for this query
            
        Returns:
            Wrong source rate (0.0 to 1.0, lower is better)
        """
        if not attributions:
            return 0.0
            
        wrong_count = sum(1 for attr in attributions if attr.source_id not in valid_sources)
        return wrong_count / len(attributions)
    
    def attribution_precision_recall(self, predicted_attributions: List[AttributionSpan],
                                   ground_truth_attributions: List[Dict[str, any]]) -> Tuple[float, float]:
        """
        Calculate precision and recall for attributions.
        
        Args:
            predicted_attributions: Predicted attribution spans
            ground_truth_attributions: Expected attributions
            
        Returns:
            Tuple of (precision, recall)
        """
        if not predicted_attributions and not ground_truth_attributions:
            return 1.0, 1.0
            
        if not predicted_attributions:
            return 0.0, 0.0 if ground_truth_attributions else 1.0
            
        if not ground_truth_attributions:
            return 0.0, 1.0
        
        # Convert ground truth to comparable format
        gt_spans = []
        for gt in ground_truth_attributions:
            gt_spans.append({
                'text': gt.get('text', ''),
                'source_id': gt.get('source_id', ''),
                'char_start': gt.get('char_start', 0),
                'char_end': gt.get('char_end', 0)
            })
        
        # Find matches
        matched_predictions = 0
        matched_ground_truth = set()
        
        for pred in predicted_attributions:
            for i, gt in enumerate(gt_spans):
                if i in matched_ground_truth:
                    continue
                    
                # Check if this prediction matches ground truth
                if (pred.source_id == gt['source_id'] and
                    self.partial_span_match(pred.text, gt['text'])):
                    matched_predictions += 1
                    matched_ground_truth.add(i)
                    break
        
        precision = matched_predictions / len(predicted_attributions)
        recall = len(matched_ground_truth) / len(gt_spans)
        
        return precision, recall
    
    def evaluate_attribution_quality(self, search_results: List[SearchResult],
                                   gold_standards: List[AttributionGoldStandard]) -> AttributionMetrics:
        """
        Comprehensive evaluation of attribution quality.
        
        Args:
            search_results: Results with attribution spans
            gold_standards: Ground truth attributions
            
        Returns:
            Attribution quality metrics
        """
        all_exact_matches = []
        all_partial_matches = []
        all_coverage_scores = []
        all_wrong_source_rates = []
        all_precisions = []
        all_recalls = []
        all_confidences = []
        high_confidence_count = 0
        total_attributions = 0
        
        # Create lookup for gold standards
        gold_lookup = {gs.query_id: gs for gs in gold_standards}
        
        for result in search_results:
            if not result.attributions:
                continue
                
            # Find corresponding gold standard (would need query_id in SearchResult)
            # For now, use a heuristic match or assume they're in order
            gold = gold_standards[0] if gold_standards else None  # Simplified
            
            if not gold:
                continue
            
            # Extract metrics for this result
            exact_matches = 0
            partial_matches = 0
            
            for attr in result.attributions:
                total_attributions += 1
                all_confidences.append(attr.confidence)
                
                if attr.confidence > self.confidence_threshold:
                    high_confidence_count += 1
                
                # Compare with expected attributions
                for expected in gold.expected_attributions:
                    if self.exact_span_match(attr.text, expected.get('text', '')):
                        exact_matches += 1
                    elif self.partial_span_match(attr.text, expected.get('text', '')):
                        partial_matches += 1
            
            # Calculate rates for this result
            if result.attributions:
                all_exact_matches.append(exact_matches / len(result.attributions))
                all_partial_matches.append(partial_matches / len(result.attributions))
            
            # Calculate coverage and wrong source rate
            required_sources = {exp.get('source_id', '') for exp in gold.expected_attributions}
            valid_sources = required_sources  # Simplified - could be broader
            
            coverage = self.citation_coverage(result.attributions, required_sources)
            wrong_rate = self.wrong_source_rate(result.attributions, valid_sources)
            
            all_coverage_scores.append(coverage)
            all_wrong_source_rates.append(wrong_rate)
            
            # Calculate precision/recall
            precision, recall = self.attribution_precision_recall(result.attributions, gold.expected_attributions)
            all_precisions.append(precision)
            all_recalls.append(recall)
        
        # Aggregate metrics
        return AttributionMetrics(
            exact_span_match_rate=np.mean(all_exact_matches) if all_exact_matches else 0.0,
            partial_span_match_rate=np.mean(all_partial_matches) if all_partial_matches else 0.0,
            citation_coverage=np.mean(all_coverage_scores) if all_coverage_scores else 0.0,
            wrong_source_rate=np.mean(all_wrong_source_rates) if all_wrong_source_rates else 0.0,
            attribution_precision=np.mean(all_precisions) if all_precisions else 0.0,
            attribution_recall=np.mean(all_recalls) if all_recalls else 0.0,
            average_confidence=np.mean(all_confidences) if all_confidences else 0.0,
            high_confidence_rate=high_confidence_count / total_attributions if total_attributions > 0 else 0.0
        )
    
    def create_attribution_report(self, metrics: AttributionMetrics) -> str:
        """Generate human-readable attribution evaluation report."""
        
        report = f"""
Attribution Fidelity Evaluation Report
=====================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Span Matching Accuracy:
  • Exact Span Match Rate: {metrics.exact_span_match_rate:.1%}
  • Partial Span Match Rate: {metrics.partial_span_match_rate:.1%}

Citation Quality:  
  • Citation Coverage: {metrics.citation_coverage:.1%}
  • Wrong Source Rate: {metrics.wrong_source_rate:.1%} (lower is better)

Attribution Precision/Recall:
  • Precision: {metrics.attribution_precision:.1%}
  • Recall: {metrics.attribution_recall:.1%}

Confidence Assessment:
  • Average Confidence: {metrics.average_confidence:.2f}
  • High Confidence Rate (>{self.confidence_threshold}): {metrics.high_confidence_rate:.1%}

Performance Summary:
{'🟢 EXCELLENT' if metrics.exact_span_match_rate > 0.8 else '🟡 GOOD' if metrics.exact_span_match_rate > 0.6 else '🔴 NEEDS IMPROVEMENT'} - Exact span matching
{'🟢 EXCELLENT' if metrics.citation_coverage > 0.9 else '🟡 GOOD' if metrics.citation_coverage > 0.7 else '🔴 NEEDS IMPROVEMENT'} - Citation coverage  
{'🟢 EXCELLENT' if metrics.wrong_source_rate < 0.1 else '🟡 ACCEPTABLE' if metrics.wrong_source_rate < 0.2 else '🔴 HIGH ERROR RATE'} - Source accuracy
        """
        
        return report


class AttributionBenchmarkLoader:
    """Load attribution evaluation benchmarks."""
    
    def load_default_attribution_benchmarks(self) -> List[AttributionGoldStandard]:
        """Load default attribution benchmarks."""
        
        benchmarks = [
            AttributionGoldStandard(
                query_id="attr_001",
                query_text="How does CRISPR-Cas9 work for gene editing?",
                generated_text="CRISPR-Cas9 uses guide RNAs to target specific DNA sequences and create double-strand breaks for precise genome editing.",
                expected_attributions=[
                    {
                        'text': 'guide RNAs to target specific DNA sequences',
                        'source_id': 'W2963920772',
                        'char_start': 17,
                        'char_end': 56,
                        'confidence': 0.9
                    },
                    {
                        'text': 'double-strand breaks for precise genome editing',
                        'source_id': 'W2963920772', 
                        'char_start': 68,
                        'char_end': 115,
                        'confidence': 0.8
                    }
                ]
            ),
            AttributionGoldStandard(
                query_id="attr_002", 
                query_text="What are the applications of machine learning in drug discovery?",
                generated_text="Machine learning accelerates drug discovery through molecular property prediction, compound screening, and target identification.",
                expected_attributions=[
                    {
                        'text': 'molecular property prediction',
                        'source_id': 'W2748394829',
                        'char_start': 58,
                        'char_end': 87,
                        'confidence': 0.85
                    },
                    {
                        'text': 'compound screening',
                        'source_id': 'W2748394829',
                        'char_start': 89, 
                        'char_end': 107,
                        'confidence': 0.8
                    }
                ]
            )
        ]
        
        return benchmarks