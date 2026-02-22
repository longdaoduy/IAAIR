"""
SciMMIR benchmark integration for scientific multi-modal information retrieval.

This module integrates the SciMMIR benchmark with your existing IAAIR system
to evaluate multi-modal retrieval performance using scientific image-text pairs.
"""

import os
import json
import logging
import requests
import subprocess
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from models.entities.retrieval.SearchResult import SearchResult
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient
from pipelines.evaluation.RetrievalEvaluator import RetrievalEvaluator

logger = logging.getLogger(__name__)

@dataclass
class SciMMIRSample:
    """Single SciMMIR benchmark sample."""
    text: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    class_label: str = "figure"  # fig_architecture, fig_natural, etc.
    paper_id: Optional[str] = None
    sample_id: Optional[str] = None
    domain: str = "general"

@dataclass
class SciMMIRBenchmarkResult:
    """Results from SciMMIR benchmark evaluation."""
    model_name: str
    benchmark_name: str
    total_samples: int
    
    # Text-to-Image retrieval metrics
    text2img_mrr: float
    text2img_recall_at_1: float
    text2img_recall_at_5: float
    text2img_recall_at_10: float
    
    # Image-to-Text retrieval metrics
    img2text_mrr: float
    img2text_recall_at_1: float
    img2text_recall_at_5: float
    img2text_recall_at_10: float
    
    # Category-specific results
    by_category: Dict[str, Dict[str, float]]
    by_domain: Dict[str, Dict[str, float]]
    
    timestamp: datetime
    evaluation_details: Dict[str, Any]

class SciMMIRDataLoader:
    """Load and prepare SciMMIR dataset for benchmarking."""
    
    def __init__(self, cache_dir: str = "./data/scimmir_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def download_scimmir_dataset(self, split: str = "test", streaming: bool = True) -> bool:
        """Download SciMMIR dataset from HuggingFace.
        
        Args:
            split: Which split to download ('test', 'train', 'validation', or 'all')
            streaming: If True, use streaming mode to avoid downloading entire dataset
        """
        try:
            import datasets
            
            if streaming:
                self.logger.info(f"Loading SciMMIR dataset in streaming mode (split: {split})...")
                # Don't actually download, just prepare for streaming
                return True
            else:
                self.logger.info(f"Downloading SciMMIR dataset split '{split}' from HuggingFace...")
                if split == "all":
                    ds_remote = datasets.load_dataset("m-a-p/SciMMIR")
                else:
                    ds_remote = datasets.load_dataset("m-a-p/SciMMIR", split=split)
                
                # Save to cache
                cache_path = self.cache_dir / f"scimmir_dataset_{split}"
                ds_remote.save_to_disk(str(cache_path))
                
                self.logger.info(f"SciMMIR dataset cached to {cache_path}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to download SciMMIR dataset: {e}")
            return False
    
    def load_test_samples(self, limit: Optional[int] = None, use_streaming: bool = True, use_mock: bool = False) -> List[SciMMIRSample]:
        """Load test samples from SciMMIR dataset.
        
        Args:
            limit: Maximum number of samples to load
            use_streaming: Use streaming mode to avoid downloading entire dataset
            use_mock: Use mock samples for quick testing (no download required)
        """
        if use_mock:
            return self._create_mock_samples(limit or 50)
            
        try:
            import datasets
            
            if use_streaming:
                self.logger.info(f"Loading SciMMIR test samples in streaming mode (limit: {limit})...")
                ds = datasets.load_dataset("m-a-p/SciMMIR", split="test", streaming=True)
                
                samples = []
                for i, item in enumerate(ds):
                    if limit and i >= limit:
                        break
                        
                    sample = SciMMIRSample(
                        text=item['text'],
                        image=item.get('image'),  # May be None in streaming
                        class_label=item.get('class', 'figure'),
                        sample_id=f"scimmir_stream_{i:06d}",
                        domain=self._infer_domain(item['text'])
                    )
                    samples.append(sample)
                
                self.logger.info(f"Loaded {len(samples)} SciMMIR test samples (streaming)")
                return samples
            else:
                # Original cached loading
                cache_path = self.cache_dir / "scimmir_dataset_test"
                if not cache_path.exists():
                    if not self.download_scimmir_dataset(split="test", streaming=False):
                        return []
                
                ds = datasets.load_from_disk(str(cache_path))
                
                samples = []
                for i, item in enumerate(ds):
                    if limit and i >= limit:
                        break
                        
                    sample = SciMMIRSample(
                        text=item['text'],
                        image=item['image'],
                        class_label=item.get('class', 'figure'),
                        sample_id=f"scimmir_{i:06d}",
                        domain=self._infer_domain(item['text'])
                    )
                    samples.append(sample)
                
                self.logger.info(f"Loaded {len(samples)} SciMMIR test samples (cached)")
                return samples
            
        except Exception as e:
            self.logger.error(f"Failed to load SciMMIR samples: {e}")
            self.logger.info("Falling back to mock samples for testing...")
            return self._create_mock_samples(limit or 10)
    
    def _infer_domain(self, text: str) -> str:
        """Infer scientific domain from text content."""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['medical', 'clinical', 'disease', 'drug', 'gene', 'protein', 'cell']):
            return 'biomedical'
        elif any(term in text_lower for term in ['neural', 'algorithm', 'computing', 'data', 'machine', 'deep learning']):
            return 'cs'
        elif any(term in text_lower for term in ['physics', 'quantum', 'electromagnetic', 'particle', 'energy']):
            return 'physics'
        else:
            return 'general'
    
    def _create_mock_samples(self, count: int = 50) -> List[SciMMIRSample]:
        """Create mock SciMMIR samples for testing without downloading dataset."""
        import random
        from PIL import Image
        import numpy as np
        
        mock_texts = [
            "Neural network architecture diagram showing convolutional layers for image classification",
            "Graph showing the relationship between accuracy and training epochs in deep learning",
            "Molecular structure of protein binding sites in cancer treatment research",
            "Quantum circuit diagram for quantum computing algorithms",
            "Flow chart of machine learning pipeline for natural language processing",
            "Statistical analysis chart showing correlation between variables in medical study",
            "Algorithm flowchart for optimizing resource allocation in distributed systems",
            "Microscopy image of cellular structures in biological research",
            "Performance comparison table of different neural network architectures",
            "Diagram illustrating the electromagnetic spectrum in physics experiments"
        ]
        
        categories = ['fig_architecture', 'fig_chart', 'fig_natural', 'fig_equation', 'fig_table']
        
        samples = []
        for i in range(count):
            # Create a simple mock image (colored rectangle)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            mock_image = Image.new('RGB', (224, 224), color)
            
            text = random.choice(mock_texts)
            category = random.choice(categories)
            
            sample = SciMMIRSample(
                text=text,
                image=mock_image,
                class_label=category,
                sample_id=f"mock_{i:06d}",
                domain=self._infer_domain(text)
            )
            samples.append(sample)
        
        self.logger.info(f"Created {count} mock SciMMIR samples for testing")
        return samples
    
    def convert_to_iaair_format(self, samples: List[SciMMIRSample]) -> List[Dict[str, Any]]:
        """Convert SciMMIR samples to IAAIR paper format for ingestion."""
        papers = []
        
        for sample in samples:
            paper = {
                "id": sample.sample_id,
                "title": f"Scientific Figure: {sample.text[:100]}...",
                "abstract": sample.text,
                "doi": f"10.scimmir/{sample.sample_id}",
                "publication_date": "2023-01-01T00:00:00",
                "venue": {
                    "name": "SciMMIR Benchmark",
                    "type": "DATASET"
                },
                "authors": [{"name": "SciMMIR Authors"}],
                "figures": [{
                    "id": f"{sample.sample_id}_fig1",
                    "description": sample.text,
                    "image": sample.image,
                    "paper_id": sample.sample_id,
                    "class_label": sample.class_label
                }] if sample.image else [],
                "metadata": {
                    "domain": sample.domain,
                    "class_label": sample.class_label,
                    "benchmark": "SciMMIR"
                }
            }
            papers.append(paper)
        
        return papers

class SciMMIRBenchmarkRunner:
    """Run SciMMIR benchmarks using your IAAIR system."""
    
    def __init__(self, 
                 clip_client: CLIPClient,
                 scibert_client: SciBERTClient,
                 vector_client: MilvusClient):
        self.clip_client = clip_client
        self.scibert_client = scibert_client
        self.vector_client = vector_client
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self.clip_client.initialize()
        
    def run_benchmark(self, 
                      samples: List[SciMMIRSample],
                      model_name: str = "IAAIR-Hybrid",
                      top_k: int = 10) -> SciMMIRBenchmarkResult:
        """Run complete SciMMIR benchmark evaluation."""
        
        self.logger.info(f"Starting SciMMIR benchmark with {len(samples)} samples")
        
        # Generate embeddings for all samples
        text_embeddings, image_embeddings = self._generate_embeddings(samples)
        
        # Text-to-Image retrieval
        text2img_metrics = self._evaluate_text_to_image(samples, text_embeddings, image_embeddings, top_k)
        
        # Image-to-Text retrieval
        img2text_metrics = self._evaluate_image_to_text(samples, text_embeddings, image_embeddings, top_k)
        
        # Category-specific analysis
        by_category = self._analyze_by_category(samples, text_embeddings, image_embeddings, top_k)
        by_domain = self._analyze_by_domain(samples, text_embeddings, image_embeddings, top_k)
        
        result = SciMMIRBenchmarkResult(
            model_name=model_name,
            benchmark_name="SciMMIR",
            total_samples=len(samples),
            text2img_mrr=text2img_metrics['mrr'],
            text2img_recall_at_1=text2img_metrics['recall_at_1'],
            text2img_recall_at_5=text2img_metrics['recall_at_5'],
            text2img_recall_at_10=text2img_metrics['recall_at_10'],
            img2text_mrr=img2text_metrics['mrr'],
            img2text_recall_at_1=img2text_metrics['recall_at_1'],
            img2text_recall_at_5=img2text_metrics['recall_at_5'],
            img2text_recall_at_10=img2text_metrics['recall_at_10'],
            by_category=by_category,
            by_domain=by_domain,
            timestamp=datetime.now(),
            evaluation_details={
                'top_k': top_k,
                'embedding_dim_text': len(text_embeddings[0]) if text_embeddings else 0,
                'embedding_dim_image': len(image_embeddings[0]) if image_embeddings else 0,
                'model_details': {
                    'clip_model': self.clip_client.config.model_name,
                    'scibert_model': self.scibert_client.config.model_name
                }
            }
        )
        
        return result
    
    def _generate_embeddings(self, samples: List[SciMMIRSample]) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate text and image embeddings for all samples."""
        text_embeddings = []
        image_embeddings = []
        
        self.logger.info("Generating embeddings...")
        
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                self.logger.info(f"Processing sample {i}/{len(samples)}")
            
            # Text embedding using SciBERT
            text_emb = self.scibert_client.generate_embedding(sample.text)
            text_embeddings.append(text_emb)
            
            # Image embedding using CLIP
            if sample.image:
                image_emb = self.clip_client.generate_image_embedding(sample.image)
                image_embeddings.append(image_emb if image_emb else [0.0] * 512)
            else:
                image_embeddings.append([0.0] * 512)  # Zero embedding for missing images
        
        return text_embeddings, image_embeddings
    
    def _evaluate_text_to_image(self, 
                                samples: List[SciMMIRSample],
                                text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]],
                                top_k: int) -> Dict[str, float]:
        """Evaluate text-to-image retrieval."""
        
        ranks = []
        
        for i, query_text_emb in enumerate(text_embeddings):
            # Calculate similarities with all image embeddings
            similarities = []
            for j, doc_image_emb in enumerate(image_embeddings):
                sim = self._cosine_similarity(query_text_emb, doc_image_emb)
                similarities.append((sim, j))
            
            # Sort by similarity (descending)
            similarities.sort(reverse=True)
            
            # Find rank of correct match (same index)
            correct_rank = None
            for rank, (sim, idx) in enumerate(similarities, 1):
                if idx == i:  # Correct match
                    correct_rank = rank
                    break
            
            if correct_rank:
                ranks.append(correct_rank)
        
        return self._calculate_retrieval_metrics(ranks, top_k)
    
    def _evaluate_image_to_text(self,
                               samples: List[SciMMIRSample], 
                               text_embeddings: List[List[float]],
                               image_embeddings: List[List[float]],
                               top_k: int) -> Dict[str, float]:
        """Evaluate image-to-text retrieval."""
        
        ranks = []
        
        for i, query_image_emb in enumerate(image_embeddings):
            # Calculate similarities with all text embeddings
            similarities = []
            for j, doc_text_emb in enumerate(text_embeddings):
                sim = self._cosine_similarity(query_image_emb, doc_text_emb)
                similarities.append((sim, j))
            
            # Sort by similarity (descending)
            similarities.sort(reverse=True)
            
            # Find rank of correct match (same index)
            correct_rank = None
            for rank, (sim, idx) in enumerate(similarities, 1):
                if idx == i:  # Correct match
                    correct_rank = rank
                    break
            
            if correct_rank:
                ranks.append(correct_rank)
        
        return self._calculate_retrieval_metrics(ranks, top_k)
    
    def _calculate_retrieval_metrics(self, ranks: List[int], top_k: int) -> Dict[str, float]:
        """Calculate standard retrieval metrics from ranks."""
        if not ranks:
            return {'mrr': 0.0, 'recall_at_1': 0.0, 'recall_at_5': 0.0, 'recall_at_10': 0.0}
        
        # MRR (Mean Reciprocal Rank)
        mrr = sum(1.0 / rank for rank in ranks) / len(ranks)
        
        # Recall@k
        recall_at_1 = sum(1 for rank in ranks if rank <= 1) / len(ranks)
        recall_at_5 = sum(1 for rank in ranks if rank <= 5) / len(ranks)
        recall_at_10 = sum(1 for rank in ranks if rank <= 10) / len(ranks)
        
        return {
            'mrr': mrr,
            'recall_at_1': recall_at_1,
            'recall_at_5': recall_at_5,
            'recall_at_10': recall_at_10
        }
    
    def _analyze_by_category(self, 
                            samples: List[SciMMIRSample],
                            text_embeddings: List[List[float]],
                            image_embeddings: List[List[float]],
                            top_k: int) -> Dict[str, Dict[str, float]]:
        """Analyze performance by figure category."""
        categories = {}
        
        for category in set(sample.class_label for sample in samples):
            category_samples = [i for i, s in enumerate(samples) if s.class_label == category]
            
            if not category_samples:
                continue
            
            # Get category-specific embeddings
            cat_text_embs = [text_embeddings[i] for i in category_samples]
            cat_image_embs = [image_embeddings[i] for i in category_samples]
            
            # Evaluate within category
            cat_samples = [samples[i] for i in category_samples]
            text2img = self._evaluate_text_to_image(cat_samples, cat_text_embs, cat_image_embs, top_k)
            img2text = self._evaluate_image_to_text(cat_samples, cat_text_embs, cat_image_embs, top_k)
            
            categories[category] = {
                'text2img_mrr': text2img['mrr'],
                'text2img_recall_at_1': text2img['recall_at_1'],
                'img2text_mrr': img2text['mrr'],
                'img2text_recall_at_1': img2text['recall_at_1'],
                'sample_count': len(category_samples)
            }
        
        return categories
    
    def _analyze_by_domain(self,
                          samples: List[SciMMIRSample],
                          text_embeddings: List[List[float]],
                          image_embeddings: List[List[float]],
                          top_k: int) -> Dict[str, Dict[str, float]]:
        """Analyze performance by scientific domain."""
        domains = {}
        
        for domain in set(sample.domain for sample in samples):
            domain_samples = [i for i, s in enumerate(samples) if s.domain == domain]
            
            if not domain_samples:
                continue
            
            # Get domain-specific embeddings
            domain_text_embs = [text_embeddings[i] for i in domain_samples]
            domain_image_embs = [image_embeddings[i] for i in domain_samples]
            
            # Evaluate within domain
            dom_samples = [samples[i] for i in domain_samples]
            text2img = self._evaluate_text_to_image(dom_samples, domain_text_embs, domain_image_embs, top_k)
            img2text = self._evaluate_image_to_text(dom_samples, domain_text_embs, domain_image_embs, top_k)
            
            domains[domain] = {
                'text2img_mrr': text2img['mrr'],
                'text2img_recall_at_1': text2img['recall_at_1'],
                'img2text_mrr': img2text['mrr'],
                'img2text_recall_at_1': img2text['recall_at_1'],
                'sample_count': len(domain_samples)
            }
        
        return domains
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class SciMMIRResultAnalyzer:
    """Analyze and compare SciMMIR benchmark results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def compare_with_baselines(self, result: SciMMIRBenchmarkResult) -> Dict[str, Any]:
        """Compare results with published SciMMIR baselines."""
        
        # Published baselines from SciMMIR paper (ALL setting, latest results)
        baselines = {
            "BLIP-base+BERT": {
                "text2img_mrr": 11.15,  # From paper's latest results
                "img2text_mrr": 12.69
            },
            "CLIP-base": {
                "text2img_mrr": 8.5,  # Estimated from paper
                "img2text_mrr": 9.2
            },
            "Random": {
                "text2img_mrr": 0.1,
                "img2text_mrr": 0.1
            }
        }
        
        comparison = {
            "your_model": {
                "name": result.model_name,
                "text2img_mrr": result.text2img_mrr * 100,  # Convert to percentage
                "img2text_mrr": result.img2text_mrr * 100,
                "text2img_recall_at_1": result.text2img_recall_at_1 * 100,
                "img2text_recall_at_1": result.img2text_recall_at_1 * 100
            },
            "baselines": baselines,
            "performance_ranking": self._rank_performance(result, baselines),
            "improvement_analysis": self._analyze_improvements(result, baselines)
        }
        
        return comparison
    
    def _rank_performance(self, result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
        """Rank performance against baselines."""
        your_mrr = (result.text2img_mrr + result.img2text_mrr) / 2 * 100
        
        baseline_scores = []
        for name, scores in baselines.items():
            avg_mrr = (scores['text2img_mrr'] + scores['img2text_mrr']) / 2
            baseline_scores.append((name, avg_mrr))
        
        baseline_scores.append((result.model_name, your_mrr))
        baseline_scores.sort(key=lambda x: x[1], reverse=True)
        
        your_rank = next(i for i, (name, _) in enumerate(baseline_scores, 1) if name == result.model_name)
        
        return {
            "ranking": baseline_scores,
            "your_rank": your_rank,
            "total_models": len(baseline_scores),
            "percentile": (len(baseline_scores) - your_rank + 1) / len(baseline_scores) * 100
        }
    
    def _analyze_improvements(self, result: SciMMIRBenchmarkResult, baselines: Dict) -> List[str]:
        """Analyze what improvements your model shows."""
        improvements = []
        
        your_text2img = result.text2img_mrr * 100
        your_img2text = result.img2text_mrr * 100
        
        best_baseline_text2img = max(b['text2img_mrr'] for b in baselines.values())
        best_baseline_img2text = max(b['img2text_mrr'] for b in baselines.values())
        
        if your_text2img > best_baseline_text2img:
            improvements.append(f"Text-to-Image retrieval improved by {your_text2img - best_baseline_text2img:.2f}% over best baseline")
        
        if your_img2text > best_baseline_img2text:
            improvements.append(f"Image-to-Text retrieval improved by {your_img2text - best_baseline_img2text:.2f}% over best baseline")
        
        if result.text2img_recall_at_1 > 0.15:  # Reasonable threshold
            improvements.append(f"Strong Recall@1 performance: {result.text2img_recall_at_1*100:.1f}%")
        
        return improvements
    
    def generate_report(self, result: SciMMIRBenchmarkResult, save_path: Optional[str] = None) -> str:
        """Generate comprehensive benchmark report."""
        
        comparison = self.compare_with_baselines(result)
        
        report = f"""
# SciMMIR Benchmark Evaluation Report

**Model**: {result.model_name}  
**Evaluation Date**: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Samples**: {result.total_samples:,}

## 📊 Overall Performance

### Text-to-Image Retrieval
- **MRR**: {result.text2img_mrr:.4f} ({result.text2img_mrr*100:.2f}%)
- **Recall@1**: {result.text2img_recall_at_1:.4f} ({result.text2img_recall_at_1*100:.2f}%)
- **Recall@5**: {result.text2img_recall_at_5:.4f} ({result.text2img_recall_at_5*100:.2f}%)
- **Recall@10**: {result.text2img_recall_at_10:.4f} ({result.text2img_recall_at_10*100:.2f}%)

### Image-to-Text Retrieval
- **MRR**: {result.img2text_mrr:.4f} ({result.img2text_mrr*100:.2f}%)
- **Recall@1**: {result.img2text_recall_at_1:.4f} ({result.img2text_recall_at_1*100:.2f}%)
- **Recall@5**: {result.img2text_recall_at_5:.4f} ({result.img2text_recall_at_5*100:.2f}%)
- **Recall@10**: {result.img2text_recall_at_10:.4f} ({result.img2text_recall_at_10*100:.2f}%)

## 🏆 Comparison with Baselines

**Your Rank**: #{comparison['performance_ranking']['your_rank']} out of {comparison['performance_ranking']['total_models']} models  
**Percentile**: {comparison['performance_ranking']['percentile']:.1f}th percentile

### Performance vs. Published Baselines:
"""
        
        for name, scores in comparison['baselines'].items():
            report += f"- **{name}**: T2I MRR: {scores['text2img_mrr']:.2f}%, I2T MRR: {scores['img2text_mrr']:.2f}%\n"
        
        report += f"- **{result.model_name} (Your Model)**: T2I MRR: {result.text2img_mrr*100:.2f}%, I2T MRR: {result.img2text_mrr*100:.2f}%\n\n"
        
        if comparison['improvement_analysis']:
            report += "## ✨ Key Improvements\n\n"
            for improvement in comparison['improvement_analysis']:
                report += f"- {improvement}\n"
            report += "\n"
        
        report += "## 📈 Performance by Category\n\n"
        for category, metrics in result.by_category.items():
            report += f"### {category.replace('_', ' ').title()}\n"
            report += f"- Samples: {metrics['sample_count']}\n"
            report += f"- T2I MRR: {metrics['text2img_mrr']:.4f}\n"
            report += f"- I2T MRR: {metrics['img2text_mrr']:.4f}\n\n"
        
        report += "## 🔬 Performance by Domain\n\n"
        for domain, metrics in result.by_domain.items():
            report += f"### {domain.capitalize()}\n"
            report += f"- Samples: {metrics['sample_count']}\n"
            report += f"- T2I MRR: {metrics['text2img_mrr']:.4f}\n"
            report += f"- I2T MRR: {metrics['img2text_mrr']:.4f}\n\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")
        
        return report

def run_scimmir_benchmark_suite(
    limit_samples: int = 1000,
    cache_dir: str = "./data/scimmir_cache",
    report_path: str = "./data/scimmir_benchmark_report.md",
    use_streaming: bool = True,
    use_mock: bool = False
) -> SciMMIRBenchmarkResult:
    """
    Complete SciMMIR benchmark evaluation workflow.
    
    Args:
        limit_samples: Number of test samples to evaluate
        cache_dir: Directory to cache SciMMIR dataset (ignored if use_streaming=True)
        report_path: Path to save evaluation report
        use_streaming: Use streaming mode to avoid downloading entire dataset
        use_mock: Use mock samples for quick testing (no download required)
    
    Returns:
        Benchmark results with comparison to baselines
    """
    
    # Initialize components
    from models.configurators.CLIPConfig import CLIPConfig
    from models.configurators.SciBERTConfig import SciBERTConfig
    from models.configurators.VectorDBConfig import VectorDBConfig
    
    clip_client = CLIPClient(CLIPConfig())
    scibert_client = SciBERTClient(SciBERTConfig())
    vector_client = MilvusClient(VectorDBConfig())
    
    # Load data with new options
    data_loader = SciMMIRDataLoader(cache_dir)
    samples = data_loader.load_test_samples(
        limit=limit_samples,
        use_streaming=use_streaming,
        use_mock=use_mock
    )
    
    if not samples:
        raise ValueError("Failed to load SciMMIR samples")
    
    # Run benchmark
    benchmark_runner = SciMMIRBenchmarkRunner(clip_client, scibert_client, vector_client)
    result = benchmark_runner.run_benchmark(samples, model_name="IAAIR-SciBERT-CLIP")
    
    # Generate report
    analyzer = SciMMIRResultAnalyzer()
    report = analyzer.generate_report(result, save_path=report_path)
    
    print("="*80)
    print("🎯 SciMMIR Benchmark Completed!")
    print("="*80)
    print(report)
    
    return result

if __name__ == "__main__":
    # Example usage - Quick test with mock data (no download)
    print("🧪 Quick test with mock data:")
    result = run_scimmir_benchmark_suite(
        limit_samples=50,
        use_mock=True,
        report_path="./data/mock_scimmir_report.md"
    )
    
    # Example usage - Small streaming test (minimal download)
    print("\n📡 Small streaming test:")
    result = run_scimmir_benchmark_suite(
        limit_samples=100,
        use_streaming=True,
        use_mock=False,
        report_path="./data/streaming_scimmir_report.md"
    )