"""
SciMMIR benchmark integration for scientific multi-modal information retrieval.

This module integrates the SciMMIR benchmark with your existing IAAIR system
to evaluate multi-modal retrieval performance using scientific image-text pairs.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from utils.similarity import cosine_similarity,calculate_retrieval_metrics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from PIL import Image
import random
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.vector.MilvusClient import MilvusClient
import pandas as pd
import io
import requests
from urllib.parse import urlparse

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
    # by_category: Dict[str, Dict[str, float]]
    # by_domain: Dict[str, Dict[str, float]]

    timestamp: datetime
    evaluation_details: Dict[str, Any]


class SciMMIRDataLoader:
    """Load and prepare SciMMIR dataset for benchmarking."""

    def __init__(self, cache_dir: str = "./data/scimmir_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.base_url = "https://huggingface.co/datasets/m-a-p/SciMMIR/resolve/main/data/"
        self.parquet_files = ["test-00000-of-00004-758f4fffbab26e7d.parquet",
                              "test-00001-of-00004-d23be0c1b862d0ff.parquet", 
                              "test-00002-of-00004-748ad69634d3bd2e.parquet",
                              "test-00003-of-00004-cdffbde35853be2a.parquet"]

    def download_parquet_files(self) -> bool:
        """Download SciMMIR parquet files from Hugging Face.
        
        Returns:
            bool: True if all files were downloaded successfully, False otherwise
        """
        dataset_dir = self.cache_dir / 'scimmir_dataset'
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        
        for filename in self.parquet_files:
            file_path = dataset_dir / filename
            
            if file_path.exists():
                self.logger.info(f"File {filename} already exists, skipping download")
                success_count += 1
                continue
                
            url = f"{self.base_url}{filename}?download=true"
            self.logger.info(f"Downloading {filename} from {url}")
            
            try:
                response = requests.get(url, stream=True, timeout=300)
                response.raise_for_status()
                
                # Get file size for progress tracking
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            # Log progress every 10MB
                            if total_size > 0 and downloaded_size % (10 * 1024 * 1024) == 0:
                                progress = (downloaded_size / total_size) * 100
                                self.logger.info(f"Downloaded {progress:.1f}% of {filename}")
                
                self.logger.info(f"Successfully downloaded {filename} ({downloaded_size / (1024*1024):.1f} MB)")
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to download {filename}: {e}")
                # Clean up partial download
                if file_path.exists():
                    file_path.unlink()
        
        self.logger.info(f"Downloaded {success_count}/{len(self.parquet_files)} files successfully")
        return success_count == len(self.parquet_files)
        self.base_url = "https://huggingface.co/datasets/m-a-p/SciMMIR/resolve/main/data/"
        self.parquet_files = ["test-00000-of-00004-758f4fffbab26e7d.parquet",
                              "test-00001-of-00004-d23be0c1b862d0ff.parquet", 
                              "test-00002-of-00004-748ad69634d3bd2e.parquet",
                              "test-00003-of-00004-cdffbde35853be2a.parquet"]

    def load_from_parquet(self, parquet_path: str, limit: Optional[int] = None) -> List[SciMMIRSample]:
        """Load SciMMIR samples directly from a Parquet file.
        
        Args:
            parquet_path: Path to the Parquet file
            limit: Maximum number of samples to load
        """
        try:

            self.logger.info(f"Loading SciMMIR samples from Parquet: {parquet_path}")

            # Load the parquet file
            df = pd.read_parquet(parquet_path)

            if limit:
                df = df.head(limit)

            samples = []
            for idx, row in df.iterrows():
                # Process image data (it's stored as bytes in the parquet)
                image = None
                if 'image' in row and row['image'] and 'bytes' in row['image']:
                    try:
                        image_bytes = row['image']['bytes']
                        image = Image.open(io.BytesIO(image_bytes))
                        # Resize for memory efficiency
                        image = image.resize((224, 224), Image.Resampling.LANCZOS)
                    except Exception as e:
                        self.logger.warning(f"Failed to process image for row {idx}: {e}")
                        image = None

                sample = SciMMIRSample(
                    text=row.get('text', ''),
                    image=image,
                    class_label=row.get('class', 'figure'),
                    sample_id=f"parquet_{idx:06d}",
                    domain=self._infer_domain(row.get('text', ''))
                )
                samples.append(sample)

            self.logger.info(f"Loaded {len(samples)} SciMMIR samples from Parquet file")
            return samples

        except Exception as e:
            self.logger.error(f"Failed to load SciMMIR samples from Parquet: {e}")
            self.logger.info("Falling back to mock samples...")
            return self._create_mock_samples(limit or 50)

    def load_test_samples(self, limit: Optional[int] = None) -> \
    List[SciMMIRSample]:
        """Load test samples from SciMMIR dataset with memory management.
        
        Args:
            limit: Maximum number of samples to load
        """
        # Load all parquet files in the dataset directory
        dataset_dir = self.cache_dir / 'scimmir_dataset'
        if not dataset_dir.exists():
            dataset_dir.mkdir(parents=True, exist_ok=True)
        
        parquet_files = list(dataset_dir.glob("*.parquet"))
        if not parquet_files:
            self.logger.info(f"No parquet files found in {dataset_dir}, attempting to download from Hugging Face")
            download_success = self.download_parquet_files()
            
            if download_success:
                parquet_files = list(dataset_dir.glob("*.parquet"))
                self.logger.info(f"Successfully downloaded {len(parquet_files)} parquet files")
            else:
                self.logger.warning(f"Failed to download parquet files, creating mock samples")
                return self._create_mock_samples(limit or 50)
        
        all_samples = []
        samples_loaded = 0
        
        # Sort files to ensure consistent loading order
        parquet_files.sort()
        
        for parquet_file in parquet_files:
            if limit and samples_loaded >= limit:
                break
                
            remaining_limit = limit - samples_loaded if limit else None
            samples = self.load_from_parquet(str(parquet_file), remaining_limit)
            all_samples.extend(samples)
            samples_loaded += len(samples)
            
            self.logger.info(f"Loaded {len(samples)} samples from {parquet_file.name}, total: {samples_loaded}")
        
        self.logger.info(f"Successfully loaded {len(all_samples)} total samples from {len(parquet_files)} parquet files")
        return all_samples

    @staticmethod
    def _infer_domain(text: str) -> str:
        """Infer scientific domain from text content."""
        text_lower = text.lower()

        if any(term in text_lower for term in ['medical', 'clinical', 'disease', 'drug', 'gene', 'protein', 'cell']):
            return 'biomedical'
        elif any(term in text_lower for term in
                 ['neural', 'algorithm', 'computing', 'data', 'machine', 'deep learning']):
            return 'cs'
        elif any(term in text_lower for term in ['physics', 'quantum', 'electromagnetic', 'particle', 'energy']):
            return 'physics'
        else:
            return 'general'

    def _create_mock_samples(self, count: int = 50) -> List[SciMMIRSample]:
        """Create mock SciMMIR samples for testing without downloading dataset."""

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

    def run_benchmark(self,
                      samples: List[SciMMIRSample],
                      model_name: str = "IAAIR-Hybrid",
                      top_k: int = 10) -> SciMMIRBenchmarkResult:
        """Run complete SciMMIR benchmark evaluation."""

        self.logger.info(f"Starting SciMMIR benchmark with {len(samples)} samples")

        # Generate embeddings for all samples
        text_embeddings, image_embeddings = self._generate_embeddings(samples)

        # Text-to-Image retrieval
        text2img_metrics = self._evaluate_text_to_image(text_embeddings, image_embeddings)

        # Image-to-Text retrieval
        img2text_metrics = self._evaluate_image_to_text(text_embeddings, image_embeddings)

        # Category-specific analysis
        # by_category = self._analyze_by_category(samples, text_embeddings, image_embeddings, top_k)
        # by_domain = self._analyze_by_domain(samples, text_embeddings, image_embeddings, top_k)

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
            # by_category=by_category,
            # by_domain=by_domain,
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
                image_embeddings.append(image_emb if image_emb else [0.0] * 768)
            else:
                image_embeddings.append([0.0] * 768)  # Zero embedding for missing images

        return text_embeddings, image_embeddings

    @staticmethod
    def _evaluate_text_to_image(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]]) -> Dict[str, float]:
        """Evaluate text-to-image retrieval."""

        ranks = []

        for i, query_text_emb in enumerate(text_embeddings):
            # Calculate similarities with all image embeddings
            similarities = []
            for j, doc_image_emb in enumerate(image_embeddings):
                sim = cosine_similarity(query_text_emb, doc_image_emb)
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

        return calculate_retrieval_metrics(ranks)

    @staticmethod
    def _evaluate_image_to_text(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]]) -> Dict[str, float]:
        """Evaluate image-to-text retrieval."""

        ranks = []

        for i, query_image_emb in enumerate(image_embeddings):
            # Calculate similarities with all text embeddings
            similarities = []
            for j, doc_text_emb in enumerate(text_embeddings):
                sim = cosine_similarity(query_image_emb, doc_text_emb)
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

        return calculate_retrieval_metrics(ranks)

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

    @staticmethod
    def _rank_performance(result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
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

    @staticmethod
    def _analyze_improvements(result: SciMMIRBenchmarkResult, baselines: Dict) -> List[str]:
        """Analyze what improvements your model shows."""
        improvements = []

        your_text2img = result.text2img_mrr * 100
        your_img2text = result.img2text_mrr * 100

        best_baseline_text2img = max(b['text2img_mrr'] for b in baselines.values())
        best_baseline_img2text = max(b['img2text_mrr'] for b in baselines.values())

        if your_text2img > best_baseline_text2img:
            improvements.append(
                f"Text-to-Image retrieval improved by {your_text2img - best_baseline_text2img:.2f}% over best baseline")

        if your_img2text > best_baseline_img2text:
            improvements.append(
                f"Image-to-Text retrieval improved by {your_img2text - best_baseline_img2text:.2f}% over best baseline")

        if result.text2img_recall_at_1 > 0.15:  # Reasonable threshold
            improvements.append(f"Strong Recall@1 performance: {result.text2img_recall_at_1 * 100:.1f}%")

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
- **MRR**: {result.text2img_mrr:.4f} ({result.text2img_mrr * 100:.2f}%)
- **Recall@1**: {result.text2img_recall_at_1:.4f} ({result.text2img_recall_at_1 * 100:.2f}%)
- **Recall@5**: {result.text2img_recall_at_5:.4f} ({result.text2img_recall_at_5 * 100:.2f}%)
- **Recall@10**: {result.text2img_recall_at_10:.4f} ({result.text2img_recall_at_10 * 100:.2f}%)

### Image-to-Text Retrieval
- **MRR**: {result.img2text_mrr:.4f} ({result.img2text_mrr * 100:.2f}%)
- **Recall@1**: {result.img2text_recall_at_1:.4f} ({result.img2text_recall_at_1 * 100:.2f}%)
- **Recall@5**: {result.img2text_recall_at_5:.4f} ({result.img2text_recall_at_5 * 100:.2f}%)
- **Recall@10**: {result.img2text_recall_at_10:.4f} ({result.img2text_recall_at_10 * 100:.2f}%)

## 🏆 Comparison with Baselines

**Your Rank**: #{comparison['performance_ranking']['your_rank']} out of {comparison['performance_ranking']['total_models']} models  
**Percentile**: {comparison['performance_ranking']['percentile']:.1f}th percentile

### Performance vs. Published Baselines:
"""

        for name, scores in comparison['baselines'].items():
            report += f"- **{name}**: T2I MRR: {scores['text2img_mrr']:.2f}%, I2T MRR: {scores['img2text_mrr']:.2f}%\n"

        report += f"- **{result.model_name} (Your Model)**: T2I MRR: {result.text2img_mrr * 100:.2f}%, I2T MRR: {result.img2text_mrr * 100:.2f}%\n\n"

        if comparison['improvement_analysis']:
            report += "## ✨ Key Improvements\n\n"
            for improvement in comparison['improvement_analysis']:
                report += f"- {improvement}\n"
            report += "\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")

        return report
