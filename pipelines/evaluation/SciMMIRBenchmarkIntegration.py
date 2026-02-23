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
import psutil
import gc

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

    timestamp: datetime
    evaluation_details: Dict[str, Any]
    
    # Subset-specific results (like CLIP-BERT evaluation) - must be after non-default fields
    subset_results: Optional[Dict[str, Dict[str, float]]] = None  # e.g., "figure_result": {"text2img_mrr": 0.12, ...}
    
    # Category-specific results
    # by_category: Dict[str, Dict[str, float]] = None
    # by_domain: Dict[str, Dict[str, float]] = None


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
        
        # Subset mapping based on SciMMIR benchmark methodology
        self.subset_mapping = {
            # Figure subsets
            'fig_result': 'figure_result',
            'fig_chart': 'figure_result', 
            'fig_plot': 'figure_result',
            'fig_graph': 'figure_result',
            'fig_diagram': 'figure_illustration',
            'fig_drawing': 'figure_illustration',
            'fig_illustration': 'figure_illustration',
            'fig_schema': 'figure_illustration',
            'fig_architecture': 'figure_architecture',
            'fig_flowchart': 'figure_architecture',
            'fig_pipeline': 'figure_architecture',
            
            # Table subsets
            'tab_result': 'table_result',
            'tab_comparison': 'table_result',
            'tab_data': 'table_result',
            'tab_parameter': 'table_parameter',
            'tab_config': 'table_parameter',
            'tab_hyperparameter': 'table_parameter'
        }

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

    def get_subset_category(self, class_label: str, text: str = "") -> str:
        """Categorize a sample into SciMMIR benchmark subsets.
        
        Args:
            class_label: The class label from SciMMIR dataset
            text: Optional text content for additional context
            
        Returns:
            Subset category (figure_result, figure_illustration, etc.)
        """
        # Direct mapping from class label
        if class_label in self.subset_mapping:
            return self.subset_mapping[class_label]
            
        # Fallback inference from text content
        text_lower = text.lower()
        
        # Figure classifications
        if class_label.startswith('fig'):
            if any(term in text_lower for term in ['result', 'performance', 'accuracy', 'comparison', 'evaluation']):
                return 'figure_result'
            elif any(term in text_lower for term in ['architecture', 'model', 'network', 'pipeline', 'workflow']):
                return 'figure_architecture'
            else:
                return 'figure_illustration'
        
        # Table classifications  
        elif class_label.startswith('tab'):
            if any(term in text_lower for term in ['parameter', 'hyperparameter', 'config', 'setting']):
                return 'table_parameter'
            else:
                return 'table_result'
        
        # Default fallback
        return 'figure_result' if 'fig' in class_label else 'table_result'
    
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

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
        except Exception as e:
            self.logger.warning(f"Could not get memory usage: {e}")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': 0}

    def _optimize_batch_size(self, total_samples: int, target_memory_mb: float = 4000) -> int:
        """Dynamically determine optimal batch size based on available memory.
        
        Args:
            total_samples: Total number of samples to process
            target_memory_mb: Target memory usage in MB
        """
        memory_stats = self._get_memory_usage()
        available_mb = memory_stats.get('available_mb', 1000)  # Default fallback
        
        # Conservative batch size calculation
        # Assume each sample uses ~10MB (text + image embeddings + overhead)
        estimated_mb_per_sample = 10
        safe_memory_limit = min(target_memory_mb, available_mb * 0.7)  # Use 70% of available
        
        optimal_batch_size = max(1, int(safe_memory_limit / estimated_mb_per_sample))
        
        # Clamp to reasonable bounds
        optimal_batch_size = min(optimal_batch_size, 128)  # Max 128
        optimal_batch_size = max(optimal_batch_size, 8)    # Min 8
        
        self.logger.info(f"Memory optimization: Available={available_mb:.0f}MB, "
                        f"Target={target_memory_mb:.0f}MB, Batch size={optimal_batch_size}")
        
        return optimal_batch_size

    def run_benchmark(self,
                      samples: List[SciMMIRSample],
                      model_name: str = "IAAIR-Hybrid",
                      top_k: int = 10,
                      evaluate_subsets: bool = True,
                      batch_size: Optional[int] = None) -> SciMMIRBenchmarkResult:
        """Run complete SciMMIR benchmark evaluation.
        
        Args:
            samples: List of SciMMIR samples
            model_name: Name of the model being evaluated
            top_k: Top-k for evaluation metrics
            evaluate_subsets: Whether to evaluate subsets
            batch_size: Batch size for embedding generation (auto-optimized if None)
        """

        # Auto-optimize batch size if not provided
        if batch_size is None:
            batch_size = self._optimize_batch_size(len(samples))
        
        self.logger.info(f"Starting SciMMIR benchmark with {len(samples)} samples (batch_size={batch_size})")
        
        # Log initial memory usage
        initial_memory = self._get_memory_usage()
        self.logger.info(f"Initial memory usage: {initial_memory['rss_mb']:.1f}MB RSS, "
                        f"{initial_memory['available_mb']:.1f}MB available")

        # Generate embeddings for all samples with batch processing
        text_embeddings, image_embeddings = self._generate_embeddings(samples, batch_size=batch_size)

        # Text-to-Image retrieval with batch processing
        text2img_metrics = self._evaluate_text_to_image(text_embeddings, image_embeddings, batch_size)

        # Image-to-Text retrieval with batch processing
        img2text_metrics = self._evaluate_image_to_text(text_embeddings, image_embeddings, batch_size)

        # Evaluate subsets if requested (like CLIP-BERT methodology) with batch processing
        subset_results = None
        if evaluate_subsets:
            subset_results = self._evaluate_subsets(samples, text_embeddings, image_embeddings, batch_size)

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
            subset_results=subset_results,
            timestamp=datetime.now(),
            evaluation_details={
                'top_k': top_k,
                'batch_size': batch_size,
                'embedding_dim_text': len(text_embeddings[0]) if text_embeddings and text_embeddings[0] else 0,
                'embedding_dim_image': len(image_embeddings[0]) if image_embeddings and image_embeddings[0] else 0,
                'model_details': {
                    'clip_model': getattr(self.clip_client.config, 'model_name', 'unknown') if hasattr(self.clip_client, 'config') and self.clip_client.config else 'unknown',
                    'scibert_model': getattr(self.scibert_client.config, 'model_name', 'unknown') if hasattr(self.scibert_client, 'config') and self.scibert_client.config else 'unknown'
                }
            }
        )

        return result

    def _generate_embeddings(self, samples: List[SciMMIRSample], batch_size: int = 32) -> Tuple[List[List[float]], List[List[float]]]:
        """Generate text and image embeddings for all samples with memory optimization.
        
        Args:
            samples: List of SciMMIR samples
            batch_size: Number of samples to process at once (reduce if OOM)
        """
        import gc
        
        text_embeddings = []
        image_embeddings = []
        
        total_batches = (len(samples) + batch_size - 1) // batch_size
        self.logger.info(f"Generating embeddings in {total_batches} batches of {batch_size}...")
        
        for batch_idx in range(0, len(samples), batch_size):
            batch_samples = samples[batch_idx:batch_idx + batch_size]
            batch_num = (batch_idx // batch_size) + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_samples)} samples)")
            
            # Process text embeddings for this batch
            batch_texts = [sample.text for sample in batch_samples]
            try:
                if hasattr(self.scibert_client, 'generate_batch_embeddings'):
                    # Use batch processing if available
                    batch_text_embeddings = self.scibert_client.generate_batch_embeddings(batch_texts)
                    if not batch_text_embeddings:
                        # Fallback if batch processing returns None/empty
                        fallback_dim = 768
                        batch_text_embeddings = [[0.0] * fallback_dim for _ in batch_texts]
                else:
                    # Fall back to individual processing
                    batch_text_embeddings = []
                    for text in batch_texts:
                        emb = self.scibert_client.generate_embedding(text) if self.scibert_client else None
                        if emb is None:
                            emb = [0.0] * 768  # Standard BERT dimension fallback
                        batch_text_embeddings.append(emb)
                        
                text_embeddings.extend(batch_text_embeddings)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate text embeddings for batch {batch_num}: {e}")
                # Create zero embeddings as fallback
                fallback_dim = 768  # Standard BERT dimension
                batch_text_embeddings = [[0.0] * fallback_dim for _ in batch_texts]
                text_embeddings.extend(batch_text_embeddings)
            
            # Process image embeddings for this batch
            batch_images = [sample.image for sample in batch_samples]
            try:
                if hasattr(self.clip_client, 'generate_batch_image_embeddings'):
                    # Use batch processing if available
                    batch_image_embeddings = self.clip_client.generate_batch_image_embeddings(batch_images)
                    if not batch_image_embeddings:
                        # Fallback if batch processing returns None/empty
                        fallback_dim = 768
                        batch_image_embeddings = [[0.0] * fallback_dim for _ in batch_images]
                else:
                    # Fall back to individual processing
                    batch_image_embeddings = []
                    for image in batch_images:
                        if image and self.clip_client:
                            emb = self.clip_client.generate_image_embedding(image)
                            batch_image_embeddings.append(emb if emb is not None else [0.0] * 768)
                        else:
                            batch_image_embeddings.append([0.0] * 768)  # Standard CLIP dimension
                            
                image_embeddings.extend(batch_image_embeddings)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate image embeddings for batch {batch_num}: {e}")
                # Create zero embeddings as fallback
                fallback_dim = 768  # Standard CLIP dimension
                batch_image_embeddings = [[0.0] * fallback_dim for _ in batch_images]
                image_embeddings.extend(batch_image_embeddings)
            
            # Clear batch data and force garbage collection
            del batch_samples, batch_texts, batch_images
            if 'batch_text_embeddings' in locals():
                del batch_text_embeddings
            if 'batch_image_embeddings' in locals():
                del batch_image_embeddings
            gc.collect()
            
            # Progress update
            progress = (batch_num / total_batches) * 100
            self.logger.info(f"Embedding generation progress: {progress:.1f}% ({len(text_embeddings)} samples processed)")
        
        self.logger.info(f"Completed embedding generation for {len(text_embeddings)} samples")
        return text_embeddings, image_embeddings

    @staticmethod
    def _evaluate_text_to_image(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]],
                                batch_size: int = 100) -> Dict[str, float]:
        """Evaluate text-to-image retrieval with memory optimization."""
        import gc
        
        ranks = []
        total_queries = len(text_embeddings)
        
        # Process queries in batches to avoid memory issues
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_text_embeddings = text_embeddings[batch_start:batch_end]
            
            for local_i, query_text_emb in enumerate(batch_text_embeddings):
                global_i = batch_start + local_i
                
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
                    if idx == global_i:  # Correct match
                        correct_rank = rank
                        break

                if correct_rank:
                    ranks.append(correct_rank)
            
            # Clear batch data and force garbage collection
            del batch_text_embeddings
            gc.collect()
            
            # Log progress for large datasets
            if total_queries > 1000 and batch_start % (batch_size * 10) == 0:
                progress = (batch_end / total_queries) * 100
                print(f"Text-to-Image evaluation progress: {progress:.1f}%")

        return calculate_retrieval_metrics(ranks)

    @staticmethod
    def _evaluate_image_to_text(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]],
                                batch_size: int = 100) -> Dict[str, float]:
        """Evaluate image-to-text retrieval with memory optimization."""
        import gc
        
        ranks = []
        total_queries = len(image_embeddings)
        
        # Process queries in batches to avoid memory issues
        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_image_embeddings = image_embeddings[batch_start:batch_end]
            
            for local_i, query_image_emb in enumerate(batch_image_embeddings):
                global_i = batch_start + local_i
                
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
                    if idx == global_i:  # Correct match
                        correct_rank = rank
                        break

                if correct_rank:
                    ranks.append(correct_rank)
            
            # Clear batch data and force garbage collection
            del batch_image_embeddings
            gc.collect()
            
            # Log progress for large datasets
            if total_queries > 1000 and batch_start % (batch_size * 10) == 0:
                progress = (batch_end / total_queries) * 100
                print(f"Image-to-Text evaluation progress: {progress:.1f}%")

        return calculate_retrieval_metrics(ranks)

    def _evaluate_subsets(self, 
                         samples: List[SciMMIRSample], 
                         text_embeddings: List[List[float]], 
                         image_embeddings: List[List[float]],
                         batch_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Evaluate performance on SciMMIR subsets like CLIP-BERT methodology with memory optimization.
        
        Based on the SciMMIR benchmark methodology:
        • Figure Subset (11,491 total test samples):
            ◦ Figure Result: 9,488 samples
            ◦ Figure Illustration: 1,536 samples  
            ◦ Figure Architecture: 467 samples
        • Table Subset (4,772 total test samples):
            ◦ Table Result: 4,229 samples
            ◦ Table Parameter: 543 samples
        """
        import gc
        
        # Initialize data loader to get subset mapping
        data_loader = SciMMIRDataLoader()
        
        # Group samples by subset
        subset_groups = {}
        for i, sample in enumerate(samples):
            subset_category = data_loader.get_subset_category(sample.class_label, sample.text)
            if subset_category not in subset_groups:
                subset_groups[subset_category] = []
            subset_groups[subset_category].append(i)
        
        self.logger.info(f"Evaluating {len(subset_groups)} subsets: {list(subset_groups.keys())}")
        for subset, indices in subset_groups.items():
            self.logger.info(f"  {subset}: {len(indices)} samples")
        
        subset_results = {}
        
        # Evaluate each subset with memory optimization
        for subset_name, indices in subset_groups.items():
            if len(indices) < 2:  # Need at least 2 samples for meaningful evaluation
                self.logger.warning(f"Skipping subset {subset_name} with only {len(indices)} samples")
                continue
                
            self.logger.info(f"Evaluating subset {subset_name} with {len(indices)} samples...")
            
            # Extract embeddings for this subset
            subset_text_embeddings = [text_embeddings[i] for i in indices]
            subset_image_embeddings = [image_embeddings[i] for i in indices]
            
            # Use smaller batch size for subsets to be more memory efficient
            subset_batch_size = min(batch_size, max(10, len(indices) // 4))
            
            # Evaluate text-to-image for this subset with batch processing
            text2img_metrics = self._evaluate_text_to_image(subset_text_embeddings, subset_image_embeddings, subset_batch_size)
            
            # Evaluate image-to-text for this subset with batch processing
            img2text_metrics = self._evaluate_image_to_text(subset_text_embeddings, subset_image_embeddings, subset_batch_size)
            
            # Store results
            subset_results[subset_name] = {
                'sample_count': len(indices),
                'text2img_mrr': text2img_metrics['mrr'],
                'text2img_recall_at_1': text2img_metrics['recall_at_1'],
                'text2img_recall_at_5': text2img_metrics['recall_at_5'],
                'text2img_recall_at_10': text2img_metrics['recall_at_10'],
                'img2text_mrr': img2text_metrics['mrr'],
                'img2text_recall_at_1': img2text_metrics['recall_at_1'],
                'img2text_recall_at_5': img2text_metrics['recall_at_5'],
                'img2text_recall_at_10': img2text_metrics['recall_at_10']
            }
            
            self.logger.info(f"Subset {subset_name}: T2I MRR={text2img_metrics['mrr']:.4f}, I2T MRR={img2text_metrics['mrr']:.4f}")
            
            # Clean up subset embeddings and force garbage collection
            del subset_text_embeddings, subset_image_embeddings
            gc.collect()
        
        return subset_results

class SciMMIRResultAnalyzer:
    """Analyze and compare SciMMIR benchmark results."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def compare_with_baselines(self, result: SciMMIRBenchmarkResult) -> Dict[str, Any]:
        """Compare results with published SciMMIR baselines."""

        # Published baselines from SciMMIR paper with subset-specific performance
        baselines = {
            "CLIP-base": {
                # Overall performance (estimated averages)
                "text2img_mrr": 0.652,  # Average across all subsets
                "img2text_mrr": 0.641,
                # Subset-specific performance
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 1.351, "text2img_hits_at_10": 1.927,
                        "img2text_mrr": 1.074, "img2text_hits_at_10": 2.141
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.750, "text2img_hits_at_10": 1.237,
                        "img2text_mrr": 0.458, "img2text_hits_at_10": 0.716
                    },
                    "figure_result": {
                        "text2img_mrr": 0.373, "text2img_hits_at_10": 0.643,
                        "img2text_mrr": 0.386, "img2text_hits_at_10": 0.738
                    },
                    "table_result": {
                        "text2img_mrr": 0.281, "text2img_hits_at_10": 0.544,
                        "img2text_mrr": 0.177, "img2text_hits_at_10": 0.284
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.545, "text2img_hits_at_10": 0.921,
                        "img2text_mrr": 0.558, "img2text_hits_at_10": 1.105
                    }
                }
            },
            "BLIP2-OPT-2.7B": {
                "text2img_mrr": 0.100,  # Average across all subsets
                "img2text_mrr": 0.027,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.130, "text2img_hits_at_10": 0.214,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.033, "text2img_hits_at_10": 0.130,
                        "img2text_mrr": 0.006, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.031, "text2img_hits_at_10": 0.042,
                        "img2text_mrr": 0.014, "img2text_hits_at_10": 0.032
                    },
                    "table_result": {
                        "text2img_mrr": 0.076, "text2img_hits_at_10": 0.213,
                        "img2text_mrr": 0.010, "img2text_hits_at_10": 0.024
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.228, "text2img_hits_at_10": 0.368,
                        "img2text_mrr": 0.101, "img2text_hits_at_10": 0.184
                    }
                }
            },
            "BLIP2-FLAN-T5-XLL": {
                "text2img_mrr": 0.045,
                "img2text_mrr": 0.004,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.056, "text2img_hits_at_10": 0.214,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.037, "text2img_hits_at_10": 0.065,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.062, "text2img_hits_at_10": 0.105,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.041, "text2img_hits_at_10": 0.095,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.030, "text2img_hits_at_10": 0.184,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    }
                }
            },
            "mPLUG-Owl2-LLaMA2-7B": {
                "text2img_mrr": 0.070,
                "img2text_mrr": 0.003,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.022, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.302, "text2img_hits_at_10": 0.521,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.019, "text2img_hits_at_10": 0.021,
                        "img2text_mrr": 0.002, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.001, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.002, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.005, "img2text_hits_at_10": 0.000
                    }
                }
            },
            "Kosmos-2": {
                "text2img_mrr": 0.028,
                "img2text_mrr": 0.004,
                "subsets": {
                    "figure_architecture": {
                        "text2img_mrr": 0.123, "text2img_hits_at_10": 0.428,
                        "img2text_mrr": 0.008, "img2text_hits_at_10": 0.000
                    },
                    "figure_illustration": {
                        "text2img_mrr": 0.011, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.004, "img2text_hits_at_10": 0.000
                    },
                    "figure_result": {
                        "text2img_mrr": 0.006, "text2img_hits_at_10": 0.011,
                        "img2text_mrr": 0.002, "img2text_hits_at_10": 0.000
                    },
                    "table_result": {
                        "text2img_mrr": 0.000, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.001, "img2text_hits_at_10": 0.000
                    },
                    "table_parameter": {
                        "text2img_mrr": 0.000, "text2img_hits_at_10": 0.000,
                        "img2text_mrr": 0.003, "img2text_hits_at_10": 0.000
                    }
                }
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
        
        # Add subset comparisons if available
        if result.subset_results:
            comparison["subset_performance"] = {
                "figure_subset": {
                    subset: metrics for subset, metrics in result.subset_results.items() 
                    if subset.startswith('figure_')
                },
                "table_subset": {
                    subset: metrics for subset, metrics in result.subset_results.items() 
                    if subset.startswith('table_')
                }
            }
            
            # Calculate subset totals (like CLIP-BERT methodology)
            figure_total_samples = sum(
                metrics['sample_count'] for subset, metrics in result.subset_results.items()
                if subset.startswith('figure_')
            )
            table_total_samples = sum(
                metrics['sample_count'] for subset, metrics in result.subset_results.items()
                if subset.startswith('table_')
            )
            
            comparison["subset_totals"] = {
                "figure_subset_samples": figure_total_samples,
                "table_subset_samples": table_total_samples,
                "total_subset_samples": figure_total_samples + table_total_samples
            }
            
            # Add detailed subset-specific comparisons with baselines
            comparison["subset_baseline_comparison"] = self._compare_subset_performance(result, baselines)

        return comparison

    def _compare_subset_performance(self, result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
        """Compare subset-specific performance with baseline models."""
        subset_comparisons = {}
        
        if not result.subset_results:
            return subset_comparisons
        
        for subset_name, your_metrics in result.subset_results.items():
            subset_comparisons[subset_name] = {
                "your_performance": {
                    "text2img_mrr": your_metrics['text2img_mrr'] * 100,
                    "img2text_mrr": your_metrics['img2text_mrr'] * 100,
                    "text2img_recall_at_10": your_metrics.get('text2img_recall_at_10', 0) * 100,
                    "img2text_recall_at_10": your_metrics.get('img2text_recall_at_10', 0) * 100,
                    "sample_count": your_metrics['sample_count']
                },
                "baseline_comparison": {},
                "ranking": {}
            }
            
            # Compare with each baseline that has subset data
            baseline_scores = []
            for model_name, baseline_data in baselines.items():
                if "subsets" in baseline_data and subset_name in baseline_data["subsets"]:
                    baseline_subset = baseline_data["subsets"][subset_name]
                    subset_comparisons[subset_name]["baseline_comparison"][model_name] = {
                        "text2img_mrr": baseline_subset["text2img_mrr"],
                        "img2text_mrr": baseline_subset["img2text_mrr"],
                        "text2img_hits_at_10": baseline_subset.get("text2img_hits_at_10", 0),
                        "img2text_hits_at_10": baseline_subset.get("img2text_hits_at_10", 0)
                    }
                    
                    # Calculate average MRR for ranking
                    avg_mrr = (baseline_subset["text2img_mrr"] + baseline_subset["img2text_mrr"]) / 2
                    baseline_scores.append((model_name, avg_mrr))
            
            # Add your model to ranking
            your_avg_mrr = (your_metrics['text2img_mrr'] + your_metrics['img2text_mrr']) / 2 * 100
            baseline_scores.append((result.model_name, your_avg_mrr))
            baseline_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Find your rank
            your_rank = next(i for i, (name, _) in enumerate(baseline_scores, 1) if name == result.model_name)
            
            subset_comparisons[subset_name]["ranking"] = {
                "your_rank": your_rank,
                "total_models": len(baseline_scores),
                "all_rankings": baseline_scores
            }
        
        return subset_comparisons

    @staticmethod
    def _rank_performance(result: SciMMIRBenchmarkResult, baselines: Dict) -> Dict[str, Any]:
        """Rank performance against baselines."""
        your_mrr = (result.text2img_mrr + result.img2text_mrr) / 2 * 100

        baseline_scores = []
        for name, baseline_data in baselines.items():
            # Use overall MRR if available, otherwise skip
            if "text2img_mrr" in baseline_data and "img2text_mrr" in baseline_data:
                avg_mrr = (baseline_data['text2img_mrr'] + baseline_data['img2text_mrr']) / 2 * 100
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

        # Find best baseline performance across all models
        best_baseline_text2img = 0
        best_baseline_img2text = 0
        best_text2img_model = ""
        best_img2text_model = ""
        
        for model_name, baseline_data in baselines.items():
            if "text2img_mrr" in baseline_data and "img2text_mrr" in baseline_data:
                t2i_mrr = baseline_data['text2img_mrr'] * 100 if baseline_data['text2img_mrr'] < 1 else baseline_data['text2img_mrr']
                i2t_mrr = baseline_data['img2text_mrr'] * 100 if baseline_data['img2text_mrr'] < 1 else baseline_data['img2text_mrr']
                
                if t2i_mrr > best_baseline_text2img:
                    best_baseline_text2img = t2i_mrr
                    best_text2img_model = model_name
                    
                if i2t_mrr > best_baseline_img2text:
                    best_baseline_img2text = i2t_mrr
                    best_img2text_model = model_name

        if your_text2img > best_baseline_text2img:
            improvements.append(
                f"Text-to-Image retrieval improved by {your_text2img - best_baseline_text2img:.3f}% over {best_text2img_model}")

        if your_img2text > best_baseline_img2text:
            improvements.append(
                f"Image-to-Text retrieval improved by {your_img2text - best_baseline_img2text:.3f}% over {best_img2text_model}")

        if result.text2img_recall_at_1 > 0.01:  # 1% threshold
            improvements.append(f"Strong Recall@1 performance: {result.text2img_recall_at_1 * 100:.2f}%")
            
        # Check subset-specific improvements
        if result.subset_results:
            best_subset_performance = {}
            for subset_name in result.subset_results:
                best_t2i = 0
                best_i2t = 0
                for model_name, baseline_data in baselines.items():
                    if "subsets" in baseline_data and subset_name in baseline_data["subsets"]:
                        subset_data = baseline_data["subsets"][subset_name]
                        if subset_data["text2img_mrr"] > best_t2i:
                            best_t2i = subset_data["text2img_mrr"]
                        if subset_data["img2text_mrr"] > best_i2t:
                            best_i2t = subset_data["img2text_mrr"]
                
                your_subset = result.subset_results[subset_name]
                if your_subset['text2img_mrr'] * 100 > best_t2i:
                    improvements.append(f"Superior {subset_name.replace('_', ' ')} text-to-image performance")
                if your_subset['img2text_mrr'] * 100 > best_i2t:
                    improvements.append(f"Superior {subset_name.replace('_', ' ')} image-to-text performance")

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

        # Add subset evaluation results (like CLIP-BERT methodology)
        if result.subset_results:
            report += "## 📋 Subset Performance (CLIP-BERT Methodology)\n\n"
            
            # Add detailed baseline comparison table
            if "subset_baseline_comparison" in comparison:
                report += "### 🏆 Subset Performance vs. Baselines\n\n"
                
                subset_comparison = comparison["subset_baseline_comparison"]
                
                # Create comparison table for each subset
                for subset_name, subset_data in subset_comparison.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"#### {display_name}\n\n"
                    report += f"**Your Model Rank**: #{subset_data['ranking']['your_rank']} out of {subset_data['ranking']['total_models']} models\n\n"
                    
                    # Performance table
                    report += "| Model | Text→Image MRR | Image→Text MRR | Text→Image Hits@10 | Image→Text Hits@10 |\n"
                    report += "|-------|----------------|----------------|--------------------|--------------------|\n"
                    
                    # Sort by average performance for display
                    rankings = sorted(subset_data['ranking']['all_rankings'], key=lambda x: x[1], reverse=True)
                    
                    for model_name, avg_score in rankings:
                        if model_name == result.model_name:
                            your_perf = subset_data['your_performance']
                            report += f"| **{model_name}** | **{your_perf['text2img_mrr']:.3f}%** | **{your_perf['img2text_mrr']:.3f}%** | **{your_perf['text2img_recall_at_10']:.3f}%** | **{your_perf['img2text_recall_at_10']:.3f}%** |\n"
                        elif model_name in subset_data['baseline_comparison']:
                            baseline_perf = subset_data['baseline_comparison'][model_name]
                            report += f"| {model_name} | {baseline_perf['text2img_mrr']:.3f}% | {baseline_perf['img2text_mrr']:.3f}% | {baseline_perf['text2img_hits_at_10']:.3f}% | {baseline_perf['img2text_hits_at_10']:.3f}% |\n"
                    
                    report += "\n"
            
            # Figure subset breakdown
            figure_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('figure_')}
            if figure_subsets:
                total_figure_samples = sum(metrics['sample_count'] for metrics in figure_subsets.values())
                report += f"### Figure Subset ({total_figure_samples:,} total samples):\n\n"
                
                for subset_name, metrics in figure_subsets.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"**{display_name}**: {metrics['sample_count']:,} samples\n"
                    report += f"- T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr'] * 100:.2f}%)\n"
                    report += f"- I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr'] * 100:.2f}%)\n"
                    report += f"- T2I Recall@1: {metrics['text2img_recall_at_1']:.4f} ({metrics['text2img_recall_at_1'] * 100:.2f}%)\n"
                    report += f"- I2T Recall@1: {metrics['img2text_recall_at_1']:.4f} ({metrics['img2text_recall_at_1'] * 100:.2f}%)\n\n"
            
            # Table subset breakdown
            table_subsets = {k: v for k, v in result.subset_results.items() if k.startswith('table_')}
            if table_subsets:
                total_table_samples = sum(metrics['sample_count'] for metrics in table_subsets.values())
                report += f"### Table Subset ({total_table_samples:,} total samples):\n\n"
                
                for subset_name, metrics in table_subsets.items():
                    display_name = subset_name.replace('_', ' ').title()
                    report += f"**{display_name}**: {metrics['sample_count']:,} samples\n"
                    report += f"- T2I MRR: {metrics['text2img_mrr']:.4f} ({metrics['text2img_mrr'] * 100:.2f}%)\n"
                    report += f"- I2T MRR: {metrics['img2text_mrr']:.4f} ({metrics['img2text_mrr'] * 100:.2f}%)\n"
                    report += f"- T2I Recall@1: {metrics['text2img_recall_at_1']:.4f} ({metrics['text2img_recall_at_1'] * 100:.2f}%)\n"
                    report += f"- I2T Recall@1: {metrics['img2text_recall_at_1']:.4f} ({metrics['img2text_recall_at_1'] * 100:.2f}%)\n\n"
            
            # Expected subset distribution (based on your requirements)
            report += "### Expected SciMMIR Subset Distribution:\n"
            report += "- **Figure Result**: ~9,488 samples (Expected from CLIP-BERT evaluation)\n"
            report += "- **Figure Illustration**: ~1,536 samples\n"
            report += "- **Figure Architecture**: ~467 samples\n"
            report += "- **Table Result**: ~4,229 samples\n"
            report += "- **Table Parameter**: ~543 samples\n"
            report += "- **Total**: 16,263 samples\n\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {save_path}")

        return report
