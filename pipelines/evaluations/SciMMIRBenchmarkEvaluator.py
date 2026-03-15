"""
SciMMIR benchmark integration for scientific multi-modal information retrievals.

This module integrates the SciMMIR benchmark with your existing IAAIR system
to evaluate multi-modal retrievals performance using scientific image-text pairs.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from utils.metrics import calculate_retrieval_metrics
from datetime import datetime
from clients.huggingface.CLIPClient import CLIPClient
from clients.huggingface.SciBERTClient import SciBERTClient
from clients.milvus.MilvusClient import MilvusClient
from pipelines.evaluations.SciMMIRDataLoader import SciMMIRDataLoader
from models.entities.evaluations.SciMMIRSample import SciMMIRSample
from models.entities.evaluations.SciMMIRBenchmarkResult import SciMMIRBenchmarkResult

import psutil
import gc
import numpy as np

logger = logging.getLogger(__name__)


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
        optimal_batch_size = max(optimal_batch_size, 8)  # Min 8

        self.logger.info(f"Memory optimization: Available={available_mb:.0f}MB, "
                         f"Target={target_memory_mb:.0f}MB, Batch size={optimal_batch_size}")

        return optimal_batch_size

    def run_benchmark(self,
                      samples: List[SciMMIRSample],
                      model_name: str = "IAAIR-Zero-Shot",
                      top_k: int = 10,
                      evaluate_subsets: bool = True,
                      batch_size: Optional[int] = None) -> SciMMIRBenchmarkResult:
        """Run complete SciMMIR benchmark evaluations.
        
        Args:
            samples: List of SciMMIR samples
            model_name: Name of the model being evaluated
            top_k: Top-k for evaluations metrics
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

        self.logger.info("Starting SciMMIR benchmark evaluations")
        # Text-to-Image retrievals with batch processing
        text2img_metrics = self._evaluate_text_to_image(text_embeddings, image_embeddings, batch_size)

        # Image-to-Text retrievals with batch processing
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
                    'clip_model': getattr(self.clip_client.config, 'model_name', 'unknown') if hasattr(self.clip_client,
                                                                                                       'config') and self.clip_client.config else 'unknown',
                    'scibert_model': getattr(self.scibert_client.config, 'model_name', 'unknown') if hasattr(
                        self.scibert_client, 'config') and self.scibert_client.config else 'unknown'
                }
            }
        )

        return result

    def _generate_embeddings(self, samples: List[SciMMIRSample], batch_size: int = 32) -> Tuple[
        List[List[float]], List[List[float]]]:
        """Generate text and image embeddings for all samples with memory optimization.
        
        Args:
            samples: List of SciMMIR samples
            batch_size: Number of samples to process at once (reduce if OOM)
        """

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
                        emb = self.scibert_client.generate_text_embedding(text) if self.scibert_client else None
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
            self.logger.info(
                f"Embedding generation progress: {progress:.1f}% ({len(text_embeddings)} samples processed)")

        self.logger.info(f"Completed embedding generation for {len(text_embeddings)} samples")
        return text_embeddings, image_embeddings

    @staticmethod
    def _evaluate_text_to_image(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]],
                                batch_size: int = 100) -> Dict[str, float]:
        """Highly optimized evaluations using Matrix Multiplication."""

        # 1. Convert to NumPy arrays and Normalize once (Pre-calculation)
        # Normalizing makes cosine similarity identical to a simple dot product
        img_array = np.array(image_embeddings)
        img_norms = np.linalg.norm(img_array, axis=1, keepdims=True)
        # Avoid division by zero
        img_array = np.divide(img_array, img_norms, out=np.zeros_like(img_array), where=img_norms != 0)

        txt_array = np.array(text_embeddings)
        txt_norms = np.linalg.norm(txt_array, axis=1, keepdims=True)
        txt_array = np.divide(txt_array, txt_norms, out=np.zeros_like(txt_array), where=txt_norms != 0)

        ranks = []
        total_queries = len(txt_array)

        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)
            batch_txt = txt_array[batch_start:batch_end]

            # 2. MATRIX MULTIPLICATION: This replaces your inner loops
            # (Batch_Size x Dim) @ (Dim x Total_Images) -> (Batch_Size x Total_Images)
            sim_matrix = np.matmul(batch_txt, img_array.T)

            # 3. Vectorized Ranking
            # For each row, find the rank of the element at the 'correct' index
            for i, global_idx in enumerate(range(batch_start, batch_end)):
                scores = sim_matrix[i]
                # argsort sorts ascending, so we subtract from total to get descending rank
                # This finds how many scores are better than our target score
                target_score = scores[global_idx]
                rank = np.sum(scores > target_score) + 1
                ranks.append(rank)

        return calculate_retrieval_metrics(ranks)

    @staticmethod
    def _evaluate_image_to_text(text_embeddings: List[List[float]],
                                image_embeddings: List[List[float]],
                                batch_size: int = 100) -> Dict[str, float]:
        """Highly optimized Image-to-Text evaluations using Matrix Ops."""

        # 1. Convert to NumPy and Normalize (L2 Norm = 1.0)
        # This turns Cosine Similarity into a simple Dot Product
        txt_array = np.array(text_embeddings)
        txt_norms = np.linalg.norm(txt_array, axis=1, keepdims=True)
        # Handle zero-vectors to avoid NaN
        txt_array = np.divide(txt_array, txt_norms, out=np.zeros_like(txt_array), where=txt_norms != 0)

        img_array = np.array(image_embeddings)
        img_norms = np.linalg.norm(img_array, axis=1, keepdims=True)
        img_array = np.divide(img_array, img_norms, out=np.zeros_like(img_array), where=img_norms != 0)

        ranks = []
        total_queries = len(img_array)

        for batch_start in range(0, total_queries, batch_size):
            batch_end = min(batch_start + batch_size, total_queries)

            # Take a batch of query images
            batch_img = img_array[batch_start:batch_end]

            # 2. CALCULATE ALL SIMILARITIES AT ONCE
            # Resulting shape: (batch_size, total_text_embeddings)
            sim_matrix = np.dot(batch_img, txt_array.T)

            # 3. VECTORIZED RANKING
            for i, global_idx in enumerate(range(batch_start, batch_end)):
                # scores[j] is the similarity between current image and text j
                scores = sim_matrix[i]

                # The 'correct' text embedding is at the same index as the image
                target_score = scores[global_idx]

                # Rank = (Number of elements strictly better than our target) + 1
                rank = np.sum(scores > target_score) + 1
                ranks.append(rank)

            # Basic logging
            if total_queries > 1000 and batch_start % (batch_size * 10) == 0:
                print(f"Image-to-Text progress: {(batch_end / total_queries) * 100:.1f}%")

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
            if len(indices) < 2:  # Need at least 2 samples for meaningful evaluations
                self.logger.warning(f"Skipping subset {subset_name} with only {len(indices)} samples")
                continue

            self.logger.info(f"Evaluating subset {subset_name} with {len(indices)} samples...")

            # Extract embeddings for this subset
            subset_text_embeddings = [text_embeddings[i] for i in indices]
            subset_image_embeddings = [image_embeddings[i] for i in indices]

            # Use smaller batch size for subsets to be more memory efficient
            subset_batch_size = min(batch_size, max(10, len(indices) // 4))

            # Evaluate text-to-image for this subset with batch processing
            text2img_metrics = self._evaluate_text_to_image(subset_text_embeddings, subset_image_embeddings,
                                                            subset_batch_size)

            # Evaluate image-to-text for this subset with batch processing
            img2text_metrics = self._evaluate_image_to_text(subset_text_embeddings, subset_image_embeddings,
                                                            subset_batch_size)

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

            self.logger.info(
                f"Subset {subset_name}: T2I MRR={text2img_metrics['mrr']:.4f}, I2T MRR={img2text_metrics['mrr']:.4f}")

            # Clean up subset embeddings and force garbage collection
            del subset_text_embeddings, subset_image_embeddings
            gc.collect()

        return subset_results