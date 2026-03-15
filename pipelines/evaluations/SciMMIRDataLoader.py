import logging
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image
import random
from models.entities.evaluations.SciMMIRSample import SciMMIRSample
import pandas as pd
import io
import requests

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

                self.logger.info(f"Successfully downloaded {filename} ({downloaded_size / (1024 * 1024):.1f} MB)")
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

        self.logger.info(
            f"Successfully loaded {len(all_samples)} total samples from {len(parquet_files)} parquet files")
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
            if any(term in text_lower for term in ['result', 'performance', 'accuracy', 'comparison', 'evaluations']):
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