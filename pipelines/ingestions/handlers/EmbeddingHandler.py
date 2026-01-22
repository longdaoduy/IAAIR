"""
SciBERT Embedding Service for Academic Papers.

This module provides functionality to generate SciBERT embeddings for academic papers
from JSON data files. It automatically detects the most recent input file and
provides configurable embedding generation.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
import glob
from typing import List, Dict, Optional
from datetime import datetime

from models.configurators.SciBERTConfig import SciBERTConfig


class EmbeddingHandler:
    """Service for generating SciBERT embeddings from academic papers."""
    
    def __init__(self, config: Optional[SciBERTConfig] = None):
        """Initialize the SciBERT embedding service.
        
        Args:
            config: SciBERT configuration. If None, loads from environment.
        """
        self.config = config or SciBERTConfig.from_env()
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """Load SciBERT model and tokenizer."""
        print(f"Loading SciBERT model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        
        # Determine device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully on device: {self.device}")

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence-level embeddings."""
        token_embeddings = model_output.last_hidden_state  # (batch, seq_len, hidden)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def find_latest_input_file(self, pattern: str = "enriched_openalex_papers_*.json") -> Optional[str]:
        """Find the most recent input file matching the pattern.
        
        Args:
            pattern: File pattern to search for
            
        Returns:
            Path to the most recent file, or None if no files found
        """
        files = glob.glob(pattern)
        if not files:
            # Try alternative patterns
            alternative_patterns = [
                "openalex_papers_*.json",
                "*papers*.json",
                "papers.json"
            ]
            for alt_pattern in alternative_patterns:
                files = glob.glob(alt_pattern)
                if files:
                    break
        
        if not files:
            return None
        
        # Sort by modification time (most recent first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[0]

    def load_papers(self, input_file: Optional[str] = None) -> List[Dict]:
        """Load papers from JSON file.
        
        Args:
            input_file: Specific input file path. If None, auto-detects latest file.
            
        Returns:
            List of paper data
        """
        if input_file is None:
            input_file = self.find_latest_input_file()
            if input_file is None:
                raise FileNotFoundError("No input JSON files found. Please provide a specific file path.")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        print(f"📄 Loading papers from: {input_file}")
        
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            papers = data
        elif isinstance(data, dict) and "papers" in data:
            papers = data["papers"]
        elif isinstance(data, dict) and "data" in data:
            papers = data["data"]
        else:
            # Assume it's a single paper or list of papers
            papers = [data] if isinstance(data, dict) else data
        
        print(f"✅ Loaded {len(papers)} papers")
        return papers

    def extract_text_for_embedding(self, paper_data: Dict) -> Optional[str]:
        """Extract and concatenate title and abstract safely."""
        # Handle nested structure
        paper = paper_data.get("paper", paper_data)

        # Use .get(key, "") to default to an empty string if key is missing
        # Then use 'or ""' to handle cases where the value is explicitly None
        title = (paper.get("title") or "").strip()
        abstract = (paper.get("abstract") or "").strip()

        # Filter out common "empty" strings
        invalid_abstracts = ["no abstract", "n/a", "null", "none", ""]
        if abstract.lower() in invalid_abstracts:
            abstract = ""

        # Return concatenated string
        if title and abstract:
            return f"TITLE: {title} [SEP] ABSTRACT: {abstract}"
        elif title:
            return f"TITLE: {title}"
        elif abstract:
            return f"ABSTRACT: {abstract}"

        return None

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.config.max_sequence_length
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        embedding = self._mean_pooling(outputs, inputs["attention_mask"])
        return embedding.squeeze().cpu().tolist()

    def process_papers(self, input_file: Optional[str] = None, output_file: Optional[str] = None) -> str:
        """Process papers and generate embeddings.
        
        Args:
            input_file: Input JSON file path. Auto-detects if None.
            output_file: Output JSON file path. Auto-generates if None.
            
        Returns:
            Path to output file
        """
        # Load papers
        papers = self.load_papers(input_file)
        
        # Generate output filename if not provided
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"paper_embeddings_scibert_{timestamp}.json"
        
        results = []
        skipped_count = 0
        
        print(f"🧠 Generating embeddings with batch size: {self.config.batch_size}")
        
        for i, paper_data in enumerate(tqdm(papers, desc="Processing papers")):
            # Handle nested paper structure
            if "paper" in paper_data:
                paper = paper_data["paper"]
            else:
                paper = paper_data
            
            paper_id = paper.get("id", f"paper_{i}")
            text = self.extract_text_for_embedding(paper_data)
            
            if not text:
                skipped_count += 1
                continue
            
            try:
                embedding = self.generate_embedding(text)
                
                results.append({
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "embedding": embedding,
                    "embedding_source": "abstract" if paper.get("abstract") else "title",
                    "embedding_dim": len(embedding)
                })
            except Exception as e:
                print(f"❌ Error processing paper {paper_id}: {e}")
                skipped_count += 1
                continue
        
        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\nEmbedding Generation Summary:")
        print(f"   Total papers: {len(papers)}")
        print(f"   Embeddings generated: {len(results)}")
        print(f"   Papers skipped: {skipped_count}")
        print(f"   Success rate: {len(results)/len(papers)*100:.1f}%")
        print(f"   Embedding dimension: {len(results[0]['embedding']) if results else 0}")
        print(f"   Output saved to: {output_file}")
        
        return output_file
