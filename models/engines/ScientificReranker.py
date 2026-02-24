from typing import List
import logging
import torch
from models.entities.retrieval.SearchResult import SearchResult
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class ScientificReranker:
    """Rerank results using BGE reranker model from Hugging Face."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = 'BAAI/bge-reranker-base'
        self.max_length = 512
        self.initialized = False
        
        # Initialize model on first use to avoid startup delays
        logger.info(f"ScientificReranker initialized - will load {self.model_name} on first use")

    def _initialize_model(self):
        """Initialize the BGE reranker model."""
        if self.initialized:
            return
            
        try:
            logger.info(f"Loading BGE reranker model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()
            self.initialized = True
            
            logger.info(f"BGE reranker model loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import transformers: {e}")
            logger.info("Install transformers with: pip install transformers torch")
            self.initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize BGE reranker: {e}")
            self.initialized = False

    async def rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results using BGE reranker model."""
        if not results:
            return results
            
        # Initialize model if needed
        self._initialize_model()
        
        if not self.initialized:
            logger.warning("BGE reranker not available, falling back to basic reranking")
            return self._fallback_rerank(results)
        
        try:
            logger.info(f"Reranking {len(results)} results using BGE reranker")
            
            # Prepare query-document pairs for reranking
            pairs = []
            for result in results:
                # Create document text from available fields
                doc_text = self._create_document_text(result)
                pairs.append([query, doc_text])
            
            # Score pairs using BGE reranker
            scores = self._score_pairs(pairs)
            
            # Update results with rerank scores
            for i, result in enumerate(results):
                rerank_score = scores[i] if i < len(scores) else 0.0
                result.rerank_score = rerank_score
                result.confidence_scores.update({
                    'bge_rerank_score': rerank_score,
                    'original_relevance': result.relevance_score
                })
            
            # Sort by rerank score
            results.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            
            logger.info(f"Reranking completed. Score range: {min(scores):.3f} - {max(scores):.3f}")
            return results
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return self._fallback_rerank(results)

    def _create_document_text(self, result: SearchResult) -> str:
        """Create document text from SearchResult for reranking."""
        parts = []
        
        # Add title (most important)
        if result.title:
            parts.append(f"Title: {result.title}")
        
        # Add authors
        if result.authors:
            authors_str = ", ".join(result.authors[:5])  # Limit to first 5 authors
            if len(result.authors) > 5:
                authors_str += " et al."
            parts.append(f"Authors: {authors_str}")
        
        # Add venue if available
        if result.venue:
            parts.append(f"Venue: {result.venue}")
        
        # Add abstract (truncated to fit within token limit)
        if result.abstract:
            # Reserve space for title, authors, venue (~100 tokens)
            # Use remaining space for abstract
            max_abstract_chars = (self.max_length - 100) * 4  # Rough chars per token
            abstract = result.abstract[:max_abstract_chars]
            if len(result.abstract) > max_abstract_chars:
                abstract += "..."
            parts.append(f"Abstract: {abstract}")
        
        return " | ".join(parts)

    def _score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """Score query-document pairs using BGE reranker."""
        if not pairs:
            return []
            
        try:
            # Batch processing for efficiency
            batch_size = 8  # Adjust based on GPU memory
            all_scores = []
            
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                batch_scores = self._score_batch(batch_pairs)
                all_scores.extend(batch_scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Error scoring pairs: {e}")
            return [0.0] * len(pairs)

    def _score_batch(self, batch_pairs: List[List[str]]) -> List[float]:
        """Score a batch of query-document pairs."""
        try:
            # Tokenize the batch
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            # Get scores from model
            with torch.no_grad():
                outputs = self.model(**inputs)
                # BGE reranker outputs logits, apply sigmoid to get scores
                scores = torch.sigmoid(outputs.logits).squeeze(-1)
                
                # Convert to list of floats
                if scores.dim() == 0:  # Single score
                    return [scores.item()]
                else:
                    return scores.cpu().tolist()
                    
        except Exception as e:
            logger.error(f"Error scoring batch: {e}")
            return [0.0] * len(batch_pairs)

    def _fallback_rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """Fallback reranking when BGE model is not available."""
        logger.info("Using fallback reranking based on original relevance scores")
        
        for result in results:
            # Use original relevance score as rerank score
            result.rerank_score = result.relevance_score
            result.confidence_scores.update({
                'fallback_rerank': True,
                'original_relevance': result.relevance_score
            })
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results

    def get_model_info(self) -> dict:
        """Get information about the reranker model."""
        return {
            'model_name': self.model_name,
            'device': str(self.device),
            'initialized': self.initialized,
            'max_length': self.max_length,
            'cuda_available': torch.cuda.is_available()
        }