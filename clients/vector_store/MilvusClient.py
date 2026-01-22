"""
Zilliz Cloud Data Upload Service for SciBERT Embeddings.

This module uploads SciBERT paper embeddings to Zilliz Cloud (managed Milvus service)
with proper authentication and error handling.
"""

import json
import os
import glob
from typing import List, Dict, Optional, Any
# from google.colab import userdata
from tqdm import tqdm

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, AnnSearchRequest, RRFRanker
)

from models.configurators.VectorDBConfig import VectorDBConfig
from pipelines.ingestions.handlers.EmbeddingHandler import EmbeddingHandler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix


class MilvusClient:
    """Service for uploading embeddings to Zilliz Cloud."""

    def __init__(self, config: Optional[VectorDBConfig] = None):
        """Initialize Zilliz upload service.
        
        Args:
            config: Vector database configuration. If None, loads from environment.
        """
        self.config = config or VectorDBConfig.from_env()
        self.collection = None
        self.is_connected = False
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        self.is_tfidf_fitted = False

    def connect(self) -> bool:
        try:
            print("Connecting to Zilliz Cloud...")
            token = self.config.token
            uri = self.config.uri

            if not isinstance(uri, str) or not isinstance(token, str):
                raise RuntimeError("ZILLIZ_URI or ZILLIZ_TOKEN invalid")
            print(uri)
            print(token)

            connections.connect(
                alias="default",
                uri=uri,
                token=token
            )

            self.is_connected = True
            print("✅ Connected to Zilliz Cloud")
            return True

        except Exception as e:
            print(f"❌ Failed to connect to Zilliz: {e}")
            return False

    def create_collection_schema(self, embedding_dim: int) -> CollectionSchema:
        """Create collection schema for paper embeddings with hybrid search support.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            
        Returns:
            Collection schema
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="dense_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Hybrid embeddings for academic papers (dense + sparse)"
        )

        return schema

    def create_collection(self, embedding_dim: int, force_recreate: bool = False) -> bool:
        """Create or get existing collection.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            force_recreate: Whether to drop existing collection and recreate
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection_name = self.config.collection_name

            # Check if collection exists
            if utility.has_collection(collection_name):
                print(f"📋 Using existing collection: {collection_name}")
                self.collection = Collection(collection_name)
                return True

            # Create new collection
            print(f"🏗️ Creating new collection: {collection_name}")
            schema = self.create_collection_schema(embedding_dim)

            self.collection = Collection(
                name=collection_name,
                schema=schema
            )

            # Create index for dense vector similarity search
            dense_index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist}
            }

            # Create index for sparse vector search
            sparse_index_params = {
                "metric_type": "IP",  # Inner Product for sparse vectors
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }

            print(f"🔍 Creating dense vector index with params: {dense_index_params}")
            self.collection.create_index(
                field_name="dense_embedding",
                index_params=dense_index_params
            )

            print(f"🔍 Creating sparse vector index with params: {sparse_index_params}")
            self.collection.create_index(
                field_name="sparse_embedding",
                index_params=sparse_index_params
            )

            print("✅ Collection created successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to create collection: {e}")
            return False

    def find_latest_embedding_file(self) -> Optional[str]:
        """Find the most recent embedding file.
        
        Returns:
            Path to the most recent embedding file, or None if not found
        """
        patterns = [
            "paper_embeddings_scibert_*.json",
            "paper_embeddings_*.json",
            "*embeddings*.json"
        ]

        for pattern in patterns:
            files = glob.glob(pattern)
            if files:
                # Sort by modification time (most recent first)
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0]

        return None

    def generate_sparse_embedding(self, text: str) -> Dict[int, float]:
        """Generate sparse embedding using TF-IDF.
        
        Args:
            text: Input text
            
        Returns:
            Sparse vector as dictionary {index: value}
        """
        if not self.is_tfidf_fitted:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf_vectorizer first.")
        
        # Transform text to TF-IDF vector
        tfidf_vector = self.tfidf_vectorizer.transform([text])
        
        # Convert to sparse dictionary format expected by Milvus
        sparse_dict = {}
        coo_matrix = tfidf_vector.tocoo()
        
        for i, j, val in zip(coo_matrix.row, coo_matrix.col, coo_matrix.data):
            if val > 0.01:  # Filter out very small values
                sparse_dict[j] = float(val)
        
        return sparse_dict

    def fit_tfidf_vectorizer(self, texts: List[str]):
        """Fit TF-IDF vectorizer on the corpus.
        
        Args:
            texts: List of texts to fit the vectorizer
        """
        print("🔧 Fitting TF-IDF vectorizer...")
        self.tfidf_vectorizer.fit(texts)
        self.is_tfidf_fitted = True
        print(f"✅ TF-IDF vectorizer fitted with {len(self.tfidf_vectorizer.vocabulary_)} features")

    def load_embeddings(self, embedding_file: Optional[str] = None) -> List[Dict]:
        """Load embedding data from JSON file.
        
        Args:
            embedding_file: Path to embedding file. Auto-detects if None.
            
        Returns:
            List of embedding data
        """
        if embedding_file is None:
            embedding_file = self.find_latest_embedding_file()
            if embedding_file is None:
                raise FileNotFoundError("No embedding files found. Please run SciBERT embedding generation first.")

        if not os.path.exists(embedding_file):
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")

        print(f"📄 Loading embeddings from: {embedding_file}")

        with open(embedding_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"✅ Loaded {len(data)} embeddings")
        return data

    def prepare_batch_data(self, embeddings: List[Dict], papers_data: List[Dict], batch_size: int = 1000) -> List[List[Any]]:
        """Prepare embedding data for batch insertion with hybrid search support.
        
        Args:
            embeddings: List of embedding data
            papers_data: Original papers data for text extraction
            batch_size: Number of records per batch
            
        Returns:
            List of batches, each containing field data
        """
        # Create lookup for paper data
        paper_lookup = {paper.get('id', paper.get('paper_id')): paper for paper in papers_data}
        
        # Collect all texts for TF-IDF fitting
        all_texts = []
        for item in embeddings:
            paper_id = item['paper_id']
            paper = paper_lookup.get(paper_id, {})
            
            # Extract text for sparse embedding
            text_for_sparse = ""
            if 'abstract' in paper and paper['abstract']:
                text_for_sparse = paper['abstract']
            elif 'title' in paper and paper['title']:
                text_for_sparse = paper['title']
            
            if text_for_sparse:
                all_texts.append(text_for_sparse)
        
        # Fit TF-IDF vectorizer if not already fitted
        if not self.is_tfidf_fitted and all_texts:
            self.fit_tfidf_vectorizer(all_texts)
        
        batches = []
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            # Prepare field data for hybrid schema
            ids = []
            titles = []
            abstracts = []
            dense_embeddings = []
            sparse_embeddings = []
            
            for item in batch:
                paper_id = item['paper_id']
                paper = paper_lookup.get(paper_id, {})
                
                ids.append(paper_id)
                titles.append(paper.get('title', '')[:1000])  # Truncate to max length
                abstracts.append(paper.get('abstract', '')[:8000])  # Truncate to max length
                
                # Dense embedding
                if 'embedding' in item:
                    dense_embeddings.append(item['embedding'])
                else:
                    # Default to zero vector if no embeddings
                    dim = 768  # Default SciBERT dimension
                    dense_embeddings.append([0.0] * dim)
                
                # Sparse embedding
                text_for_sparse = paper.get('abstract', '') or paper.get('title', '')
                if text_for_sparse and self.is_tfidf_fitted:
                    sparse_embedding = self.generate_sparse_embedding(text_for_sparse)
                else:
                    sparse_embedding = {0: 0.01}  # Default sparse vector
                
                sparse_embeddings.append(sparse_embedding)
            
            batch_data = [ids, titles, abstracts, dense_embeddings, sparse_embeddings]
            batches.append(batch_data)
        
        return batches

    def upload_embeddings(self, embedding_file: Optional[str] = None, papers_data: Optional[List[Dict]] = None, batch_size: int = 1000) -> bool:
        """Upload embeddings to Zilliz Cloud with hybrid search support.
        
        Args:
            embedding_file: Path to embedding file. Auto-detects if None.
            papers_data: Original papers data for text extraction
            batch_size: Number of records per batch
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load embeddings
            embeddings = self.load_embeddings(embedding_file)

            if not embeddings:
                print("❌ No embeddings to upload")
                return False
            
            if not papers_data:
                print("⚠️ No papers data provided, sparse embeddings will be minimal")
                papers_data = []

            # Get embedding dimension
            first_item = embeddings[0]
            if 'embedding' in first_item:
                embedding_dim = len(first_item['embedding'])
            else:
                embedding_dim = 768  # Default SciBERT dimension
            
            print(f"📏 Dense embedding dimension: {embedding_dim}")

            # Create collection
            if not self.create_collection(embedding_dim):
                return False

            # Prepare batch data with hybrid embeddings
            batches = self.prepare_batch_data(embeddings, papers_data, batch_size)

            # Upload batches
            print(f"📤 Uploading {len(embeddings)} hybrid embeddings in {len(batches)} batches...")

            total_inserted = 0
            for i, batch_data in enumerate(tqdm(batches, desc="Uploading batches")):
                try:
                    insert_result = self.collection.insert(batch_data)
                    batch_count = len(batch_data[0])  # Number of IDs in batch
                    total_inserted += batch_count

                    if (i + 1) % 10 == 0:  # Flush every 10 batches
                        self.collection.flush()

                except Exception as e:
                    print(f"❌ Error uploading batch {i + 1}: {e}")
                    continue

            # Final flush
            self.collection.flush()

            # Load collection for search
            self.collection.load()

            print(f"\n📊 Upload Summary:")
            print(f"   Total records: {len(embeddings)}")
            print(f"   Successfully inserted: {total_inserted}")
            print(f"   Success rate: {total_inserted / len(embeddings) * 100:.1f}%")
            print(f"   Collection name: {self.config.collection_name}")
            print("✅ Hybrid upload completed successfully")

            return True

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False

    def search_similar_papers(self, query_text: str, top_k: int = 10, use_hybrid: bool = True) -> List[Dict]:
        """Search for similar papers using hybrid search (dense + sparse) or dense-only search.
        
        Args:
            query_text: Text query to search for similar papers
            top_k: Number of top results to return
            use_hybrid: Whether to use hybrid search or dense-only search
            
        Returns:
            List of similar papers with scores and IDs
        """
        try:
            if not self.collection:
                self.collection = Collection(self.config.collection_name)
                self.collection.load()

            # Generate embedding for the query text
            embedding_handler = EmbeddingHandler()
            query_embedding = embedding_handler.generate_embedding(query_text)
            
            if query_embedding is None:
                print("❌ Failed to generate query embedding")
                return []

            if use_hybrid and self.is_tfidf_fitted:
                return self._hybrid_search(query_text, query_embedding, top_k)
            else:
                return self._dense_search(query_embedding, top_k)

        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []

    def _dense_search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        """Perform dense vector search only.
        
        Args:
            query_embedding: Dense query embedding
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        search_params = {
            "metric_type": self.config.metric_type,
            "params": {"nprobe": min(32, max(1, top_k))}
        }

        results = self.collection.search(
            data=[query_embedding],
            anns_field="dense_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["id", "title", "abstract"]
        )

        # Format results
        formatted_results = []
        if results and len(results[0]) > 0:
            for hit in results[0]:
                formatted_results.append({
                    "paper_id": hit.entity.get("id"),
                    "title": hit.entity.get("title"),
                    "abstract": hit.entity.get("abstract"),
                    "similarity_score": float(hit.score),
                    "distance": float(hit.distance),
                    "search_type": "dense_only"
                })

        print(f"🔍 Dense search found {len(formatted_results)} similar papers")
        return formatted_results

    def _hybrid_search(self, query_text: str, dense_embedding: List[float], top_k: int) -> List[Dict]:
        """Perform hybrid search combining dense and sparse vectors.
        
        Args:
            query_text: Original query text
            dense_embedding: Dense query embedding
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        try:
            # Generate sparse embedding for the query
            sparse_embedding = self.generate_sparse_embedding(query_text)
            
            # Create search requests for both dense and sparse
            dense_search_request = AnnSearchRequest(
                data=[dense_embedding],
                anns_field="dense_embedding",
                param={
                    "metric_type": self.config.metric_type,
                    "params": {"nprobe": min(32, max(1, top_k * 2))}
                },
                limit=top_k * 2  # Get more candidates for reranking
            )
            
            sparse_search_request = AnnSearchRequest(
                data=[sparse_embedding],
                anns_field="sparse_embedding",
                param={
                    "metric_type": "IP",
                    "params": {}
                },
                limit=top_k * 2  # Get more candidates for reranking
            )
            
            # Perform hybrid search with RRF (Reciprocal Rank Fusion) reranking
            hybrid_results = self.collection.hybrid_search(
                reqs=[dense_search_request, sparse_search_request],
                rerank=RRFRanker(),
                limit=top_k,
                output_fields=["id", "title", "abstract"]
            )
            
            # Format results
            formatted_results = []
            if hybrid_results and len(hybrid_results[0]) > 0:
                for hit in hybrid_results[0]:
                    formatted_results.append({
                        "paper_id": hit.entity.get("id"),
                        "title": hit.entity.get("title"),
                        "abstract": hit.entity.get("abstract"),
                        "similarity_score": float(hit.score),
                        "distance": float(hit.distance),
                        "search_type": "hybrid"
                    })
            
            print(f"🔍 Hybrid search found {len(formatted_results)} similar papers")
            return formatted_results
            
        except Exception as e:
            print(f"⚠️ Hybrid search failed, falling back to dense search: {e}")
            return self._dense_search(dense_embedding, top_k)

    def verify_upload(self) -> bool:
        """Verify the uploaded data.
        
        Returns:
            True if verification successful, False otherwise
        """
        try:
            if not self.collection:
                print("❌ No collection available for verification")
                return False

            # Get collection statistics
            self.collection.load()
            num_entities = self.collection.num_entities

            print(f"\n🔍 Collection Verification:")
            print(f"   Collection name: {self.config.collection_name}")
            print(f"   Total entities: {num_entities}")

            # Test search functionality
            if num_entities > 0:
                # Test dense search
                try:
                    dense_search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": 10}}
                    
                    # Use zero vector for test search (just to verify search works)
                    dense_results = self.collection.search(
                        data=[[0.0] * 768],  # Assume 768-dim embeddings
                        anns_field="dense_embedding",
                        param=dense_search_params,
                        limit=1
                    )

                    if dense_results:
                        print(f"   Dense search test: ✅ Working")
                    else:
                        print(f"   Dense search test: ⚠️ No results")
                        
                except Exception as e:
                    print(f"   Dense search test: ❌ Failed ({e})")

                # Test sparse search
                try:
                    sparse_search_params = {"metric_type": "IP", "params": {}}
                    
                    # Use minimal sparse vector for test
                    sparse_results = self.collection.search(
                        data=[{0: 0.1}],  # Minimal sparse vector
                        anns_field="sparse_embedding",
                        param=sparse_search_params,
                        limit=1
                    )

                    if sparse_results:
                        print(f"   Sparse search test: ✅ Working")
                    else:
                        print(f"   Sparse search test: ⚠️ No results")
                        
                except Exception as e:
                    print(f"   Sparse search test: ❌ Failed ({e})")

            return True

        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from Zilliz."""
        try:
            connections.disconnect("default")
            self.is_connected = False
            print("🔌 Disconnected from Zilliz")
        except:
            pass