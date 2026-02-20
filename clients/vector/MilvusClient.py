"""
Zilliz Cloud Data Upload Service for SciBERT Embeddings.

This module uploads SciBERT paper embeddings to Zilliz Cloud (managed Milvus service)
with proper authentication and error handling.
"""

import json
import os
import glob
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, AnnSearchRequest, RRFRanker
)

from models.configurators.VectorDBConfig import VectorDBConfig
from sklearn.feature_extraction.text import TfidfVectorizer


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

    def create_figures_collection_schema(self) -> CollectionSchema:
        """Create collection schema for figures with specific fields.
        
        Returns:
            Collection schema for figures
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # SciBERT dimension
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # CLIP dimension
            FieldSchema(name="sparse_description_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Figures collection with description and image embeddings"
        )

        return schema

    def create_tables_collection_schema(self) -> CollectionSchema:
        """Create collection schema for tables with specific fields.

        Returns:
            Collection schema for tables
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=200, is_primary=True),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=8000),
            FieldSchema(name="description_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # SciBERT dimension
            FieldSchema(name="sparse_description_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Tables collection with description embeddings"
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

            # Check if collection exists and has compatible schema
            if utility.has_collection(collection_name):
                self.collection = Collection(collection_name)

                # Check if the existing collection has the hybrid schema (title, abstract, dense_embedding, sparse_embedding)
                schema = self.collection.schema
                field_names = [field.name for field in schema.fields]

                required_fields = ['id', 'title', 'abstract', 'dense_embedding', 'sparse_embedding']
                has_hybrid_schema = all(field in field_names for field in required_fields)

                if has_hybrid_schema:
                    print(f"📋 Using existing collection with hybrid schema: {collection_name}")
                    return True
                else:
                    print(f"⚠️  Existing collection has incompatible schema. Required fields: {required_fields}")
                    print(f"⚠️  Existing fields: {field_names}")

                    if force_recreate:
                        print(f"🗑️  Dropping existing collection to recreate with hybrid schema...")
                        utility.drop_collection(collection_name)
                    else:
                        print(f"❌ Set force_recreate=True to automatically recreate the collection with hybrid schema")
                        return False

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

    def create_figures_collection(self, force_recreate: bool = False) -> bool:
        """Create or get existing figures collection.

        Args:
            force_recreate: Whether to drop existing collection and recreate

        Returns:
            True if successful, False otherwise
        """
        try:
            collection_name = "figures_collection"

            # Check if collection exists
            if utility.has_collection(collection_name):
                figures_collection = Collection(collection_name)

                # Check if the existing collection has the correct schema
                schema = figures_collection.schema
                field_names = [field.name for field in schema.fields]

                required_fields = ['id', 'paper_id', 'description', 'description_embedding', 'image_embedding',
                                   'sparse_description_embedding']
                has_correct_schema = all(field in field_names for field in required_fields)

                if has_correct_schema:
                    print(f"📋 Using existing figures collection: {collection_name}")
                    return True
                else:
                    print(
                        f"⚠️  Existing figures collection has incompatible schema. Required fields: {required_fields}")
                    print(f"⚠️  Existing fields: {field_names}")

                    if force_recreate:
                        print(f"🗑️  Dropping existing figures collection to recreate...")
                        utility.drop_collection(collection_name)
                    else:
                        print(f"❌ Set force_recreate=True to automatically recreate the figures collection")
                        return False

            # Create new figures collection
            print(f"🏗️ Creating new figures collection: {collection_name}")
            schema = self.create_figures_collection_schema()

            figures_collection = Collection(
                name=collection_name,
                schema=schema
            )

            # Create indexes for vector fields
            # Index for description embedding (dense)
            desc_index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            print(f"🔍 Creating description embedding index...")
            figures_collection.create_index(
                field_name="description_embedding",
                index_params=desc_index_params
            )

            # Index for image embedding (dense)
            img_index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            print(f"🔍 Creating image embedding index...")
            figures_collection.create_index(
                field_name="image_embedding",
                index_params=img_index_params
            )

            # Index for sparse description embedding
            sparse_index_params = {
                "metric_type": "IP",  # Inner Product for sparse vectors
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }

            print(f"🔍 Creating sparse description embedding index...")
            figures_collection.create_index(
                field_name="sparse_description_embedding",
                index_params=sparse_index_params
            )

            # Load collection for immediate use
            figures_collection.load()

            print("✅ Figures collection created successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to create figures collection: {e}")
            return False

    def create_tables_collection(self, force_recreate: bool = False) -> bool:
        """Create or get existing tables collection.

        Args:
            force_recreate: Whether to drop existing collection and recreate

        Returns:
            True if successful, False otherwise
        """
        try:
            collection_name = "tables_collection"

            # Check if collection exists
            if utility.has_collection(collection_name):
                tables_collection = Collection(collection_name)

                # Check if the existing collection has the correct schema
                schema = tables_collection.schema
                field_names = [field.name for field in schema.fields]

                required_fields = ['id', 'paper_id', 'description', 'description_embedding',
                                   'sparse_description_embedding']
                has_correct_schema = all(field in field_names for field in required_fields)

                if has_correct_schema:
                    print(f"📋 Using existing tables collection: {collection_name}")
                    return True
                else:
                    print(f"⚠️  Existing tables collection has incompatible schema. Required fields: {required_fields}")
                    print(f"⚠️  Existing fields: {field_names}")

                    if force_recreate:
                        print(f"🗑️  Dropping existing tables collection to recreate...")
                        utility.drop_collection(collection_name)
                    else:
                        print(f"❌ Set force_recreate=True to automatically recreate the tables collection")
                        return False

            # Create new tables collection
            print(f"🏗️ Creating new tables collection: {collection_name}")
            schema = self.create_tables_collection_schema()

            tables_collection = Collection(
                name=collection_name,
                schema=schema
            )

            # Create indexes for vector fields
            # Index for description embedding (dense)
            desc_index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            print(f"🔍 Creating description embedding index...")
            tables_collection.create_index(
                field_name="description_embedding",
                index_params=desc_index_params
            )

            # Index for sparse description embedding
            sparse_index_params = {
                "metric_type": "IP",  # Inner Product for sparse vectors
                "index_type": "SPARSE_INVERTED_INDEX",
                "params": {"drop_ratio_build": 0.2}
            }

            print(f"🔍 Creating sparse description embedding index...")
            tables_collection.create_index(
                field_name="sparse_description_embedding",
                index_params=sparse_index_params
            )

            # Load collection for immediate use
            tables_collection.load()

            print("✅ Tables collection created successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to create tables collection: {e}")
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

    def prepare_batch_data(self, embeddings: List[Dict], papers_data: List[Dict], batch_size: int = 1000) -> List[
        List[Any]]:
        """Prepare embedding data for batch insertion with hybrid search support."""

        paper_lookup = {}

        for item in papers_data:
            # 1. Access the 'paper' object/dict from the item
            paper = item.get('paper') if isinstance(item, dict) else getattr(item, 'paper', None)
            if not paper:
                continue

            # 2. Extract ID safely (handles both Object and Dict)
            p_id = getattr(paper, 'id', None) if not isinstance(paper, dict) else paper.get('id')

            if p_id:
                # Helper to get attributes regardless of type
                def get_val(obj, attr, default=""):
                    val = getattr(obj, attr, default) if not isinstance(obj, dict) else obj.get(attr, default)
                    return val if val is not None else default

                paper_lookup[p_id] = {
                    'id': p_id,
                    'title': get_val(paper, 'title'),
                    'abstract': get_val(paper, 'abstract'),
                    'doi': get_val(paper, 'doi')
                }

        # Collect texts for TF-IDF (Concatenate Title + Abstract for better Sparse signal)
        all_texts = []
        for item in embeddings:
            p_id = item.get('paper_id')
            p_info = paper_lookup.get(p_id, {})
            text = f"{p_info.get('title', '')} {p_info.get('abstract', '')}".strip()
            if text:
                all_texts.append(text)

        if not self.is_tfidf_fitted and all_texts:
            self.fit_tfidf_vectorizer(all_texts)

        batches = []
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]

            ids, titles, abstracts, dense_embs, sparse_embs = [], [], [], [], []

            for item in batch:
                p_id = item['paper_id']
                p_info = paper_lookup.get(p_id, {})

                title = p_info.get('title', '')
                abstract = p_info.get('abstract', '')

                ids.append(p_id)
                titles.append(title[:1000])
                abstracts.append(abstract[:8000])

                # Dense Embedding
                dense_embs.append(item.get('embedding', [0.0] * 768))

                # Sparse Embedding (Using concatenated text for richer hybrid search)
                full_text = f"{title} {abstract}".strip()
                if full_text and self.is_tfidf_fitted:
                    sparse_embs.append(self.generate_sparse_embedding(full_text))
                else:
                    sparse_embs.append({0: 0.001})

            batches.append([ids, titles, abstracts, dense_embs, sparse_embs])

        return batches

    def upload_embeddings(self, embedding_file: Optional[str] = None, papers_data: Optional[List[Dict]] = None,
                          batch_size: int = 1000) -> bool:
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

            # Create collection (force recreate for hybrid schema compatibility)
            if not self.create_collection(embedding_dim, force_recreate=True):
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

    def upload_figures_embeddings(self, figures_data: List[Dict], batch_size: int = 100) -> bool:
        """Upload figures embeddings to figures collection.

        Args:
            figures_data: List of figure data with embeddings
            batch_size: Number of records per batch

        Returns:
            True if successful, False otherwise
        """
        try:
            if not figures_data:
                print("❌ No figures data to upload")
                return False

            # Create figures collection
            if not self.create_figures_collection(force_recreate=True):
                return False

            # Prepare texts for TF-IDF fitting if not already fitted
            if not self.is_tfidf_fitted:
                all_texts = []
                for figure in figures_data:
                    description = figure.get('description', '')
                    if description:
                        all_texts.append(description)

                if all_texts:
                    self.fit_tfidf_vectorizer(all_texts)

            # Prepare batch data
            batches = []
            for i in range(0, len(figures_data), batch_size):
                batch = figures_data[i:i + batch_size]

                ids, paper_ids, descriptions = [], [], []
                description_embeddings, image_embeddings, sparse_embeddings = [], [], []

                for figure in batch:
                    ids.append(figure['id'])
                    paper_ids.append(figure['paper_id'])
                    descriptions.append(figure['description'][:2000])  # Truncate to max length

                    # Dense embeddings
                    description_embeddings.append(figure.get('description_embedding', [0.0] * 768))
                    image_embeddings.append(figure.get('image_embedding', [0.0] * 512))

                    # Sparse embedding
                    desc_text = figure.get('description', '')
                    if desc_text and self.is_tfidf_fitted:
                        sparse_embeddings.append(self.generate_sparse_embedding(desc_text))
                    else:
                        sparse_embeddings.append({0: 0.001})  # Minimal sparse vector

                batches.append(
                    [ids, paper_ids, descriptions, description_embeddings, image_embeddings, sparse_embeddings])

            # Upload batches
            print(f"📤 Uploading {len(figures_data)} figures in {len(batches)} batches...")

            figures_collection = Collection("figures_collection")
            total_inserted = 0

            for i, batch_data in enumerate(tqdm(batches, desc="Uploading figures batches")):
                try:
                    insert_result = figures_collection.insert(batch_data)
                    batch_count = len(batch_data)  # Number of IDs in batch
                    total_inserted += batch_count

                    if (i + 1) % 5 == 0:  # Flush every 5 batches
                        figures_collection.flush()

                except Exception as e:
                    print(f"❌ Error uploading figures batch {i + 1}: {e}")
                    continue

            # Final flush and load
            figures_collection.flush()
            figures_collection.load()

            print(f"\n📊 Figures Upload Summary:")
            print(f"   Total figures: {len(figures_data)}")
            print(f"   Successfully inserted: {total_inserted}")
            print(f"   Success rate: {total_inserted / len(figures_data) * 100:.1f}%")
            print(f"   Collection name: figures_collection")
            print("✅ Figures upload completed successfully")

            return True

        except Exception as e:
            print(f"❌ Figures upload failed: {e}")
            return False

    def upload_tables_embeddings(self, tables_data: List[Dict], batch_size: int = 100) -> bool:
        """Upload tables embeddings to tables collection.

        Args:
            tables_data: List of table data with embeddings
            batch_size: Number of records per batch

        Returns:
            True if successful, False otherwise
        """
        try:
            if not tables_data:
                print("❌ No tables data to upload")
                return False

            # Create tables collection
            if not self.create_tables_collection(force_recreate=True):
                return False

            # Prepare texts for TF-IDF fitting if not already fitted
            if not self.is_tfidf_fitted:
                all_texts = []
                for table in tables_data:
                    # Use both description and table text for richer embeddings
                    description = table.get('description', '')
                    if description:
                        all_texts.append(description)

                if all_texts:
                    self.fit_tfidf_vectorizer(all_texts)

            # Prepare batch data
            batches = []
            for i in range(0, len(tables_data), batch_size):
                batch = tables_data[i:i + batch_size]

                ids, paper_ids, descriptions = [], [], []
                description_embeddings, sparse_embeddings = [], []

                for table in batch:
                    ids.append(table['id'])
                    paper_ids.append(table['paper_id'])
                    descriptions.append(table['description'][:8000])  # Truncate to max length

                    # Dense embedding
                    description_embeddings.append(table.get('description_embedding', [0.0] * 768))

                    # Sparse embedding (use combined text for richer representation)
                    desc_text = table.get('description', '')

                    if desc_text and self.is_tfidf_fitted:
                        sparse_embeddings.append(self.generate_sparse_embedding(desc_text))
                    else:
                        sparse_embeddings.append({0: 0.001})  # Minimal sparse vector

                batches.append([ids, paper_ids, descriptions, description_embeddings, sparse_embeddings])

            # Upload batches
            print(f"📤 Uploading {len(tables_data)} tables in {len(batches)} batches...")

            tables_collection = Collection("tables_collection")
            total_inserted = 0

            for i, batch_data in enumerate(tqdm(batches, desc="Uploading tables batches")):
                try:
                    insert_result = tables_collection.insert(batch_data)
                    batch_count = len(batch_data)  # Number of IDs in batch
                    total_inserted += batch_count

                    if (i + 1) % 5 == 0:  # Flush every 5 batches
                        tables_collection.flush()

                except Exception as e:
                    print(f"❌ Error uploading tables batch {i + 1}: {e}")
                    continue

            # Final flush and load
            tables_collection.flush()
            tables_collection.load()

            print(f"\n📊 Tables Upload Summary:")
            print(f"   Total tables: {len(tables_data)}")
            print(f"   Successfully inserted: {total_inserted}")
            print(f"   Success rate: {total_inserted / len(tables_data) * 100:.1f}%")
            print(f"   Collection name: tables_collection")
            print("✅ Tables upload completed successfully")

            return True

        except Exception as e:
            print(f"❌ Tables upload failed: {e}")
            return False

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

            # Display schema information
            schema = self.collection.schema
            field_names = [field.name for field in schema.fields]
            print(f"   Schema fields: {field_names}")

            # Test search functionality
            if num_entities > 0:
                # Test a simple query to verify data exists
                try:
                    sample_results = self.collection.query(
                        expr="",
                        limit=3,
                        output_fields=["id", "title", "abstract"]
                    )

                    if sample_results:
                        print(f"   Sample records:")
                        for i, record in enumerate(sample_results[:2]):
                            print(f"     {i + 1}. ID: {record.get('id', 'N/A')}")
                            print(f"        Title: {record.get('title', 'N/A')[:80]}...")
                            print(f"        Abstract: {record.get('abstract', 'N/A')[:80]}...")

                except Exception as e:
                    print(f"   Sample query failed: {e}")

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
