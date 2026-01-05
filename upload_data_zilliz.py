"""
Zilliz Cloud Data Upload Service for SciBERT Embeddings.

This module uploads SciBERT paper embeddings to Zilliz Cloud (managed Milvus service)
with proper authentication and error handling.
"""

import json
import os
import glob
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)

from models.configurators.VectorDBConfig import VectorDBConfig


class ZillizUploadService:
    """Service for uploading embeddings to Zilliz Cloud."""
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """Initialize Zilliz upload service.
        
        Args:
            config: Vector database configuration. If None, loads from environment.
        """
        self.config = config or VectorDBConfig.from_env()
        self.collection = None
        self.is_connected = False
        
    def connect(self) -> bool:
        """Connect to Zilliz Cloud using authentication token.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Get authentication from environment or config
            token = os.getenv("ZILLIZ_TOKEN") or getattr(self.config, 'token', None)
            uri = os.getenv("ZILLIZ_URI") or getattr(self.config, 'uri', None)
            
            if not token or not uri:
                # Fallback to username/password authentication
                username = getattr(self.config, 'username', None) 
                password = getattr(self.config, 'password', None)
                host = getattr(self.config, 'host', 'localhost')
                port = getattr(self.config, 'port', 19530)
                
                if username and password:
                    print(f"🔐 Connecting to Zilliz with username/password: {host}:{port}")
                    connections.connect(
                        alias="default",
                        host=host,
                        port=port,
                        user=username,
                        password=password
                    )
                else:
                    raise ValueError("No valid authentication method found. Please set ZILLIZ_TOKEN and ZILLIZ_URI, or provide username/password in config.")
            else:
                print(f"🔐 Connecting to Zilliz Cloud with token authentication: {uri}")
                connections.connect(
                    alias="default",
                    uri=uri,
                    token=token
                )
            
            self.is_connected = True
            print("✅ Successfully connected to Zilliz Cloud")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to Zilliz: {e}")
            return False
    
    def create_collection_schema(self, embedding_dim: int) -> CollectionSchema:
        """Create collection schema for paper embeddings.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            
        Returns:
            Collection schema
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="paper_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="embedding_source", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="SciBERT embeddings for academic papers"
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
                if force_recreate:
                    print(f"🗑️ Dropping existing collection: {collection_name}")
                    utility.drop_collection(collection_name)
                else:
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
            
            # Create index for vector similarity search
            index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist}
            }
            
            print(f"🔍 Creating index with params: {index_params}")
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
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
    
    def prepare_batch_data(self, embeddings: List[Dict], batch_size: int = 1000) -> List[List[Any]]:
        """Prepare embedding data for batch insertion.
        
        Args:
            embeddings: List of embedding data
            batch_size: Number of records per batch
            
        Returns:
            List of batches, each containing field data
        """
        batches = []
        
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            
            # Prepare field data
            ids = []
            paper_ids = []
            titles = []
            embedding_sources = []
            embedding_vectors = []
            
            for item in batch:
                # Generate unique ID
                unique_id = f"{item['paper_id']}_{datetime.now().strftime('%Y%m%d')}"
                ids.append(unique_id)
                paper_ids.append(item['paper_id'])
                titles.append(item.get('title', '')[:1000])  # Truncate if too long
                embedding_sources.append(item.get('embedding_source', 'unknown'))
                embedding_vectors.append(item['embedding'])
            
            batch_data = [ids, paper_ids, titles, embedding_sources, embedding_vectors]
            batches.append(batch_data)
        
        return batches
    
    def upload_embeddings(self, embedding_file: Optional[str] = None, batch_size: int = 1000) -> bool:
        """Upload embeddings to Zilliz Cloud.
        
        Args:
            embedding_file: Path to embedding file. Auto-detects if None.
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
            
            # Get embedding dimension
            embedding_dim = len(embeddings[0]['embedding'])
            print(f"📏 Embedding dimension: {embedding_dim}")
            
            # Create collection
            if not self.create_collection(embedding_dim):
                return False
            
            # Prepare batch data
            batches = self.prepare_batch_data(embeddings, batch_size)
            
            # Upload batches
            print(f"📤 Uploading {len(embeddings)} embeddings in {len(batches)} batches...")
            
            total_inserted = 0
            for i, batch_data in enumerate(tqdm(batches, desc="Uploading batches")):
                try:
                    insert_result = self.collection.insert(batch_data)
                    batch_count = len(batch_data[0])  # Number of IDs in batch
                    total_inserted += batch_count
                    
                    if (i + 1) % 10 == 0:  # Flush every 10 batches
                        self.collection.flush()
                        
                except Exception as e:
                    print(f"❌ Error uploading batch {i+1}: {e}")
                    continue
            
            # Final flush
            self.collection.flush()
            
            # Load collection for search
            self.collection.load()
            
            print(f"\n📊 Upload Summary:")
            print(f"   Total records: {len(embeddings)}")
            print(f"   Successfully inserted: {total_inserted}")
            print(f"   Success rate: {total_inserted/len(embeddings)*100:.1f}%")
            print(f"   Collection name: {self.config.collection_name}")
            print("✅ Upload completed successfully")
            
            return True
            
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return False
    
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
            
            # Test a simple search
            if num_entities > 0:
                # Get a random vector for testing
                search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": 10}}
                
                # Use zero vector for test search (just to verify search works)
                results = self.collection.search(
                    data=[[0.0] * 768],  # Assume 768-dim embeddings
                    anns_field="embedding",
                    param=search_params,
                    limit=1
                )
                
                if results:
                    print(f"   Search test: ✅ Working")
                else:
                    print(f"   Search test: ⚠️ No results")
            
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


def main():
    """Main function to upload embeddings to Zilliz."""
    print("🚀 Starting Zilliz Upload Service")
    
    # Initialize service
    service = ZillizUploadService()
    
    try:
        # Connect to Zilliz
        if not service.connect():
            print("❌ Failed to connect to Zilliz. Please check your configuration.")
            return
        
        # Upload embeddings
        success = service.upload_embeddings(batch_size=500)
        
        if success:
            # Verify upload
            service.verify_upload()
            print("✅ All operations completed successfully")
        else:
            print("❌ Upload failed")
    
    except Exception as e:
        print(f"❌ Error during upload process: {e}")
        
    finally:
        service.disconnect()


if __name__ == "__main__":
    main()
