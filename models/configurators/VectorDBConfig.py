"""
Configuration settings for the Knowledge Fabric system.

Centralizes all configuration parameters with environment variable support
and validation for production deployment.
"""

import os
from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    """Vector database configuration (Milvus/Weaviate)."""
    provider: str = "milvus"  # "milvus" or "weaviate"
    username : str = "db_c605f97af57afe9"
    password : str = "Cm3&Xz<+cJ},p|%+"
    host: str = "localhost"
    port: int = 19530

    # Milvus specific
    collection_name: str = "scientific_papers"
    index_type: str = "IVF_FLAT"
    metric_type: str = "L2"
    nlist: int = 1024

    # Weaviate specific
    scheme: str = "http"
    @classmethod
    def from_env(cls) -> 'VectorDBConfig':
        return cls(
            provider=os.getenv("VECTOR_DB_PROVIDER", cls.provider),
            host=os.getenv("VECTOR_DB_HOST", cls.host),
            port=int(os.getenv("VECTOR_DB_PORT", cls.port)),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", cls.collection_name),
        )
