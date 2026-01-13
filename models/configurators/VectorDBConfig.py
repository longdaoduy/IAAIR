"""
Configuration settings for the Knowledge Fabric system.

Centralizes all configuration parameters with environment variable support
and validation for production deployment.
"""

import os
from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    """Vector database configuration (Milvus/Weaviate/Zilliz)."""
    provider: str = "milvus"  # "milvus", "weaviate", or "zilliz"
    
    # Authentication
    username: str = "db_c605f97af57afe9"
    password: str = "Cm3&Xz<+cJ},p|%+"
    token: str = "077c916c3ea903b26c6158ec923a657b8c674d5a46e1561b57eb6e54d5467a12dec4f58d87d54c752e548cf1f283de84d224b1f3"  # For Zilliz Cloud authentication
    
    # Connection
    host: str = "localhost"
    port: int = 19530
    uri: str = "https://in03-c605f97af57afe9.serverless.aws-eu-central-1.cloud.zilliz.com"

    # Milvus/Zilliz specific
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
            username=os.getenv("VECTOR_DB_USERNAME", cls.username),
            password=os.getenv("VECTOR_DB_PASSWORD", cls.password),
            token=os.getenv("ZILLIZ_TOKEN", cls.token),
            host=os.getenv("VECTOR_DB_HOST", cls.host),
            port=int(os.getenv("VECTOR_DB_PORT", cls.port)),
            uri=os.getenv("ZILLIZ_URI", cls.uri),
            collection_name=os.getenv("VECTOR_COLLECTION_NAME", cls.collection_name),
        )
