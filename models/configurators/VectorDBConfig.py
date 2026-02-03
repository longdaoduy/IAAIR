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
    username: str = "db_fc2099952a50820"
    password: str = "Uv6;GF<K>oOb0BwF"
    token: str = "9c756b29aa107040710292568a81291acca3afa7f6c4582b843180b94c207e43dea5bb284ba4867dfb24591a235003630d78c0c8"  # For Zilliz Cloud authentication
    
    # Connection
    host: str = "localhost"
    port: int = 19530
    uri: str = "https://in03-fc2099952a50820.serverless.aws-eu-central-1.cloud.zilliz.com"

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
