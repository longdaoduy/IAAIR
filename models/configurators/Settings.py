"""
Configuration settings for the Knowledge Fabric system.

Centralizes all configuration parameters with environment variable support
and validation for production deployment.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from . import VectorDBConfig
from . import SciBERTConfig
from . import GraphDBConfig


@dataclass
class Settings:
    """Main configuration container for the Knowledge Fabric system."""
    neo4j: GraphDBConfig = field(default_factory=GraphDBConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    embedding: SciBERTConfig = field(default_factory=SciBERTConfig)

    # Global settings
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        return cls(
            neo4j=GraphDBConfig.from_env(),
            vector_db=VectorDBConfig.from_env(),
            embedding=SciBERTConfig.from_env(),
            environment=os.getenv("ENVIRONMENT", "development"),
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            data_dir=Path(os.getenv("DATA_DIR", "./data")),
            cache_dir=Path(os.getenv("CACHE_DIR", "./cache")),
        )

# Global settings instance
settings = Settings.from_env()