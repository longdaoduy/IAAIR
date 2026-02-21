"""
MongoDB Configuration for the IAAIR system.

This module provides configuration management for MongoDB connections
including connection parameters and database settings.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class MongoDBConfig:
    """Configuration class for MongoDB connections."""

    host: str = "mongodb+srv://iaair:Jx5M91RozzdTrvFk@cluster0.0w7ozbp.mongodb.net/"
    port: int = 27017
    database: str = "iaair"
    username: Optional[str] = 'iaair'
    password: Optional[str] = 'Jx5M91RozzdTrvFk'
    auth_source: str = "admin"
    timeout: int = 5000  # milliseconds

    @classmethod
    def from_env(cls) -> 'MongoDBConfig':
        """Create configuration from environment variables.

        Environment variables:
            MONGODB_HOST: MongoDB host (default: localhost)
            MONGODB_PORT: MongoDB port (default: 27017)
            MONGODB_DATABASE: Database name (default: iaair)
            MONGODB_USERNAME: Username for authentication
            MONGODB_PASSWORD: Password for authentication
            MONGODB_AUTH_SOURCE: Authentication source (default: admin)
            MONGODB_TIMEOUT: Connection timeout in ms (default: 5000)

        Returns:
            MongoDBConfig instance
        """
        return cls(
            host=os.getenv("MONGODB_HOST", cls.host),
            port=int(os.getenv("MONGODB_PORT", "27017")),
            database=os.getenv("MONGODB_DATABASE", "iaair"),
            username=os.getenv("MONGODB_USERNAME", cls.username),
            password=os.getenv("MONGODB_PASSWORD", cls.password),
            auth_source=os.getenv("MONGODB_AUTH_SOURCE", "admin"),
            timeout=int(os.getenv("MONGODB_TIMEOUT", "5000"))
        )