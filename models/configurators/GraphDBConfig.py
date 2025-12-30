import os
from dataclasses import dataclass

@dataclass
class GraphDBConfig:
    """Neo4j database configuration."""
    provider: str = "neo4j"
    username: str = "neo4j"
    password: str = "LkIC3FNqbGTGHtkxiYujARXY6IUZnvyvyrMFJBeagCI"
    database: str = "neo4j"
    uri: str = "bolt://localhost:7687"
    max_connection_lifetime: int = 3600
    max_connection_pool_size: int = 50

    @classmethod
    def from_env(cls) -> 'GraphDBConfig':
        return cls(
            uri=os.getenv("NEO4J_URI", cls.uri),
            username=os.getenv("NEO4J_USERNAME", cls.username),
            password=os.getenv("NEO4J_PASSWORD", cls.password),
            database=os.getenv("NEO4J_DATABASE", cls.database),
        )