"""
MongoDB Client for storing academic papers, figures, and tables.

This module provides functionality to store and retrieve academic paper content
including figures, tables, and metadata in MongoDB.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient as PyMongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

from models.configurators.MongoDBConfig import MongoDBConfig
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Figure import Figure
from models.schemas.nodes.Table import Table

logger = logging.getLogger(__name__)


class MongoClient:
    """Client for interacting with MongoDB for academic paper storage."""

    def __init__(self, config: Optional[MongoDBConfig] = None):
        """Initialize MongoDB client.

        Args:
            config: MongoDB configuration. If None, loads from environment.
        """
        self.config = config or MongoDBConfig.from_env()
        self.client = None
        self.db = None
        self.is_connected = False

        # Collection names
        self.papers_collection = "papers"
        self.figures_collection = "figures"
        self.tables_collection = "tables"

    def connect(self) -> bool:
        """Establish connection to MongoDB.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = PyMongoClient(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                authSource=self.config.auth_source,
                serverSelectionTimeoutMS=self.config.timeout
            )

            # Test connection
            self.client.admin.command('ping')

            self.db = self.client[self.config.database]
            self.is_connected = True

            # Create indexes for better query performance
            self._create_indexes()

            logger.info(f"Successfully connected to MongoDB database: {self.config.database}")
            return True

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to MongoDB: {e}")
            return False

    def _create_indexes(self):
        """Create indexes for better query performance."""
        try:
            # Papers collection indexes
            papers_coll = self.db[self.papers_collection]
            papers_coll.create_index("id", unique=True)
            papers_coll.create_index("doi")
            papers_coll.create_index("title")
            papers_coll.create_index("publication_date")

            # Figures collection indexes
            figures_coll = self.db[self.figures_collection]
            figures_coll.create_index("id", unique=True)
            figures_coll.create_index("paper_id")
            figures_coll.create_index("figure_number")

            # Tables collection indexes
            tables_coll = self.db[self.tables_collection]
            tables_coll.create_index("id", unique=True)
            tables_coll.create_index("paper_id")
            tables_coll.create_index("table_number")

            logger.info("Successfully created MongoDB indexes")

        except Exception as e:
            logger.warning(f"Failed to create some indexes: {e}")

    def save_paper(self, paper: Paper) -> bool:
        """Save paper to MongoDB.

        Args:
            paper: Paper entity to save

        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            logger.error("Not connected to MongoDB")
            return False

        try:
            paper_doc = self._paper_to_dict(paper)

            # Use upsert to update if exists
            result = self.db[self.papers_collection].replace_one(
                {"id": paper.id},
                paper_doc,
                upsert=True
            )

            if result.upserted_id or result.modified_count > 0:
                logger.info(f"Successfully saved paper: {paper.id}")
                return True
            else:
                logger.warning(f"No changes made to paper: {paper.id}")
                return False

        except Exception as e:
            logger.error(f"Failed to save paper {paper.id}: {e}")
            return False

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Retrieve paper by ID.

        Args:
            paper_id: Paper ID to retrieve

        Returns:
            Paper document if found, None otherwise
        """
        if not self.is_connected:
            return None

        try:
            return self.db[self.papers_collection].find_one({"id": paper_id})
        except Exception as e:
            logger.error(f"Failed to retrieve paper {paper_id}: {e}")
            return None

    def get_figures_by_paper(self, paper_id: str) -> List[Dict]:
        """Retrieve all figures for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            List of figure documents
        """
        if not self.is_connected:
            return []

        try:
            return list(self.db[self.figures_collection].find({"paper_id": paper_id}))
        except Exception as e:
            logger.error(f"Failed to retrieve figures for paper {paper_id}: {e}")
            return []

    def get_tables_by_paper(self, paper_id: str) -> List[Dict]:
        """Retrieve all tables for a paper.

        Args:
            paper_id: Paper ID

        Returns:
            List of table documents
        """
        if not self.is_connected:
            return []

        try:
            return list(self.db[self.tables_collection].find({"paper_id": paper_id}))
        except Exception as e:
            logger.error(f"Failed to retrieve tables for paper {paper_id}: {e}")
            return []

    def _paper_to_dict(self, paper: Paper) -> Dict[str, Any]:
        """Convert Paper entity to MongoDB document.

        Args:
            paper: Paper entity

        Returns:
            Dictionary representation for MongoDB
        """
        return {
            "id": paper.id,
            "title": paper.title,
            "abstract": paper.abstract,
            "publication_date": paper.publication_date,
            "doi": paper.doi,
            "pmid": paper.pmid,
            "pdf_url": paper.pdf_url,
            "source": paper.source,
            "metadata": paper.metadata,
            "ingested_at": paper.ingested_at,
            "last_updated": datetime.now(),
            "stored_in_mongodb": True
        }

    def _figure_to_dict(self, figure: Figure) -> Dict[str, Any]:
        """Convert Figure entity to MongoDB document.

        Args:
            figure: Figure entity

        Returns:
            Dictionary representation for MongoDB
        """
        return {
            "id": figure.id,
            "paper_id": figure.paper_id,
            "figure_number": figure.figure_number,
            "description": figure.description,
            "caption": figure.caption,
            "page_number": figure.page_number,
            "image_path": figure.image_path,
            "created_at": datetime.now(),
            "last_updated": datetime.now(),
            "stored_in_mongodb": True,
            # Note: We don't store embeddings in MongoDB as they're in Milvus
            "has_image_embedding": figure.image_embedding is not None,
            "has_description_embedding": figure.description_embedding is not None
        }

    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from MongoDB")

    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()