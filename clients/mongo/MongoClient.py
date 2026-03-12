"""
MongoDB Client for storing academic papers, figures, and tables.

This module provides functionality to store and retrieve academic paper content
including figures, tables, and metadata in MongoDB.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymongo import MongoClient as PyMongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError

try:
    import certifi
    CA_CERTS = certifi.where()
except ImportError:
    CA_CERTS = None

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
        self.cypher_templates_collection = "cypher_templates"

    def connect(self) -> bool:
        """Establish connection to MongoDB.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            host = self.config.host

            # If host is an SRV URI (MongoDB Atlas), pass it directly as a URI
            if host.startswith("mongodb+srv://") or host.startswith("mongodb://"):
                connect_kwargs = {
                    "host": host,
                    "serverSelectionTimeoutMS": self.config.timeout,
                }
            else:
                # Local / standard MongoDB connection
                connect_kwargs = {
                    "host": host,
                    "port": self.config.port,
                    "username": self.config.username,
                    "password": self.config.password,
                    "authSource": self.config.auth_source,
                    "serverSelectionTimeoutMS": self.config.timeout,
                }

            # Add TLS CA certs for Atlas SSL connections
            if CA_CERTS:
                connect_kwargs["tlsCAFile"] = CA_CERTS

            self.client = PyMongoClient(**connect_kwargs)

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

            # Cypher templates collection indexes
            templates_coll = self.db[self.cypher_templates_collection]
            templates_coll.create_index("name", unique=True)
            templates_coll.create_index("created_at")

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
        # if not self.is_connected:
        #     logger.error("Not connected to MongoDB")
        #     return False

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

    # ── Cypher Template CRUD (with local JSON fallback) ──

    _LOCAL_TEMPLATES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "cypher_templates.json")
    _last_connect_attempt = None
    _CONNECT_RETRY_SECONDS = 60  # Don't retry connection more than once per minute

    def _ensure_connected(self) -> bool:
        """Ensure MongoDB is connected, auto-connecting if needed."""
        if self.is_connected and self.db is not None:
            return True
        # Avoid hammering connection attempts
        now = datetime.now()
        if self._last_connect_attempt and (now - self._last_connect_attempt).total_seconds() < self._CONNECT_RETRY_SECONDS:
            return False
        self._last_connect_attempt = now
        logger.info("MongoDB not connected, attempting auto-connect...")
        return self.connect()

    # ── Local JSON helpers ──

    def _load_local_templates(self) -> List[Dict]:
        """Load templates from local JSON file."""
        path = os.path.abspath(self._LOCAL_TEMPLATES_FILE)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read local templates file: {e}")
            return []

    def _save_local_templates(self, templates: List[Dict]) -> bool:
        """Persist templates list to local JSON file."""
        path = os.path.abspath(self._LOCAL_TEMPLATES_FILE)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(templates, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Failed to write local templates file: {e}")
            return False

    # ── Public CRUD ──

    def save_cypher_template(self, name: str, query: str, icon: str = "📋", description: str = "") -> Optional[Dict]:
        """Save a custom Cypher query template.
        Falls back to local JSON file if MongoDB is unavailable.
        """
        doc = {
            "name": name.strip(),
            "query": query.strip(),
            "icon": icon.strip() or "📋",
            "description": description.strip(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # Try MongoDB first
        if self._ensure_connected():
            try:
                mongo_doc = dict(doc)
                mongo_doc["created_at"] = datetime.now()
                mongo_doc["updated_at"] = datetime.now()
                self.db[self.cypher_templates_collection].replace_one(
                    {"name": mongo_doc["name"]},
                    mongo_doc,
                    upsert=True
                )
                logger.info(f"Saved Cypher template to MongoDB: {name}")
                return doc
            except Exception as e:
                logger.warning(f"MongoDB save failed, falling back to local file: {e}")

        # Fallback: local JSON file
        templates = self._load_local_templates()
        # Upsert by name
        templates = [t for t in templates if t.get("name") != doc["name"]]
        templates.append(doc)
        if self._save_local_templates(templates):
            logger.info(f"Saved Cypher template to local file: {name}")
            return doc
        return None

    def get_cypher_templates(self) -> List[Dict]:
        """Retrieve all saved Cypher templates.
        Falls back to local JSON file if MongoDB is unavailable.
        """
        # Try MongoDB first
        if self._ensure_connected():
            try:
                templates = list(
                    self.db[self.cypher_templates_collection]
                    .find({}, {"_id": 0})
                    .sort("created_at", 1)
                )
                # Serialize datetime fields
                for t in templates:
                    for key in ("created_at", "updated_at"):
                        if key in t and hasattr(t[key], "isoformat"):
                            t[key] = t[key].isoformat()
                return templates
            except Exception as e:
                logger.warning(f"MongoDB read failed, falling back to local file: {e}")

        # Fallback: local JSON file
        templates = self._load_local_templates()
        return sorted(templates, key=lambda t: t.get("created_at", ""))

    def delete_cypher_template(self, name: str) -> bool:
        """Delete a Cypher template by name.
        Falls back to local JSON file if MongoDB is unavailable.
        """
        # Try MongoDB first
        if self._ensure_connected():
            try:
                result = self.db[self.cypher_templates_collection].delete_one({"name": name})
                if result.deleted_count > 0:
                    logger.info(f"Deleted Cypher template from MongoDB: {name}")
                    return True
                else:
                    logger.warning(f"Cypher template not found in MongoDB: {name}")
                    return False
            except Exception as e:
                logger.warning(f"MongoDB delete failed, falling back to local file: {e}")

        # Fallback: local JSON file
        templates = self._load_local_templates()
        before = len(templates)
        templates = [t for t in templates if t.get("name") != name]
        if len(templates) < before:
            self._save_local_templates(templates)
            logger.info(f"Deleted Cypher template from local file: {name}")
            return True
        logger.warning(f"Cypher template not found locally: {name}")
        return False

    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.is_connected = False
            logger.info("Disconnected from MongoDB")

    def __del__(self):
        """Cleanup on object destruction."""
        self.disconnect()