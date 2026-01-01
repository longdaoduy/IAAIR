"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from models.schemas.nodes.Paper import Paper
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

@dataclass
class CitedBy:
    """Represents a citation relationship between papers."""
    src_paper: Paper
    dest_paper: Paper
    # context: Optional[str] = None
    # intent: Optional[str] = None  # background, method, result, etc.
    # confidence: float = 1.0
    # created_at: datetime = field(default_factory=datetime.now)