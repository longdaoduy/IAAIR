"""
Core data schemas for the Knowledge Fabric system.

Defines the fundamental data structures for scientific literature,
citations, evidence bundles, and provenance tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

class VenueType(Enum):
    """Types of publication venues."""
    JOURNAL = "journal"
    CONFERENCE = "conference"
    WORKSHOP = "workshop"
    ARXIV = "arxiv"
    BOOK = "book"

@dataclass
class Venue:
    """Represents a publication venue."""
    id: str
    name: str
    venue_type: VenueType
    issn: Optional[str] = None
    impact_factor: Optional[float] = None
    publisher: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
