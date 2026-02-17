"""
Institution entity for the Knowledge Fabric system.

Represents academic institutions, organizations, and affiliations
that authors can be associated with.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
import uuid

@dataclass
class Institution:
    """Represents an academic institution or organization."""
    id: str
    name: str
    
    # Institution details
    country: Optional[str] = None
    city: Optional[str] = None
    type: Optional[str] = None  # university, research_institute, company, etc.
    website: Optional[str] = None
    
    # Tracking fields
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())