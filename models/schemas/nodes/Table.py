"""
Table entity for the Knowledge Fabric system.

Represents tables extracted from academic papers with their
descriptions and content structure.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List

@dataclass
class Table:
    """Represents a table extracted from a paper."""
    id: str  # Format: paper_id#table_number
    paper_id: str
    table_number: int
    
    # Content details
    description: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = None
    
    # Table structure
    headers: Optional[List[str]] = None
    rows: Optional[List[List[str]]] = None

    # Visual representation (if available)
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None  # CLIP embedding if image available
    description_embedding: Optional[List[float]] = None  # SciBERT embedding for description
    
    # Tracking fields
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id and self.paper_id and self.table_number:
            self.id = f"{self.paper_id}#table_{self.table_number}"