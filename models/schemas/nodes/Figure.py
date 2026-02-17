"""
Figure entity for the Knowledge Fabric system.

Represents figures extracted from academic papers with their
descriptions and visual embeddings.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
import numpy as np

@dataclass
class Figure:
    """Represents a figure extracted from a paper."""
    id: str  # Format: paper_id#figure_number
    paper_id: str
    figure_number: int
    
    # Content details
    description: Optional[str] = None
    caption: Optional[str] = None
    page_number: Optional[int] = None
    
    # Visual representation
    image_path: Optional[str] = None
    image_embedding: Optional[List[float]] = None  # CLIP embedding
    
    # Tracking fields
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id and self.paper_id and self.figure_number:
            self.id = f"{self.paper_id}#figure_{self.figure_number}"