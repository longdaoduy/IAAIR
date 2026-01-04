from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Union
import uuid
@dataclass
class Author:
    """Represents an author in the knowledge fabric."""
    id: str
    name: str
    orcid: Optional[str] = None
    # affiliation: Optional[str] = None
    email: Optional[str] = None
    h_index: Optional[int] = None
    
    # Tracking fields
    ingested_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())