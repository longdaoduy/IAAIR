from dataclasses import dataclass, field
from datetime import datetime
from models.schemas.nodes.Paper import Paper
from models.schemas.nodes.Venue import Venue
from typing import List, Dict, Optional, Any, Union
from enum import Enum
import uuid

@dataclass
class PublishIn:
    """Represents a citation relationship between papers."""
    paper: Paper
    venue: Venue
    # context: Optional[str] = None
    # intent: Optional[str] = None  # background, method, result, etc.
    # confidence: float = 1.0
    # created_at: datetime = field(default_factory=datetime.now)