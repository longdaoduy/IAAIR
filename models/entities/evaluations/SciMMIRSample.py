from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from PIL import Image
@dataclass
class SciMMIRSample:
    """Single SciMMIR benchmark sample."""
    text: str
    image_path: Optional[str] = None
    image: Optional[Image.Image] = None
    class_label: str = "figure"  # fig_architecture, fig_natural, etc.
    paper_id: Optional[str] = None
    sample_id: Optional[str] = None
    domain: str = "general"