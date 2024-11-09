from pydantic import BaseModel
from typing import Dict, Optional

class AnalysisResponse(BaseModel):
    status: str
    results_directory: str
    summaries: Optional[Dict[int, str]]