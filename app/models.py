from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime

class TextRequest(BaseModel):
    """Request model for text fact-checking."""
    user_id: str = Field(..., description="Unique identifier for the user") 
    text: str = Field(..., min_length=1, description="Text to fact-check")

class Source(BaseModel):
    """Model for a source reference."""
    title: str = Field(..., description="Title of the source")
    url: str = Field(default="", description="URL of the source")

class CheckResponse(BaseModel):
    verdict: str
    confidence: float
    claim: str
    conclusion: str
    evidence: Dict[str, List[str]]
    sources: List[Source]
    timestamp: datetime

class ResponseModel(CheckResponse):
    """Extended response model that inherits from CheckResponse."""
    user_id: str = Field(..., description="Unique identifier for the user")