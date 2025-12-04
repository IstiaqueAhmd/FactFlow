from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime

class TextRequest(BaseModel):
    """Request model for text fact-checking."""
    text: str = Field(..., min_length=1, description="Text to fact-check")

class URLRequest(BaseModel):
    """Request model for URL fact-checking."""
    url: str = Field(..., description="URL of the content to fact-check")

class Source(BaseModel):
    """Model for a source reference."""
    title: str = Field(..., description="Title of the source")
    url: str = Field(default="", description="URL of the source")

class CheckResponse(BaseModel):
    verdict: Literal["TRUE", "FALSE", "UNVERIFIABLE", "ERROR"] = Field(..., description="The verdict of the fact-check")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    claim: str = Field(..., description="The main claim being checked")
    conclusion: str = Field(..., description="1-2 sentence conclusion")
    evidence: Dict[str, List[str]] = Field(..., description="Supporting and counter evidence")
    sources: List[Source] = Field(..., description="Citations with title and URL")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the fact-check")

class ResponseModel(CheckResponse):
    """Extended response model that inherits from CheckResponse."""
    uid: Optional[str] = Field(None, description="Unique identifier for the fact check")
    user_id: str = Field(..., description="Unique identifier for the user")