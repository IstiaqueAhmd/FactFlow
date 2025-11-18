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

class Evidence(BaseModel):
    """Model for evidence supporting or countering a claim."""
    supporting: List[str] = Field(default_factory=list, description="Evidence supporting the claim")
    counter: List[str] = Field(default_factory=list, description="Evidence countering the claim")

class CheckResponse(BaseModel):
    verdict: Literal["TRUE", "FALSE", "UNVERIFIABLE", "ERROR"] = Field(..., description="The verdict of the fact-check")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    claim: str = Field(..., description="The main claim being checked")
    conclusion: str = Field(..., description="1-2 sentence conclusion")
    evidence: Evidence = Field(..., description="Supporting and counter evidence")
    sources: List[Source] = Field(default_factory=list, description="Citations with title and URL")
    timestamp: Optional[datetime] = Field(default=None, description="Timestamp of the fact-check")
    
    def model_post_init(self, __context):
        """Set timestamp after model initialization if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class ResponseModel(CheckResponse):
    """Extended response model that inherits from CheckResponse."""
    user_id: str = Field(..., description="Unique identifier for the user")