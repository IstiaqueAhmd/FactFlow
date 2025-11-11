from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal
from datetime import datetime

class TextRequest(BaseModel):
    """Request model for text fact-checking."""
    text: str = Field(..., min_length=1, description="Text to fact-check")

class Source(BaseModel):
    """Model for a source reference."""
    title: str = Field(..., description="Title of the source")
    url: str = Field(default="", description="URL of the source")

class CheckResponse(BaseModel):
    verdict: str
    confidence: float
    summary: str
    reasoning: str
    sources: List[Source]
    timestamp: datetime


# class CheckResponse(BaseModel):
#     """Response model for fact-checking results."""
#     verdict: Literal["TRUE", "FALSE", "UNVERIFIABLE", "ERROR"] = Field(
#         ..., 
#         description="The fact-check verdict"
#     )
#     confidence: float = Field(
#         ..., 
#         ge=0.0, 
#         le=1.0,
#         description="Confidence level of the verdict (0.0 to 1.0)"
#     )
#     summary: str = Field(
#         ..., 
#         description="Brief summary of the fact-check result"
#     )
#     reasoning: str = Field(
#         default="",
#         description="Detailed reasoning behind the verdict"
#     )
#     sources: List[Source] = Field(
#         default_factory=list,
#         description="List of sources used for verification"
#     )
#     timestamp: datetime = Field(
#         default_factory=datetime.utcnow,
#         description="When the fact-check was performed"
#     )

class ResponseModel(CheckResponse):
    """Extended response model that inherits from CheckResponse."""
    pass