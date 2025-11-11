from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from models import TextRequest, ResponseModel
from factchecker import FactChecker

#initialize components
fact_checker = FactChecker()

#initialize FastAPI app
app = FastAPI(
    title="FactFlow API",
    description="API for FactFlow - A Fact-Checking Application",
    version="1.0.0",
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


@app.post("/check-text", response_model=ResponseModel)
async def text_check(request: TextRequest):
    """
    Check the factuality of a given text.
    
    Args:
        request: CheckTextRequest containing the text to fact-check
        
    Returns:
        CheckResponse with summary and sources
    """
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use FactChecker to analyze the text
        result = fact_checker.check_text(request.text)
        
        # Parse the result into CheckResponse format
        # Assuming the FactChecker returns a model with summary and sources
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking text: {str(e)}")



@app.post("/check-image")
async def image_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


@app.post("/check-image")
async def image_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


@app.post("/check-url")
async def url_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
