import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Annotated
from datetime import datetime
from contextlib import asynccontextmanager

from models import TextRequest, ResponseModel
from factchecker import FactChecker

# Define the directory to store uploaded files
UPLOADS_PDF_DIR = "PDF_Uploads"

# Create the uploads directory if it doesn't already exist
os.makedirs(UPLOADS_PDF_DIR, exist_ok=True)

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


@app.post("/check-pdf")
async def check_pdf(file: Annotated[UploadFile, File(description="The PDF file to upload.")]):
    """
    Uploads a PDF file and saves it to the 'uploads' directory.

    - **Validates** if the uploaded file is a PDF (MIME type `application/pdf`).
    - **Saves** the file to the server.
    - **Returns** a confirmation message upon success.
    """

    # 1. Validate the file's content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,  # 400 Bad Request
            detail="Invalid file type. Only PDF files (application/pdf) are allowed."
        )

    # 2. Define the full path to save the file
    # Note: This uses the original filename. Be cautious in a production
    # environment, as filenames can be malicious. You might want to
    # sanitize or generate a unique name (e.g., using UUID).
    file_path = os.path.join(UPLOADS_PDF_DIR, file.filename)

    # 3. Save the file to the directory
    try:
        # We use shutil.copyfileobj to efficiently stream the file to disk
        # This is more memory-efficient than file.read() for large files
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        # Handle potential file-saving errors
        raise HTTPException(
            status_code=500,  # 500 Internal Server Error
            detail=f"There was an error saving the file: {e}"
        )
    finally:
        # Always close the uploaded file
        await file.close()

    # 4. Return a success response
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "message": f"File '{file.filename}' was successfully uploaded to {file_path}."
    }


@app.post("/check-url")
async def url_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
