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
UPLOADS_IMG_DIR = "Image_Uploads"

# Define allowed image types
ALLOWED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]


# Create the uploads directories if they don't already exist
os.makedirs(UPLOADS_PDF_DIR, exist_ok=True)
os.makedirs(UPLOADS_IMG_DIR, exist_ok=True)

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
async def upload_text(request: TextRequest):
    """Fact-check the provided text."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use FactChecker to analyze the text
        result = fact_checker.check_text(request.text)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking text: {str(e)}")


@app.post("/check-image", response_model=ResponseModel)
async def upload_image(file: Annotated[UploadFile, File(description="The image file to upload (JPEG, PNG, GIF, WebP).")]):
    """
    Upload an image file and fact-check the text extracted from it.

    - **Validates** if the uploaded file is an allowed image type.
    - **Saves** the file to the server.
    - **Extracts** text from the image using vision AI.
    - **Fact-checks** the extracted text against web sources.
    - **Returns** the fact-check results.
    """

    # 1. Validate the file's content type
    if file.content_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_IMAGE_MIME_TYPES)}"
        )

    # 2. Define the full path to save the file
    file_path = os.path.join(UPLOADS_IMG_DIR, file.filename)

    # 3. Save the file to the directory
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving the file: {str(e)}"
        )
    finally:
        await file.close()

    # 4. Fact-check the image
    try:
        result = fact_checker.check_image(file_path)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fact-checking image: {str(e)}"
        )


@app.post("/check-pdf", response_model=ResponseModel)
async def upload_pdf(file: Annotated[UploadFile, File(description="The PDF file to upload.")]):
    """
    Upload a PDF file and fact-check the text extracted from it.

    - **Validates** if the uploaded file is a PDF (MIME type `application/pdf`).
    - **Saves** the file to the server.
    - **Extracts** text from the PDF.
    - **Fact-checks** the extracted text against web sources.
    - **Returns** the fact-check results.
    """

    # 1. Validate the file's content type
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files (application/pdf) are allowed."
        )

    # 2. Define the full path to save the file
    file_path = os.path.join(UPLOADS_PDF_DIR, file.filename)

    # 3. Save the file to the directory
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving the file: {str(e)}"
        )
    finally:
        await file.close()

    # 4. Fact-check the PDF
    try:
        result = fact_checker.check_pdf(file_path)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fact-checking PDF: {str(e)}"
        )


@app.post("/check-url", response_model=ResponseModel)
async def upload_url(url: str):
    """Fact-check the content of a given URL."""
    try:
        if not url:
            raise HTTPException(status_code=400, detail="Url cannot be empty")
        
        # Use FactChecker to analyze the text
        result = fact_checker.check_url(url)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking text: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
