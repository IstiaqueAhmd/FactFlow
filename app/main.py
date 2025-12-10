import os
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Annotated
from datetime import datetime
from contextlib import asynccontextmanager
from database import Database
from models import TextRequest, ResponseModel, Source, URLRequest, SaveResponse
from factchecker import FactChecker
from auth import verify_token, create_access_token
from datetime import timedelta
from dotenv import load_dotenv
load_dotenv()

# Define the directory to store uploaded files
UPLOADS_PDF_DIR = "PDF_Uploads"
UPLOADS_IMG_DIR = "Image_Uploads"

# Define allowed types
ALLOWED_IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
ALLOWED_DOCUMENT_MIME_TYPES = ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]

# Create the uploads directories if they don't already exist
os.makedirs(UPLOADS_PDF_DIR, exist_ok=True)
os.makedirs(UPLOADS_IMG_DIR, exist_ok=True)

#initialize components
fact_checker = FactChecker()
database = Database()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    database.init_db()
    yield
    # Shutdown (if needed)
    database.disconnect()

#initialize FastAPI app
app = FastAPI(
    title="FactFlow API",
    description="API for FactFlow - A Fact-Checking Application",
    version="1.0.0",
    lifespan=lifespan
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


# @app.post("/login")
# async def login(user_id: str):
#     """Generate JWT access token for a user."""
#     access_token = create_access_token(
#         data={"sub": user_id},
#         expires_delta=timedelta(minutes=30)
#     )
#     return {"access_token": access_token, "token_type": "bearer"}


@app.post("/check-text", response_model=ResponseModel)
async def upload_text(request: TextRequest, authenticated_user_id: str = Depends(verify_token)):
    """Fact-check the provided text."""
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Use FactChecker to analyze the text
        result = fact_checker.check_text(request.text)
        
        # Return response with authenticated user_id
        response = ResponseModel(
            user_id=authenticated_user_id,
            verdict=result.verdict,
            confidence=result.confidence,
            claim=result.claim,
            conclusion=result.conclusion,
            evidence=result.evidence,
            sources=result.sources,
            timestamp=result.timestamp
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking text: {str(e)}")


@app.post("/check-image", response_model=ResponseModel)
async def upload_image(
    file: Annotated[UploadFile, File(description="The image file to upload (JPEG, PNG, GIF, WebP).")],
    authenticated_user_id: str = Depends(verify_token)
):
    """
    Upload an image file and fact-check the text extracted from it.

    - **Validates** if the uploaded file is an allowed image type.
    - **Saves** the file to the server.
    - **Extracts** text from the image using vision AI.
    - **Fact-checks** the extracted text against web sources.
    - **Returns** the fact-check results.
    """

    # 1. Validate the file's content type
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file.content_type not in ALLOWED_IMAGE_MIME_TYPES and file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(ALLOWED_IMAGE_MIME_TYPES)}"
        )

    # 2. Fact-check the image directly from uploaded file
    try:
        # Read file bytes into memory
        file_bytes = await file.read()
        result = fact_checker.check_image(file_bytes)
        
        # Return response with authenticated user_id
        response = ResponseModel(
            user_id=authenticated_user_id,
            verdict=result.verdict,
            confidence=result.confidence,
            claim=result.claim,
            conclusion=result.conclusion,
            evidence=result.evidence,
            sources=result.sources,
            timestamp=result.timestamp
        )
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fact-checking image: {str(e)}"
        )


@app.post("/check-pdf", response_model=ResponseModel)
async def upload_pdf(
    file: Annotated[UploadFile, File(description="The PDF file to upload.")],
    authenticated_user_id: str = Depends(verify_token)
):
    """
    Upload a PDF file and fact-check the text extracted from it.

    - **Validates** if the uploaded file is a PDF (MIME type `application/pdf`).
    - **Saves** the file to the server.
    - **Extracts** text from the PDF.
    - **Fact-checks** the extracted text against web sources.
    - **Returns** the fact-check results.
    """

    # 1. Validate the file's content type
    allowed_extensions = {".pdf", ".doc", ".docx", ".txt"}
    file_extension = os.path.splitext(file.filename)[1].lower()

    if file.content_type not in ALLOWED_DOCUMENT_MIME_TYPES and file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: PDF, DOC, DOCX, TXT"
        )

    # 2. Fact-check the document directly from uploaded file
    try:
        # Read file bytes into memory
        file_bytes = await file.read()
        result = fact_checker.check_pdf(file_bytes)
        
        # Return response with authenticated user_id
        response = ResponseModel(
            user_id=authenticated_user_id,
            verdict=result.verdict,
            confidence=result.confidence,
            claim=result.claim,
            conclusion=result.conclusion,
            evidence=result.evidence,
            sources=result.sources,
            timestamp=result.timestamp
        )
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fact-checking PDF: {str(e)}"
        )


@app.post("/check-url", response_model=ResponseModel)
async def upload_url(request: URLRequest, authenticated_user_id: str = Depends(verify_token)):
    """Fact-check the content of a given URL."""
    try:
        if not request.url:
            raise HTTPException(status_code=400, detail="Url cannot be empty")
        
        # Use FactChecker to analyze the text
        result = fact_checker.check_url(request.url)
        
        
        # Return response with authenticated user_id
        response = ResponseModel(
            user_id=authenticated_user_id,
            verdict=result.verdict,
            confidence=result.confidence,
            claim=result.claim,
            conclusion=result.conclusion,
            evidence=result.evidence,
            sources=result.sources,
            timestamp=result.timestamp
        )
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking text: {str(e)}")


@app.post("/save-factcheck")
async def save_result(result: SaveResponse, authenticated_user_id: str = Depends(verify_token)):
    """Save a fact-check result for the authenticated user."""
    try:
        uid = database.save_fact_check(authenticated_user_id, result)
        return {
            "message": "Fact check saved successfully",
            "fact-check-id": uid
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving result: {str(e)}")


@app.get("/get-factchecks", response_model=List[ResponseModel])
async def get_results(
    authenticated_user_id: str = Depends(verify_token), 
    limit: Optional[int] = 10,
    verdict: Optional[str] = None
):
    """Retrieve past fact-check results for the authenticated user.
    
    Args:
        authenticated_user_id: The authenticated user's ID
        limit: Maximum number of results to return (default: 10)
        verdict: Optional filter for verdict (e.g., "true", "false"). If empty, returns all fact-checks.
    """
    try:
        results = database.get_fact_checks(authenticated_user_id, limit, verdict)
        response = [
            ResponseModel(
                uid=res["_id"],
                user_id=res["user_id"],
                verdict=res["verdict"],
                confidence=res["confidence"],
                claim=res["claim"],
                conclusion=res["conclusion"],
                evidence=res["evidence"],
                sources=[Source(**source) for source in res["sources"]],
                timestamp=res["timestamp"]
            ) for res in results
        ]
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")


@app.delete("/delete-factcheck")
async def delete_result(uid: str, authenticated_user_id: str = Depends(verify_token)):
    """Delete a specific fact-check result for the authenticated user."""
    try:
        success = database.delete_fact_check(uid, authenticated_user_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Fact check not found or you don't have permission to delete it"
            )
        
        return {
            "message": "Fact check deleted successfully",
            "uid": uid
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting result: {str(e)}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)
