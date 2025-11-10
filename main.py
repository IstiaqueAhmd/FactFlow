from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from datetime import datetime
from contextlib import asynccontextmanager



#initialize components




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

@app.post("/check-image")
async def image_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 

@app.post("/check-text")
async def text_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 

@app.post("/check-url")
async def url_check():
    return {"status": "ok", "timestamp": datetime.utcnow()} 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
