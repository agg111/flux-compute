"""
ML Workload Optimization API Server
Main entry point for the FastAPI application
"""
from fastapi import FastAPI
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize AWS S3 bucket
from utils.aws_utils import ensure_s3_bucket
from config.aws_config import s3_client

if s3_client:
    ensure_s3_bucket()

# Import routes
from routes import api_router

# Create the main app
app = FastAPI(title="ML Workload Optimization API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ML Workload Optimization Platform",
        "status": "running",
        "version": "1.0.0"
    }


# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
