import os, sys
sys.path.append(os.path.dirname(__file__))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))
except ImportError:
    # python-dotenv not available, rely on system environment variables
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, HttpUrl
from typing import Optional
import asyncio
import uvicorn

from agents.smb_agent import analyze_smb_website, SMBCompanyData, format_smb_report
from routers.analysis import router as analysis_router
# from routers.match import router as match_router  # Disabled - legacy research agent
from models.responses import HealthResponse

app = FastAPI(
    title="SMB Research Agent API",
    description="AI-powered SMB company analysis for private equity evaluation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router)
# app.include_router(match_router)  # Disabled - uses legacy research agent (archived)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

class AnalysisRequest(BaseModel):
    """Request model for SMB analysis"""
    website_url: HttpUrl
    company_name: Optional[str] = None
    include_report: bool = False

class AnalysisResponse(BaseModel):
    """Response model for SMB analysis"""
    success: bool
    data: Optional[SMBCompanyData] = None
    report: Optional[str] = None
    error: Optional[str] = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SMB Research Agent API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze - POST - Analyze SMB company website (legacy)",
            "analysis": "/analysis/ - Analysis endpoints with detailed/summary/bulk options",
            "docs": "/docs - Interactive API documentation",
            "health": "/health - Health check",
            "web_interface": "/static/index.html - Web interface for testing"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    
    return {
        "status": "healthy", 
        "service": "SMB Research Agent API",
        "google_api_key_configured": bool(google_api_key),
        "google_api_key_length": len(google_api_key) if google_api_key else 0
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_company(request: AnalysisRequest):
    """
    Analyze an SMB company website to extract comprehensive data for PE evaluation
    
    Args:
        request: Analysis request containing website URL and optional company name
        
    Returns:
        AnalysisResponse with comprehensive company data and optional formatted report
    """
    try:
        # Convert pydantic HttpUrl to string
        website_url = str(request.website_url)
        
        # Perform SMB analysis
        smb_data = await analyze_smb_website(
            website_url=website_url,
            company_name=request.company_name
        )
        
        # Generate formatted report if requested
        report = None
        if request.include_report:
            report = format_smb_report(smb_data)
        
        return AnalysisResponse(
            success=True,
            data=smb_data,
            report=report
        )
        
    except Exception as e:
        # Return error response
        return AnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@app.get("/analyze/{website_url:path}")
async def analyze_company_get(website_url: str, company_name: Optional[str] = None, include_report: bool = False):
    """
    Analyze an SMB company website via GET request (alternative endpoint)
    
    Args:
        website_url: Company website URL (path parameter)
        company_name: Optional company name (query parameter)
        include_report: Whether to include formatted report (query parameter)
        
    Returns:
        AnalysisResponse with comprehensive company data
    """
    try:
        # Ensure URL has protocol
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Create request object
        request = AnalysisRequest(
            website_url=website_url,
            company_name=company_name,
            include_report=include_report
        )
        
        # Use the POST endpoint logic
        return await analyze_company(request)
        
    except Exception as e:
        return AnalysisResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )