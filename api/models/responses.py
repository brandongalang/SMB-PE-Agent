from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
from enum import Enum

class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"

class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str = Field(description="Error message")
    error_code: Optional[str] = Field(default=None, description="Error code for debugging")
    timestamp: Optional[str] = Field(default=None, description="Error timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(description="Service health status")
    service: str = Field(description="Service name")
    version: Optional[str] = Field(default="1.0.0", description="API version")
    uptime: Optional[float] = Field(default=None, description="Service uptime in seconds")

class AnalysisMetadata(BaseModel):
    """Analysis execution metadata"""
    execution_time_seconds: Optional[float] = Field(default=None, description="Analysis execution time")
    model_used: str = Field(default="gemini-2.0-flash-exp", description="AI model used")
    api_version: str = Field(default="1.0.0", description="API version")
    analysis_timestamp: Optional[str] = Field(default=None, description="When analysis was performed")

class TokenUsage(BaseModel):
    """Token usage and cost information"""
    input_tokens: int = Field(description="Input tokens consumed")
    output_tokens: int = Field(description="Output tokens generated")
    total_tokens: int = Field(description="Total tokens used")
    estimated_cost_usd: float = Field(description="Estimated API cost in USD")

class ConfidenceScores(BaseModel):
    """Confidence scores for different analysis aspects"""
    company_identification: str = Field(description="Confidence in company identification")
    size_estimation: str = Field(description="Confidence in size estimates")
    financial_data: str = Field(description="Confidence in financial data")
    market_position: str = Field(description="Confidence in market position assessment")
    growth_assessment: str = Field(description="Confidence in growth assessment")
    overall_confidence: Optional[str] = Field(default=None, description="Overall analysis confidence")

class CompanyOverview(BaseModel):
    """High-level company overview"""
    name: str = Field(description="Company name")
    description: str = Field(description="Business description")
    industry: str = Field(description="Primary industry")
    website: str = Field(description="Company website")
    headquarters: str = Field(description="Headquarters location")

class BusinessMetrics(BaseModel):
    """Key business metrics and indicators"""
    estimated_employees: Optional[int] = Field(default=None, description="Estimated employee count")
    estimated_revenue: Optional[float] = Field(default=None, description="Estimated annual revenue (USD)")
    size_classification: Optional[str] = Field(default=None, description="Company size classification")
    growth_stage: Optional[str] = Field(default=None, description="Company growth stage")
    business_model: Optional[str] = Field(default=None, description="Business model type")

class InvestmentProfile(BaseModel):
    """Investment attractiveness profile"""
    scalability_factors: List[str] = Field(default_factory=list, description="Factors that enable scalability")
    competitive_advantages: List[str] = Field(default_factory=list, description="Key competitive advantages")
    growth_indicators: List[str] = Field(default_factory=list, description="Recent growth indicators")
    market_position: Optional[str] = Field(default=None, description="Market position assessment")
    investment_attractiveness: Optional[str] = Field(default=None, description="Overall investment attractiveness")

class DetailedAnalysisResponse(BaseModel):
    """Comprehensive analysis response with structured data"""
    success: bool = Field(description="Whether analysis was successful")
    status: AnalysisStatus = Field(description="Analysis status")
    
    # Core company data
    company_overview: Optional[CompanyOverview] = Field(default=None, description="Company overview")
    business_metrics: Optional[BusinessMetrics] = Field(default=None, description="Business metrics")
    investment_profile: Optional[InvestmentProfile] = Field(default=None, description="Investment profile")
    
    # Analysis metadata
    confidence_scores: Optional[ConfidenceScores] = Field(default=None, description="Confidence scores")
    token_usage: Optional[TokenUsage] = Field(default=None, description="Token usage information")
    metadata: Optional[AnalysisMetadata] = Field(default=None, description="Analysis metadata")
    
    # Optional formatted report
    formatted_report: Optional[str] = Field(default=None, description="Human-readable analysis report")
    
    # Raw data (if requested)
    raw_data: Optional[Dict[str, Any]] = Field(default=None, description="Raw analysis data")
    
    # Error information
    error: Optional[str] = Field(default=None, description="Error message if analysis failed")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")

class SummaryResponse(BaseModel):
    """Simplified response for quick analysis"""
    success: bool = Field(description="Whether analysis was successful")
    company_name: Optional[str] = Field(default=None, description="Company name")
    industry: Optional[str] = Field(default=None, description="Primary industry")
    size_estimate: Optional[str] = Field(default=None, description="Company size estimate")
    revenue_estimate: Optional[str] = Field(default=None, description="Revenue estimate")
    investment_score: Optional[float] = Field(default=None, description="Investment attractiveness score (0-10)")
    key_highlights: List[str] = Field(default_factory=list, description="Key analysis highlights")
    confidence_level: Optional[str] = Field(default=None, description="Overall confidence level")
    analysis_cost: Optional[float] = Field(default=None, description="Analysis cost (USD)")
    error: Optional[str] = Field(default=None, description="Error message if failed")

class BulkAnalysisRequest(BaseModel):
    """Request model for analyzing multiple companies"""
    websites: List[str] = Field(description="List of website URLs to analyze", max_items=10)
    include_reports: bool = Field(default=False, description="Whether to include formatted reports")
    analysis_type: str = Field(default="standard", description="Type of analysis to perform")

class BulkAnalysisResponse(BaseModel):
    """Response model for bulk analysis"""
    success: bool = Field(description="Whether bulk analysis was successful")
    total_analyzed: int = Field(description="Number of companies analyzed")
    successful_analyses: int = Field(description="Number of successful analyses")
    failed_analyses: int = Field(description="Number of failed analyses")
    results: List[DetailedAnalysisResponse] = Field(description="Individual analysis results")
    total_cost: Optional[float] = Field(default=None, description="Total analysis cost (USD)")
    execution_time_seconds: Optional[float] = Field(default=None, description="Total execution time")
    summary: Optional[str] = Field(default=None, description="Summary of bulk analysis results")