from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
import time
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.smb_agent import analyze_smb_website, SMBCompanyData, format_smb_report, ConfidenceLevel, CompanySize
from models.responses import (
    DetailedAnalysisResponse, SummaryResponse, BulkAnalysisRequest, BulkAnalysisResponse,
    AnalysisStatus, CompanyOverview, BusinessMetrics, InvestmentProfile,
    ConfidenceScores, TokenUsage, AnalysisMetadata
)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

def convert_smb_data_to_detailed_response(smb_data: SMBCompanyData, execution_time: float) -> DetailedAnalysisResponse:
    """Convert SMBCompanyData to DetailedAnalysisResponse"""
    
    # Create company overview
    company_overview = CompanyOverview(
        name=smb_data.company_name,
        description=smb_data.business_description,
        industry=smb_data.primary_industry,
        website=smb_data.website_url,
        headquarters=smb_data.headquarters_location
    )
    
    # Create business metrics
    business_metrics = BusinessMetrics(
        estimated_employees=smb_data.estimated_employees,
        estimated_revenue=smb_data.estimated_revenue,
        size_classification=smb_data.size_band.value if smb_data.size_band else None,
        growth_stage=smb_data.growth_stage,
        business_model=smb_data.business_model
    )
    
    # Create investment profile
    investment_profile = InvestmentProfile(
        scalability_factors=smb_data.scalability_factors,
        competitive_advantages=smb_data.competitive_advantages,
        growth_indicators=smb_data.growth_indicators,
        market_position=smb_data.market_position,
        investment_attractiveness="Calculated based on multiple factors"
    )
    
    # Create confidence scores
    confidence_scores = ConfidenceScores(
        company_identification=smb_data.confidence_scores.company_identification.value,
        size_estimation=smb_data.confidence_scores.size_estimation.value,
        financial_data=smb_data.confidence_scores.financial_data.value,
        market_position=smb_data.confidence_scores.market_position.value,
        growth_assessment=smb_data.confidence_scores.growth_assessment.value,
        overall_confidence=ConfidenceLevel.MEDIUM.value
    )
    
    # Create token usage
    token_usage = TokenUsage(
        input_tokens=smb_data.input_tokens,
        output_tokens=smb_data.output_tokens,
        total_tokens=smb_data.input_tokens + smb_data.output_tokens,
        estimated_cost_usd=smb_data.estimated_cost_usd
    )
    
    # Create metadata
    metadata = AnalysisMetadata(
        execution_time_seconds=execution_time,
        analysis_timestamp=datetime.now().isoformat(),
        model_used="gemini-2.0-flash-exp",
        api_version="1.0.0"
    )
    
    return DetailedAnalysisResponse(
        success=True,
        status=AnalysisStatus.SUCCESS,
        company_overview=company_overview,
        business_metrics=business_metrics,
        investment_profile=investment_profile,
        confidence_scores=confidence_scores,
        token_usage=token_usage,
        metadata=metadata
    )

def convert_smb_data_to_summary(smb_data: SMBCompanyData) -> SummaryResponse:
    """Convert SMBCompanyData to SummaryResponse"""
    
    # Calculate investment score (0-10) based on various factors
    investment_score = 5.0  # Base score
    
    # Adjust based on size (SMBs are more attractive to PE)
    if smb_data.size_band in [CompanySize.SMALL, CompanySize.MEDIUM]:
        investment_score += 1.5
    
    # Adjust based on growth indicators
    if len(smb_data.growth_indicators) > 2:
        investment_score += 1.0
    
    # Adjust based on competitive advantages
    if len(smb_data.competitive_advantages) > 1:
        investment_score += 1.0
    
    # Adjust based on scalability factors
    if len(smb_data.scalability_factors) > 1:
        investment_score += 0.5
    
    # Cap at 10
    investment_score = min(investment_score, 10.0)
    
    # Create key highlights
    key_highlights = []
    if smb_data.estimated_revenue:
        key_highlights.append(f"Revenue: ${smb_data.estimated_revenue:,.0f}")
    if smb_data.estimated_employees:
        key_highlights.append(f"Employees: {smb_data.estimated_employees:,}")
    if smb_data.competitive_advantages:
        key_highlights.append(f"Key advantage: {smb_data.competitive_advantages[0]}")
    if smb_data.growth_indicators:
        key_highlights.append(f"Growth signal: {smb_data.growth_indicators[0]}")
    
    return SummaryResponse(
        success=True,
        company_name=smb_data.company_name,
        industry=smb_data.primary_industry,
        size_estimate=smb_data.size_band.value if smb_data.size_band else "Unknown",
        revenue_estimate=f"${smb_data.estimated_revenue:,.0f}" if smb_data.estimated_revenue else "Unknown",
        investment_score=investment_score,
        key_highlights=key_highlights,
        confidence_level=ConfidenceLevel.MEDIUM.value,
        analysis_cost=smb_data.estimated_cost_usd
    )

@router.post("/detailed", response_model=DetailedAnalysisResponse)
async def detailed_analysis(
    website_url: str,
    company_name: Optional[str] = None,
    include_report: bool = False,
    include_raw_data: bool = False
):
    """
    Perform detailed SMB analysis with comprehensive structured data
    """
    start_time = time.time()
    
    try:
        # Ensure URL has protocol
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Perform analysis
        smb_data = await analyze_smb_website(website_url, company_name)
        
        execution_time = time.time() - start_time
        
        # Convert to detailed response
        response = convert_smb_data_to_detailed_response(smb_data, execution_time)
        
        # Add formatted report if requested
        if include_report:
            response.formatted_report = format_smb_report(smb_data)
        
        # Add raw data if requested
        if include_raw_data:
            response.raw_data = smb_data.dict()
        
        return response
        
    except Exception as e:
        return DetailedAnalysisResponse(
            success=False,
            status=AnalysisStatus.FAILED,
            error=f"Analysis failed: {str(e)}"
        )

@router.post("/summary", response_model=SummaryResponse)
async def summary_analysis(
    website_url: str,
    company_name: Optional[str] = None
):
    """
    Perform quick SMB analysis with summary information
    """
    try:
        # Ensure URL has protocol
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        # Perform analysis
        smb_data = await analyze_smb_website(website_url, company_name)
        
        # Convert to summary response
        return convert_smb_data_to_summary(smb_data)
        
    except Exception as e:
        return SummaryResponse(
            success=False,
            error=f"Analysis failed: {str(e)}"
        )

@router.post("/with-pe-matches")
async def analysis_with_pe_matches(
    website_url: str,
    company_name: Optional[str] = None,
    top_k: int = 10,
    include_report: bool = True
):
    """
    Perform SMB analysis with PE fund matches and executive report using SMB agent
    """
    try:
        # Import here to avoid circular dependencies
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from pe_fund_matcher import FundMatcher
        
        # Ensure URL has protocol
        if not website_url.startswith(('http://', 'https://')):
            website_url = 'https://' + website_url
        
        start_time = time.time()
        
        # Perform SMB analysis
        smb_data = await analyze_smb_website(website_url, company_name)
        
        # Check if SMB analysis failed
        if (smb_data.company_name in ["Analysis Failed", "Unknown", ""] or 
            smb_data.business_description in ["Could not analyze website", ""] or
            not smb_data.company_name or not smb_data.business_description):
            return {
                "success": False,
                "error": "Failed to analyze website - unable to extract company information",
                "company_data": {"company_name": "Analysis Failed"},
                "pe_matches": [],
                "confidence_level": "LOW"
            }
        
        # Get PE fund matches directly with SMB data
        matcher = FundMatcher()
        matches_raw = matcher.match(smb_data, k=top_k)
        
        execution_time = time.time() - start_time
        
        # Generate executive report using SMB data
        executive_report = None
        if include_report:
            executive_report = _generate_smb_executive_report(smb_data, matches_raw)
        
        # Prepare response
        response = {
            "success": True,
            "company_data": {
                "company_name": smb_data.company_name,
                "industry": smb_data.primary_industry,
                "size": smb_data.size_band.value if smb_data.size_band else "Unknown",
                "headquarters": smb_data.headquarters_location,
                "business_description": smb_data.business_description,
                "key_strengths": smb_data.competitive_advantages,
                "growth_stage": smb_data.growth_stage,
                "estimated_employees": smb_data.estimated_employees,
                "estimated_revenue": smb_data.estimated_revenue
            },
            "pe_matches": matches_raw,
            "confidence_level": "HIGH",
            "total_tokens": smb_data.input_tokens + smb_data.output_tokens,
            "estimated_cost": smb_data.estimated_cost_usd,
            "execution_time": execution_time
        }
        
        if executive_report:
            response["executive_report"] = executive_report
            
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Analysis failed: {str(e)}",
            "company_data": {"company_name": "Analysis Failed"},
            "pe_matches": [],
            "confidence_level": "LOW"
        }

def _generate_smb_executive_report(smb_data: SMBCompanyData, pe_matches: List[Dict]) -> str:
    """Generate executive report using SMB agent data"""
    
    # Build company snapshot from SMB data
    company_snapshot = f"""
## Company Snapshot

• **Company Name**: {smb_data.company_name}
• **Industry**: {smb_data.primary_industry}
• **Business Model**: {smb_data.business_model}
• **Size Band**: {smb_data.size_band.value if smb_data.size_band else 'Unknown'}
• **Headquarters**: {smb_data.headquarters_location}
• **Key Strengths**:
"""
    
    for strength in smb_data.competitive_advantages[:4]:
        company_snapshot += f"  • {strength}\n"
    
    # Build PE fund shortlist
    fund_shortlist = "\n## PE Fund Shortlist\n\n"
    
    for idx, match in enumerate(pe_matches[:5], 1):
        fund = match.get('fund', {})
        distance = match.get('distance', 0)
        
        # Determine fit level
        if distance < 1.0:
            fit = "High"
        elif distance < 1.5:
            fit = "Medium"
        else:
            fit = "Exploratory"
        
        fund_shortlist += f"""### {idx}. {fund.get('fund_name', 'Unknown Fund')} | {fund.get('strategy', 'Unknown')} | Fit: {fit}

• {fund.get('focus_sector', 'Unknown')} | {fund.get('focus_stage', 'Unknown')} | {fund.get('focus_geo', 'Unknown')} | AUM: ${fund.get('aum_musd', 0):.2f}M
• **Rationale**: {match.get('match_reason', 'Semantic alignment based on company profile and fund focus areas')}

"""
    
    return f"""# M&A Acquirer Shortlist Report

{company_snapshot}

{fund_shortlist}

*Generated by SMB Analyzer CLI - Powered by Gemini 2.5 Flash*
"""

@router.post("/bulk", response_model=BulkAnalysisResponse)
async def bulk_analysis(request: BulkAnalysisRequest):
    """
    Analyze multiple companies in bulk
    """
    start_time = time.time()
    results = []
    successful = 0
    failed = 0
    total_cost = 0.0
    
    for website_url in request.websites:
        try:
            # Ensure URL has protocol
            if not website_url.startswith(('http://', 'https://')):
                website_url = 'https://' + website_url
            
            # Perform analysis
            smb_data = await analyze_smb_website(website_url)
            analysis_time = time.time() - start_time
            
            # Convert to detailed response
            response = convert_smb_data_to_detailed_response(smb_data, analysis_time)
            
            # Add formatted report if requested
            if request.include_reports:
                response.formatted_report = format_smb_report(smb_data)
            
            results.append(response)
            successful += 1
            total_cost += smb_data.estimated_cost_usd
            
        except Exception as e:
            # Add failed analysis
            failed_response = DetailedAnalysisResponse(
                success=False,
                status=AnalysisStatus.FAILED,
                error=f"Analysis failed for {website_url}: {str(e)}"
            )
            results.append(failed_response)
            failed += 1
    
    execution_time = time.time() - start_time
    
    return BulkAnalysisResponse(
        success=True,
        total_analyzed=len(request.websites),
        successful_analyses=successful,
        failed_analyses=failed,
        results=results,
        total_cost=total_cost,
        execution_time_seconds=execution_time,
        summary=f"Analyzed {len(request.websites)} companies: {successful} successful, {failed} failed"
    )

@router.get("/company/{website_url:path}", response_model=DetailedAnalysisResponse)
async def analyze_company_by_url(
    website_url: str,
    company_name: Optional[str] = Query(None, description="Optional company name"),
    analysis_type: str = Query("detailed", description="Analysis type: detailed, summary"),
    include_report: bool = Query(False, description="Include formatted report"),
    include_raw_data: bool = Query(False, description="Include raw analysis data")
):
    """
    Analyze company by URL path parameter
    """
    if analysis_type == "summary":
        return await summary_analysis(website_url, company_name)
    else:
        return await detailed_analysis(website_url, company_name, include_report, include_raw_data)