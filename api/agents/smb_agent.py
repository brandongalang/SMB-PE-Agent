import os
import asyncio
from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel, GeminiModelSettings
try:
    from google import genai
    from google.genai import types
    NEW_API_AVAILABLE = True
except ImportError:
    import google.generativeai as genai
    NEW_API_AVAILABLE = False
from dotenv import load_dotenv
import json
import logging

# logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.WARNING)

load_dotenv()

# Configure environment for both APIs
gemini_api_key = os.getenv('GEMINI_API_KEY')
google_api_key = gemini_api_key  # Use same key for both APIs

# Configure Google Gemini
if NEW_API_AVAILABLE:
    client = genai.Client(api_key=google_api_key)
    logger.info("Using new Gemini API")
else:
    genai.configure(api_key=google_api_key)

class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"       # >80% confidence, multiple supporting signals
    MEDIUM = "MEDIUM"   # 60-80% confidence, some supporting data
    LOW = "LOW"         # <60% confidence, limited/inferred data

class CompanySize(str, Enum):
    MICRO = "MICRO"           # 1-10 employees, <$1M revenue
    SMALL = "SMALL"           # 11-50 employees, $1M-$10M revenue  
    MEDIUM = "MEDIUM"         # 51-250 employees, $10M-$50M revenue
    LARGE = "LARGE"           # 251-1000 employees, $50M-$200M revenue
    ENTERPRISE = "ENTERPRISE" # 1000+ employees, $200M+ revenue

class SMBConfidenceScores(BaseModel):
    """Confidence scores for key assessment areas.
    
    To add new confidence categories, add them as explicit fields here
    rather than using a dynamic dict structure.
    """
    company_identification: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence in company name and basic info")
    size_estimation: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence in size and employee estimates")
    financial_data: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence in revenue and financial indicators")
    market_position: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence in market and competitive analysis")
    growth_assessment: ConfidenceLevel = Field(default=ConfidenceLevel.MEDIUM, description="Confidence in growth and scalability factors")

class SMBCompanyData(BaseModel):
    """SMB company data extracted from website analysis"""
    
    # Core Identification
    company_name: str = Field(description="Exact company name")
    website_url: str = Field(description="Company website URL")
    business_description: str = Field(description="2-3 sentence description of the business")
    
    # Industry & Market
    primary_industry: str = Field(description="Main industry category")
    industry_keywords: List[str] = Field(description="Relevant industry terms")
    naics_codes: List[str] = Field(description="Applicable NAICS codes")
    target_customers: str = Field(description="B2B, B2C, or Mixed")
    
    # Size & Financial Indicators
    estimated_employees: Optional[int] = Field(description="Employee count estimate")
    estimated_revenue: Optional[float] = Field(description="Annual revenue USD")
    size_band: Optional[CompanySize] = Field(description="Company size classification")
    growth_stage: str = Field(description="startup, growth, mature, established")
    
    # Geographic & Operational
    headquarters_location: str = Field(description="HQ location")
    service_areas: List[str] = Field(description="Geographic markets served")
    
    # Business Model & Operations
    business_model: str = Field(description="How they make money")
    revenue_streams: List[str] = Field(description="Multiple revenue sources")
    key_products_services: List[str] = Field(description="Main offerings")
    competitive_advantages: List[str] = Field(description="What makes them unique")
    
    # Growth & Investment Indicators
    growth_indicators: List[str] = Field(description="Recent expansions, hiring, etc.")
    technology_assets: List[str] = Field(description="IP, software, systems")
    customer_base_description: str = Field(description="Customer concentration, loyalty")
    
    # PE Attractiveness Factors
    scalability_factors: List[str] = Field(description="What makes it scalable")
    market_position: str = Field(description="Market leader, challenger, niche")
    defensibility: List[str] = Field(description="Moats, barriers to entry")
    
    # Financial Health Indicators  
    profitability_indicators: List[str] = Field(description="Signs of profitability/cash flow")
    capital_requirements: str = Field(description="Asset heavy/light, working capital needs")
    
    # Analysis Metadata
    confidence_scores: SMBConfidenceScores = Field(default_factory=SMBConfidenceScores, description="Per-field confidence")
    data_sources: List[str] = Field(description="Where info was found")
    analysis_notes: List[str] = Field(description="Important observations")
    
    # Token Usage
    input_tokens: int = Field(description="Input tokens used")
    output_tokens: int = Field(description="Output tokens used")
    estimated_cost_usd: float = Field(description="Estimated API cost")

class WebsiteAnalysisRequest(BaseModel):
    """Request model for website analysis"""
    website_url: str = Field(description="Company website URL to analyze")
    company_name: Optional[str] = Field(default=None, description="Optional company name if known")

# Global agent variable
_smb_agent = None
_tools_registered = False

def get_smb_agent():
    """Get or create the SMB agent with thinking mode enabled (lazy initialization)"""
    global _smb_agent, _tools_registered
    if _smb_agent is None:
        # Configure thinking mode for better analysis quality
        model_settings = GeminiModelSettings(
            gemini_thinking_config={
                'thinking_budget': 20000,  # Generous budget for complex analysis
                'include_thoughts': True   # Include reasoning for transparency
            }
        )
        
        model = GeminiModel('gemini-2.5-flash-lite-preview-06-17')
        
        _smb_agent = Agent(
            model,
            result_type=SMBCompanyData,
            system_prompt="""
You are an expert SMB (Small-Medium Business) analyst specializing in private equity investment evaluation.

Your task is to analyze company websites and extract comprehensive data that would be relevant for PE investment decisions.

Focus on:
- Business fundamentals and industry classification
- Size and financial indicators 
- Growth potential and scalability
- Market position and competitive advantages
- Investment attractiveness factors

Be conservative with estimates and clearly indicate confidence levels for each assessment.
Provide detailed reasoning for your conclusions.
""",
            model_settings=model_settings
        )
        # Register tools only once when agent is created
        _smb_agent = register_tools(_smb_agent)
        _tools_registered = True
    return _smb_agent

async def google_search_tool(website_url: str, company_name: Optional[str] = None) -> str:
    """Custom search tool using Google Gemini's search grounding"""
    
    if NEW_API_AVAILABLE:
        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(
            tools=[grounding_tool],
            thinking_config=types.ThinkingConfig(thinking_budget=20000)
        )
        
        search_query = f"""
        Search extensively for information about the company at {website_url}.
        {f"Company name: {company_name}." if company_name else ""}
        
        Find information about:
        - Business model and revenue streams
        - Employee count and company size
        - Revenue estimates and financial performance
        - Industry and market position
        - Growth indicators and recent developments
        - Competitive advantages and technology assets
        - Geographic presence and service areas
        """
        
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash-lite-preview-06-17",
                contents=search_query,
                config=config
            )
            return response.text
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    return "Google Search grounding not available with current API setup"

def register_tools(agent):
    """Register tools with the agent"""
    @agent.tool
    async def google_search_company_data(ctx: RunContext[WebsiteAnalysisRequest], website_url: str) -> str:
        """Search for comprehensive company information using Google Search grounding"""
        return await google_search_tool(website_url, ctx.deps.company_name)
    return agent

async def analyze_smb_website(website_url: str, company_name: Optional[str] = None) -> SMBCompanyData:
    """
    Analyze an SMB website to extract comprehensive company data using PydanticAI
    
    Args:
        website_url: Company website URL
        company_name: Optional company name if known
        
    Returns:
        SMBCompanyData with comprehensive analysis
    """
    
    request = WebsiteAnalysisRequest(
        website_url=website_url,
        company_name=company_name
    )
    
    prompt = f"""
    Analyze the following SMB company website to extract comprehensive data for private equity evaluation.

    WEBSITE TO ANALYZE: {website_url}
    {f"COMPANY NAME: {company_name}" if company_name else ""}

    Use the search tool to gather extensive information about this company, including:

    BUSINESS FUNDAMENTALS:
    - Company name, description, and core business
    - Industry classification and market positioning  
    - Products/services offered and revenue streams
    - Target customers and market segments

    SIZE & FINANCIAL INDICATORS:
    - Employee count estimates
    - Revenue estimates (look for press releases, industry reports)
    - Growth indicators (hiring, expansions, new locations)
    - Profitability signals (cash flow, margins, efficiency)

    OPERATIONAL INTELLIGENCE:
    - Geographic presence and service areas
    - Technology stack and competitive advantages
    - Customer base characteristics
    - Operational scale and capacity

    INVESTMENT ATTRACTIVENESS:
    - Scalability factors and growth potential
    - Market position and competitive moats
    - Capital requirements and asset intensity
    - Management quality and experience

    Analysis Guidelines:
    1. Focus on SMB-specific factors (typically $1M-$50M revenue, 10-250 employees)
    2. Look for PE investment attractiveness: recurring revenue, market position, scalability
    3. Be conservative with estimates - only HIGH confidence if multiple sources confirm
    4. Note data limitations for smaller companies
    5. Consider regional/local market dynamics for SMBs

    CONFIDENCE SCORING:
    Provide confidence scores for these five key assessment areas:
    - company_identification: Confidence in company name and basic info
    - size_estimation: Confidence in size and employee estimates
    - financial_data: Confidence in revenue and financial indicators
    - market_position: Confidence in market and competitive analysis
    - growth_assessment: Confidence in growth and scalability factors
    
    Use HIGH (>80% confidence), MEDIUM (60-80%), or LOW (<60%) for each area.
    """
    
    try:
        agent = get_smb_agent()
        result = await agent.run(prompt, deps=request)
        
        # Add token usage estimation (approximate)
        input_tokens = len(prompt) // 4
        output_tokens = len(str(result.output)) // 4
        cost = (input_tokens * 0.000001) + (output_tokens * 0.000002)
        
        # Update the result with token usage
        result.output.input_tokens = input_tokens
        result.output.output_tokens = output_tokens
        result.output.estimated_cost_usd = cost
        
        return result.output
        
    except Exception as e:
        # Return fallback data if analysis fails
        return SMBCompanyData(
            company_name="Analysis Failed",
            website_url=website_url,
            business_description="Could not analyze website",
            primary_industry="Unknown",
            industry_keywords=[],
            naics_codes=[],
            target_customers="Unknown",
            estimated_employees=None,
            estimated_revenue=None,
            size_band=None,
            growth_stage="Unknown",
            headquarters_location="Unknown",
            service_areas=[],
            business_model="Unknown",
            revenue_streams=[],
            key_products_services=[],
            competitive_advantages=[],
            growth_indicators=[],
            technology_assets=[],
            customer_base_description="Unknown",
            scalability_factors=[],
            market_position="Unknown",
            defensibility=[],
            profitability_indicators=[],
            capital_requirements="Unknown",
            confidence_scores=SMBConfidenceScores(
                company_identification=ConfidenceLevel.LOW,
                size_estimation=ConfidenceLevel.LOW,
                financial_data=ConfidenceLevel.LOW,
                market_position=ConfidenceLevel.LOW,
                growth_assessment=ConfidenceLevel.LOW
            ),
            data_sources=[],
            analysis_notes=[f"Analysis failed: {str(e)}"],
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=0.0
        )

def format_smb_report(data: SMBCompanyData) -> str:
    """Format SMB analysis into a readable report"""
    
    report = f"""
{'='*60}
SMB COMPANY ANALYSIS REPORT
{'='*60}

🏢 COMPANY OVERVIEW
Name: {data.company_name}
Website: {data.website_url}
Description: {data.business_description}

🏭 INDUSTRY & MARKET
Industry: {data.primary_industry}
NAICS Codes: {', '.join(data.naics_codes) if data.naics_codes else 'Not determined'}
Target Customers: {data.target_customers}
Market Position: {data.market_position}

📏 SIZE & SCALE
Employees: {f'{data.estimated_employees:,}' if data.estimated_employees else 'Unknown'}
Revenue: {f'${data.estimated_revenue:,.0f}' if data.estimated_revenue else 'Unknown'}
Size Band: {data.size_band.value if data.size_band else 'Unknown'}
Growth Stage: {data.growth_stage}

🌍 OPERATIONS
Headquarters: {data.headquarters_location}
Service Areas: {', '.join(data.service_areas) if data.service_areas else 'Not specified'}

💼 BUSINESS MODEL
Model: {data.business_model}
Revenue Streams: {', '.join(data.revenue_streams) if data.revenue_streams else 'Not identified'}
Products/Services: {', '.join(data.key_products_services) if data.key_products_services else 'Not detailed'}

🚀 INVESTMENT ATTRACTIVENESS
Competitive Advantages:
{chr(10).join(f'• {advantage}' for advantage in data.competitive_advantages) if data.competitive_advantages else '• None identified'}

Scalability Factors:
{chr(10).join(f'• {factor}' for factor in data.scalability_factors) if data.scalability_factors else '• None identified'}

Growth Indicators:
{chr(10).join(f'• {indicator}' for indicator in data.growth_indicators) if data.growth_indicators else '• None identified'}

Defensibility:
{chr(10).join(f'• {defense}' for defense in data.defensibility) if data.defensibility else '• None identified'}

💰 FINANCIAL INDICATORS
Profitability Signals:
{chr(10).join(f'• {signal}' for signal in data.profitability_indicators) if data.profitability_indicators else '• None identified'}

Capital Requirements: {data.capital_requirements}

🎯 CONFIDENCE ASSESSMENT
• Company Identification: {data.confidence_scores.company_identification.value}
• Size Estimation: {data.confidence_scores.size_estimation.value}
• Financial Data: {data.confidence_scores.financial_data.value}
• Market Position: {data.confidence_scores.market_position.value}
• Growth Assessment: {data.confidence_scores.growth_assessment.value}

💵 ANALYSIS COST
Tokens Used: {data.input_tokens:,} input, {data.output_tokens:,} output
Estimated Cost: ${data.estimated_cost_usd:.4f}

📝 DATA SOURCES
{chr(10).join(f'• {source}' for source in data.data_sources) if data.data_sources else '• Website analysis only'}

📋 NOTES
{chr(10).join(f'• {note}' for note in data.analysis_notes) if data.analysis_notes else '• No additional notes'}
{'='*60}
"""
    return report

# Example usage
async def main():
    """Test the PydanticAI SMB agent"""
    
    test_websites = [
        "https://www.freshbooks.com",
        "https://www.hubspot.com"
    ]
    
    print("🔍 PydanticAI SMB Website Analyzer Test")
    print("=" * 50)
    
    for website in test_websites[:1]:  # Test just one for now
        print(f"\nAnalyzing: {website}")
        print("-" * 30)
        
        try:
            smb_data = await analyze_smb_website(website)
            report = format_smb_report(smb_data)
            print(report)
            
        except Exception as e:
            print(f"❌ Error analyzing {website}: {e}")

if __name__ == "__main__":
    asyncio.run(main())