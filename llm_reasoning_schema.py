#!/usr/bin/env python3
"""
LLM Reasoning Layer Schema Design for PE Fund Analysis

This module defines the structured output schema that Gemini 2.5 Flash
should return for sophisticated PE fund ranking and analysis.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum

class InvestmentFitLevel(str, Enum):
    """Investment fit assessment levels"""
    EXCELLENT = "excellent"      # 90-100% fit
    STRONG = "strong"           # 80-89% fit  
    GOOD = "good"               # 70-79% fit
    MODERATE = "moderate"       # 60-69% fit
    WEAK = "weak"               # 50-59% fit
    POOR = "poor"               # <50% fit

class RiskLevel(str, Enum):
    """Risk assessment levels"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class InvestmentThesis(BaseModel):
    """Core investment thesis for a PE fund match"""
    
    headline: str = Field(description="One-sentence investment thesis")
    strategic_rationale: str = Field(description="Why this fund makes strategic sense")
    value_creation_potential: str = Field(description="How the fund would drive value")
    competitive_advantages: List[str] = Field(description="Fund's unique advantages for this deal")

class RiskAssessment(BaseModel):
    """Risk analysis for fund-company pairing"""
    
    overall_risk: RiskLevel = Field(description="Overall risk level")
    key_risks: List[str] = Field(description="Top 3-5 risks")
    mitigation_factors: List[str] = Field(description="How fund mitigates risks")
    deal_breakers: List[str] = Field(description="Potential show-stoppers", default_factory=list)

class FinancialProjections(BaseModel):
    """Financial analysis and projections"""
    
    estimated_valuation_multiple: Optional[float] = Field(description="Revenue multiple estimate")
    investment_size_estimate: Optional[float] = Field(description="Likely investment size in millions")
    ownership_stake_estimate: Optional[float] = Field(description="Estimated ownership percentage")
    exit_timeline: str = Field(description="Expected exit timeframe")
    return_potential: str = Field(description="Expected return profile")

class StrategicAlignment(BaseModel):
    """Strategic fit analysis"""
    
    sector_match_score: int = Field(description="Sector alignment 0-100", ge=0, le=100)
    stage_match_score: int = Field(description="Stage alignment 0-100", ge=0, le=100)
    size_match_score: int = Field(description="Size compatibility 0-100", ge=0, le=100)
    geographic_match_score: int = Field(description="Geographic fit 0-100", ge=0, le=100)
    strategic_synergies: List[str] = Field(description="Specific synergy opportunities")

class PEFundRanking(BaseModel):
    """Individual PE fund ranking with detailed reasoning"""
    
    # Core identification
    fund_id: str = Field(description="PE fund database ID") 
    rank: int = Field(description="Ranking position 1-10")
    overall_score: int = Field(description="Overall fit score 0-100", ge=0, le=100)
    fit_level: InvestmentFitLevel = Field(description="Categorical fit assessment")
    
    # Investment analysis
    investment_thesis: InvestmentThesis = Field(description="Core investment thesis")
    strategic_alignment: StrategicAlignment = Field(description="Strategic fit breakdown")
    risk_assessment: RiskAssessment = Field(description="Risk analysis")
    financial_projections: FinancialProjections = Field(description="Financial estimates")
    
    # Competitive positioning
    unique_value_proposition: str = Field(description="What makes this fund special for this deal")
    competitive_positioning: str = Field(description="How this fund compares to others")
    
    # Process insights
    likelihood_of_interest: int = Field(description="Fund interest probability 0-100", ge=0, le=100)
    negotiation_leverage: str = Field(description="Negotiation dynamics assessment")

class MarketContext(BaseModel):
    """Market and competitive context analysis"""
    
    sector_trends: List[str] = Field(description="Relevant sector trends")
    competitive_landscape: str = Field(description="Competitive environment assessment")
    market_timing: str = Field(description="Market timing considerations")
    valuation_environment: str = Field(description="Current valuation trends")

class ProcessRecommendations(BaseModel):
    """Strategic fundraising process recommendations"""
    
    recommended_approach: str = Field(description="Optimal fundraising strategy")
    sequencing_strategy: str = Field(description="How to sequence fund outreach")
    key_negotiation_points: List[str] = Field(description="Critical negotiation items")
    timeline_recommendations: str = Field(description="Suggested process timeline")

class LLMReasoningOutput(BaseModel):
    """Complete LLM reasoning output structure"""
    
    # Executive summary
    executive_summary: str = Field(description="2-3 sentence overall assessment")
    investment_attractiveness: InvestmentFitLevel = Field(description="Overall investment attractiveness")
    
    # Core rankings
    fund_rankings: List[PEFundRanking] = Field(description="Top 10 ranked PE funds with reasoning")
    
    # Market context
    market_context: MarketContext = Field(description="Market and timing analysis")
    
    # Strategic recommendations  
    process_recommendations: ProcessRecommendations = Field(description="Fundraising strategy advice")
    
    # Meta information
    analysis_confidence: int = Field(description="Confidence in analysis 0-100", ge=0, le=100)
    key_assumptions: List[str] = Field(description="Critical assumptions made")
    additional_diligence_needed: List[str] = Field(description="Areas needing more research")

# Example usage for prompt engineering
EXAMPLE_OUTPUT = {
    "executive_summary": "HealthTech Analytics represents a compelling growth equity opportunity with strong defensive characteristics and scalable technology platform.",
    "investment_attractiveness": "strong", 
    "fund_rankings": [
        {
            "fund_id": "adv_12345",
            "rank": 1,
            "overall_score": 92,
            "fit_level": "excellent",
            "investment_thesis": {
                "headline": "Premier healthcare technology growth story with defensive moat and proven scalability.",
                "strategic_rationale": "Perfect intersection of healthcare digitization trend and AI/ML expertise with established customer base.",
                "value_creation_potential": "Accelerate market expansion through portfolio synergies and operational excellence in customer success.",
                "competitive_advantages": [
                    "Deep healthcare domain expertise and regulatory navigation",
                    "Portfolio company synergies with 15+ healthcare systems", 
                    "Proven track record in healthcare IT value creation"
                ]
            },
            "strategic_alignment": {
                "sector_match_score": 95,
                "stage_match_score": 90,
                "size_match_score": 85,
                "geographic_match_score": 80,
                "strategic_synergies": [
                    "Cross-selling opportunities with portfolio healthcare systems",
                    "Clinical advisory board access for product development",
                    "Regulatory expertise for compliance automation"
                ]
            },
            "risk_assessment": {
                "overall_risk": "medium",
                "key_risks": [
                    "Healthcare regulation changes",
                    "Customer concentration in large health systems", 
                    "Competition from Big Tech healthcare initiatives"
                ],
                "mitigation_factors": [
                    "Fund's regulatory expertise and compliance focus",
                    "Diversification through portfolio network expansion",
                    "Defensive positioning through clinical workflow integration"
                ],
                "deal_breakers": []
            },
            "financial_projections": {
                "estimated_valuation_multiple": 8.5,
                "investment_size_estimate": 25.0,
                "ownership_stake_estimate": 35.0,
                "exit_timeline": "4-6 years",
                "return_potential": "3-5x with IPO or strategic exit potential"
            },
            "unique_value_proposition": "Only fund with healthcare-specific expertise and established health system relationships.",
            "competitive_positioning": "Clear differentiation through sector focus vs. generalist growth funds.",
            "likelihood_of_interest": 88,
            "negotiation_leverage": "High - fund actively seeking healthcare tech investments in this size range."
        }
    ],
    "market_context": {
        "sector_trends": [
            "Healthcare digital transformation acceleration post-COVID",
            "Value-based care driving technology adoption",
            "AI/ML integration becoming table stakes"
        ],
        "competitive_landscape": "Consolidating market with scale advantages to AI-native platforms",
        "market_timing": "Optimal timing with healthcare IT budgets recovering and AI adoption accelerating", 
        "valuation_environment": "Growth tech multiples compressed but healthcare tech maintaining premium"
    },
    "process_recommendations": {
        "recommended_approach": "Lead with healthcare-focused funds, create competitive tension with generalist growth funds",
        "sequencing_strategy": "Start with KKR Healthcare, add General Atlantic for validation, include sector specialists",
        "key_negotiation_points": [
            "Board composition with healthcare expertise",
            "Portfolio company collaboration agreements",
            "International expansion support commitments"
        ],
        "timeline_recommendations": "6-8 week process aligned with Q2 healthcare budget cycles"
    },
    "analysis_confidence": 85,
    "key_assumptions": [
        "Current growth trajectory sustainable through market expansion",
        "Healthcare regulatory environment remains stable", 
        "AI/ML capabilities provide sustainable competitive advantage"
    ],
    "additional_diligence_needed": [
        "Clinical outcome data validation",
        "Competitive moat sustainability analysis",
        "International market opportunity quantification"
    ]
}

if __name__ == "__main__":
    # Validate schema
    try:
        output = LLMReasoningOutput(**EXAMPLE_OUTPUT)
        print("✅ Schema validation successful")
        print(f"📊 Example output has {len(output.fund_rankings)} fund rankings")
        print(f"🎯 Overall investment attractiveness: {output.investment_attractiveness}")
        print(f"🔍 Analysis confidence: {output.analysis_confidence}%")
    except Exception as e:
        print(f"❌ Schema validation failed: {e}")