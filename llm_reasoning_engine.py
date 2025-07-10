#!/usr/bin/env python3
"""
LLM Reasoning Engine for PE Fund Analysis

This module implements the LLM-powered reasoning layer that takes SMB company data
and PE fund candidates to generate sophisticated investment analysis and rankings.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

try:
    from google import genai
    from google.genai import types
    NEW_API_AVAILABLE = True
except ImportError:
    import google.generativeai as genai
    NEW_API_AVAILABLE = False

from llm_reasoning_schema import LLMReasoningOutput, PEFundRanking
from api.agents.smb_agent import SMBCompanyData
from enhanced_fund_matcher import EnhancedFundMatcher
from adv_semantic_layer import AdvSemanticSearcher

logger = logging.getLogger(__name__)

class LLMRankingReasoner:
    """
    LLM-powered PE fund ranking and analysis engine using Gemini 2.5 Flash.
    
    Takes SMB company data and PE fund candidates, generates sophisticated
    investment analysis with rankings, reasoning, and strategic recommendations.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM reasoning engine with Gemini 2.5 Flash"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable required")
        
        # Configure Gemini
        if NEW_API_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.0-flash-exp"
        else:
            genai.configure(api_key=self.api_key)
            self.model_name = "gemini-1.5-pro"
        
        logger.info(f"Initialized LLM reasoning engine with {self.model_name}")
    
    async def analyze_and_rank_funds(
        self,
        smb_data: SMBCompanyData,
        pe_fund_candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> LLMReasoningOutput:
        """
        Generate sophisticated LLM analysis and ranking of PE funds.
        
        Args:
            smb_data: Rich SMB company analysis data
            pe_fund_candidates: List of PE fund candidates with full data
            top_k: Number of top funds to rank (max 10)
            
        Returns:
            Structured LLM reasoning output with rankings and analysis
        """
        
        logger.info(f"Starting LLM analysis for {smb_data.company_name} with {len(pe_fund_candidates)} PE fund candidates")
        
        # Build comprehensive analysis prompt
        prompt = self._build_investment_analysis_prompt(smb_data, pe_fund_candidates, top_k)
        
        # Generate LLM analysis with structured output
        llm_response = await self._call_gemini_with_structured_output(prompt)
        
        # Validate and return structured output
        try:
            reasoning_output = LLMReasoningOutput(**llm_response)
            
            logger.info(f"LLM analysis completed: {len(reasoning_output.fund_rankings)} funds ranked, "
                       f"confidence: {reasoning_output.analysis_confidence}%")
        except Exception as e:
            logger.error(f"Failed to validate LLM response: {e}")
            
            # Try to fix common issues
            if 'fund_rankings' in llm_response:
                for ranking in llm_response['fund_rankings']:
                    # Fix risk level enum issues
                    if 'risk_assessment' in ranking and 'overall_risk' in ranking['risk_assessment']:
                        risk_val = ranking['risk_assessment']['overall_risk'].lower()
                        if risk_val not in ['low', 'medium', 'high']:
                            logger.warning(f"Invalid risk level '{risk_val}', defaulting to 'medium'")
                            ranking['risk_assessment']['overall_risk'] = 'medium'
                    
                    # Fix fit level enum issues
                    if 'fit_level' in ranking:
                        fit_val = ranking['fit_level'].lower()
                        if fit_val not in ['excellent', 'strong', 'good', 'moderate', 'weak', 'poor']:
                            logger.warning(f"Invalid fit level '{fit_val}', defaulting to 'good'")
                            ranking['fit_level'] = 'good'
                
                # Fix investment attractiveness enum
                if 'investment_attractiveness' in llm_response:
                    attract_val = llm_response['investment_attractiveness'].lower()
                    if attract_val not in ['excellent', 'strong', 'good', 'moderate', 'weak', 'poor']:
                        logger.warning(f"Invalid investment attractiveness '{attract_val}', defaulting to 'moderate'")
                        llm_response['investment_attractiveness'] = 'moderate'
            
            # Try validation again
            try:
                reasoning_output = LLMReasoningOutput(**llm_response)
                logger.info(f"LLM response fixed and validated: {len(reasoning_output.fund_rankings)} funds ranked")
            except Exception as e2:
                logger.error(f"Failed to validate even after fixes: {e2}")
                logger.error(f"Problematic response: {json.dumps(llm_response, indent=2)[:1000]}")
                raise
        
        return reasoning_output
    
    def _build_investment_analysis_prompt(
        self,
        smb_data: SMBCompanyData,
        pe_funds: List[Dict[str, Any]],
        top_k: int
    ) -> str:
        """Build comprehensive investment analysis prompt for Gemini."""
        
        # Format company profile
        company_profile = self._format_company_profile(smb_data)
        
        # Format PE fund candidates
        pe_fund_profiles = self._format_pe_fund_profiles(pe_funds)
        
        # Build sophisticated prompt
        prompt = f"""You are a senior managing director at a top-tier investment bank specializing in private equity placements. Analyze the following company and PE fund candidates to provide sophisticated investment analysis.

**ANALYSIS OBJECTIVE:**
Rank the top {top_k} PE funds most suitable for this company and provide detailed investment reasoning, strategic analysis, and process recommendations.

**COMPANY PROFILE:**
{company_profile}

**PE FUND CANDIDATES:**
{pe_fund_profiles}

**ANALYSIS FRAMEWORK:**
Evaluate each fund across these dimensions:
1. **Strategic Fit**: Sector focus, stage alignment, geographic presence
2. **Investment Capacity**: Fund size, check size range, portfolio construction
3. **Value Creation Potential**: Operational expertise, network effects, portfolio synergies
4. **Market Position**: Fund reputation, track record, competitive advantages
5. **Risk Assessment**: Potential concerns, mitigation factors, deal breakers
6. **Financial Alignment**: Valuation approach, return expectations, exit strategy
7. **Process Dynamics**: Likelihood of interest, competitive positioning, negotiation leverage

**OUTPUT REQUIREMENTS:**
Return a structured JSON response following this exact schema:

{{
  "executive_summary": "2-3 sentence overall investment assessment",
  "investment_attractiveness": "excellent|strong|good|moderate|weak|poor",
  "fund_rankings": [
    {{
      "fund_id": "40",  // Use the exact ID from "Fund ID: 40" in the profiles, NOT candidate numbers
      "rank": 1,
      "overall_score": 95,
      "fit_level": "excellent|strong|good|moderate|weak|poor",
      "investment_thesis": {{
        "headline": "One-sentence investment thesis",
        "strategic_rationale": "Why this fund makes strategic sense",
        "value_creation_potential": "How fund would drive value",
        "competitive_advantages": ["List of fund's unique advantages"]
      }},
      "strategic_alignment": {{
        "sector_match_score": 95,
        "stage_match_score": 90,
        "size_match_score": 85,
        "geographic_match_score": 80,
        "strategic_synergies": ["Specific synergy opportunities"]
      }},
      "risk_assessment": {{
        "overall_risk": "low|medium|high",
        "key_risks": ["Top 3-5 risks"],
        "mitigation_factors": ["How fund mitigates risks"],
        "deal_breakers": ["Potential show-stoppers"]
      }},
      "financial_projections": {{
        "estimated_valuation_multiple": 8.5,
        "investment_size_estimate": 25.0,
        "ownership_stake_estimate": 35.0,
        "exit_timeline": "Expected exit timeframe",
        "return_potential": "Expected return profile"
      }},
      "unique_value_proposition": "What makes this fund special",
      "competitive_positioning": "How fund compares to others",
      "likelihood_of_interest": 88,
      "negotiation_leverage": "Negotiation dynamics assessment"
    }}
  ],
  "market_context": {{
    "sector_trends": ["Relevant sector trends"],
    "competitive_landscape": "Competitive environment assessment",
    "market_timing": "Market timing considerations",
    "valuation_environment": "Current valuation trends"
  }},
  "process_recommendations": {{
    "recommended_approach": "Optimal fundraising strategy",
    "sequencing_strategy": "How to sequence fund outreach",
    "key_negotiation_points": ["Critical negotiation items"],
    "timeline_recommendations": "Suggested process timeline"
  }},
  "analysis_confidence": 85,
  "key_assumptions": ["Critical assumptions made"],
  "additional_diligence_needed": ["Areas needing more research"]
}}

**CRITICAL INSTRUCTIONS:**
1. Use the EXACT fund_id values shown after "Fund ID:" in the candidate profiles
2. Since funds are deduplicated, Fund IDs are simple numbers: "1", "2", "3", etc.
3. For example, if you see "Fund ID: 1", use "1" as the fund_id in your response
4. DO NOT use database IDs or any other identifiers - ONLY use the Fund ID number shown
5. Rank exactly {top_k} funds (or fewer if insufficient candidates)
6. Provide quantitative scores (0-100) for all scoring fields
7. Be specific and actionable in all reasoning
8. Consider real market dynamics and investment criteria
9. Focus on strategic fit over pure financial metrics
10. Identify concrete value creation opportunities
11. Assess realistic risks and mitigation strategies
12. **ENUM VALUES MUST BE EXACT:** 
    - overall_risk: ONLY use "low", "medium", or "high" - NO other values like "very high"
    - fit_level: ONLY use "excellent", "strong", "good", "moderate", "weak", or "poor"
    - investment_attractiveness: ONLY use "excellent", "strong", "good", "moderate", "weak", or "poor"

Generate your analysis now:"""

        return prompt
    
    def _format_company_profile(self, smb_data: SMBCompanyData) -> str:
        """Format SMB company data for LLM analysis."""
        
        return f"""
**Company Name:** {smb_data.company_name}
**Industry:** {smb_data.primary_industry}
**Business Description:** {smb_data.business_description}

**Financial Profile:**
- Estimated Revenue: ${smb_data.estimated_revenue:,.0f} if smb_data.estimated_revenue else 'Not disclosed'
- Estimated Employees: {smb_data.estimated_employees:,} if smb_data.estimated_employees else 'Not disclosed'
- Size Band: {smb_data.size_band.value if smb_data.size_band else 'Unknown'}
- Growth Stage: {smb_data.growth_stage}

**Business Model & Operations:**
- Business Model: {smb_data.business_model}
- Revenue Streams: {', '.join(smb_data.revenue_streams)}
- Key Products/Services: {', '.join(smb_data.key_products_services)}
- Target Customers: {smb_data.target_customers}

**Market Position & Competitive Advantages:**
- Market Position: {smb_data.market_position}
- Competitive Advantages: {', '.join(smb_data.competitive_advantages)}
- Growth Indicators: {', '.join(smb_data.growth_indicators)}
- Scalability Factors: {', '.join(smb_data.scalability_factors)}

**Geographic & Operational:**
- Headquarters: {smb_data.headquarters_location}
- Service Areas: {', '.join(smb_data.service_areas)}

**Technology & Innovation:**
- Technology Assets: {', '.join(smb_data.technology_assets)}
- Defensibility: {smb_data.defensibility}

**Financial Health:**
- Profitability Indicators: {', '.join(smb_data.profitability_indicators)}
- Capital Requirements: {smb_data.capital_requirements}

**Industry Context:**
- Industry Keywords: {', '.join(smb_data.industry_keywords)}
- NAICS Codes: {', '.join(smb_data.naics_codes)}
"""
    
    def _format_pe_fund_profiles(self, pe_funds: List[Dict[str, Any]]) -> str:
        """Format PE fund candidate data for LLM analysis."""
        
        formatted_funds = []
        
        for i, fund_data in enumerate(pe_funds, 1):
            # Handle both enhanced and basic fund data structures
            if 'firm' in fund_data:
                firm = fund_data['firm']
                # Use candidate position as ID to ensure uniqueness after deduplication
                fund_id = str(i)
            else:
                firm = fund_data.get('fund', {})
                fund_id = str(i)
            
            # Extract key fund information
            # Make it clear that the fund_id is what should be used in responses
            fund_profile = f"""
**Fund ID: {fund_id}** (Candidate #{i}):
- Name: {firm.get('firm_name') or firm.get('fund_name', 'Unknown')}
- Strategy: {firm.get('pe_strategy') or firm.get('strategy', 'Unknown')}
- Stage Preference: {firm.get('stage_preference') or firm.get('focus_stage', 'Unknown')}
- Focus Sectors: {firm.get('focus_sector', 'Unknown')}
- AUM: {f"${firm.get('total_assets'):,.0f}" if firm.get('total_assets') else (f"${firm.get('aum_musd', 0):,.0f}M" if firm.get('aum_musd') else 'Unknown')}
- Check Size Range: {firm.get('check_size_range', 'Unknown')}
- Target Company Size: {firm.get('target_company_size', 'Unknown')}
- Legal Structure: {firm.get('legal_structure', 'Unknown')}
- Location: {firm.get('address_city', '')}, {firm.get('address_state', '')}
- Investment Thesis: {firm.get('thesis', 'Not available')}
"""
            
            # Add PE analysis if available
            if 'pe_analysis' in fund_data:
                pe_analysis = fund_data['pe_analysis']
                fund_profile += f"""
- PE Analysis: Strategy Match: {pe_analysis.get('strategy_match', 'Unknown')}, 
  Size Compatible: {pe_analysis.get('size_match', 'Unknown')}, 
  Stage Aligned: {pe_analysis.get('stage_match', 'Unknown')}
- Fit Score: {pe_analysis.get('fit_score', 'Unknown')}
- Reasoning: {', '.join(pe_analysis.get('reasoning', []))}
"""
            
            formatted_funds.append(fund_profile)
        
        return '\n'.join(formatted_funds)
    
    async def _call_gemini_with_structured_output(self, prompt: str) -> Dict[str, Any]:
        """Call Gemini with structured JSON output requirements."""
        
        try:
            if NEW_API_AVAILABLE:
                # Use new Gemini API with structured output
                response = await self._call_new_gemini_api(prompt)
            else:
                # Use legacy API
                response = await self._call_legacy_gemini_api(prompt)
            
            # Parse JSON response
            if isinstance(response, str):
                # Clean up response (remove markdown formatting if present)
                response = response.strip()
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                response = response.strip()
                
                # Debug: Log raw response
                logger.debug(f"Raw LLM response (first 500 chars): {response[:500]}")
                
                parsed_response = json.loads(response)
                
                # Debug: Log fund IDs from LLM
                if 'fund_rankings' in parsed_response:
                    llm_fund_ids = [r.get('fund_id') for r in parsed_response['fund_rankings']]
                    # logger.info(f"LLM returned fund IDs: {llm_fund_ids}")
                
                return parsed_response
            else:
                return response
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.error(f"Response that failed to parse: {response[:1000] if 'response' in locals() else 'No response'}")
            # Return fallback structure
            return self._generate_fallback_response()
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return self._generate_fallback_response()
    
    async def _call_new_gemini_api(self, prompt: str) -> str:
        """Call new Gemini API (if available)."""
        # Use the correct method for the new API
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.6,  # Balanced temperature for analytical yet creative output
                    max_output_tokens=8192,
                    system_instruction="You are an expert investment banking analyst. Always respond with valid JSON following the exact schema provided."
                )
            )
            return response.text
        except AttributeError:
            # Fallback to legacy API if new API structure is different
            return await self._call_legacy_gemini_api(prompt)
    
    async def _call_legacy_gemini_api(self, prompt: str) -> str:
        """Call legacy Gemini API."""
        model = genai.GenerativeModel(self.model_name)
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.6,
                max_output_tokens=8192,
            )
        )
        return response.text
    
    def _generate_fallback_response(self) -> Dict[str, Any]:
        """Generate fallback response when LLM fails."""
        return {
            "executive_summary": "LLM analysis unavailable - using fallback ranking based on semantic similarity.",
            "investment_attractiveness": "moderate",
            "fund_rankings": [],
            "market_context": {
                "sector_trends": ["Unable to generate market analysis"],
                "competitive_landscape": "Analysis unavailable",
                "market_timing": "Unable to assess",
                "valuation_environment": "Analysis unavailable"
            },
            "process_recommendations": {
                "recommended_approach": "Standard fundraising process recommended",
                "sequencing_strategy": "Simultaneous outreach to top candidates",
                "key_negotiation_points": ["Standard terms negotiation"],
                "timeline_recommendations": "6-8 week standard process"
            },
            "analysis_confidence": 30,
            "key_assumptions": ["LLM analysis failed - using basic heuristics"],
            "additional_diligence_needed": ["Complete investment analysis required"]
        }

async def demo_llm_reasoning():
    """Demonstrate LLM reasoning capabilities."""
    
    # This would be called from the CLI with actual data
    print("🧠 LLM Reasoning Engine Demo")
    print("Note: This requires GOOGLE_API_KEY environment variable")
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ GOOGLE_API_KEY not found - cannot run demo")
        return
    
    try:
        reasoner = LLMRankingReasoner()
        print("✅ LLM reasoning engine initialized")
        
        # In real usage, this would get data from the enhanced matcher
        print("📊 In production, this would:")
        print("  1. Get SMB analysis from analyze_smb_website()")
        print("  2. Get PE candidates from EnhancedFundMatcher")
        print("  3. Generate LLM analysis and rankings")
        print("  4. Combine with full PE data for display")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_llm_reasoning())