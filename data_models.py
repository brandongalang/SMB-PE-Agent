"""
Shared data models for the SMB Analyzer system
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class ConfidenceLevel(Enum):
    HIGH = "HIGH"       # >80% confidence, multiple supporting signals
    MEDIUM = "MEDIUM"   # 60-80% confidence, some supporting data
    LOW = "LOW"         # <60% confidence, limited/inferred data

class CompanySize(Enum):
    MICRO = "MICRO"           # 1-10 employees
    SMALL = "SMALL"           # 11-50 employees
    MEDIUM = "MEDIUM"         # 51-250 employees
    LARGE = "LARGE"           # 251-1000 employees
    ENTERPRISE = "ENTERPRISE" # 1000+ employees

# DEPRECATED: ProcessedCompanyData has been deprecated in favor of using SMBCompanyData directly
# This change eliminates data loss during the conversion process and ensures all rich
# company analysis data flows through the entire PE matching and reasoning pipeline.
#
# Migration guide:
# - Replace ProcessedCompanyData with SMBCompanyData from api.agents.smb_agent
# - Update field mappings:
#   - company_stage -> growth_stage
#   - key_strengths -> competitive_advantages
#   - company_summary -> business_description
#
# @dataclass
# class ProcessedCompanyData:
#     """LLM-processed and structured company data for PE matching"""
#     
#     # Core Identification
#     company_name: str                     # Verified company name
#     business_description: str             # 2-3 sentence LLM summary
#     
#     # Industry Classification
#     naics_codes: List[str]               # 6-digit NAICS codes
#     industry_keywords: List[str]          # Refined industry terms
#     primary_industry: str                 # Main industry category
#     
#     # Size Estimation
#     estimated_employees: Optional[int] = None   # Employee count estimate
#     estimated_revenue: Optional[float] = None   # Revenue in USD
#     size_band: Optional[CompanySize] = None
#     
#     # Business Intelligence
#     business_model: str = "Unknown"                   # B2B, B2C, B2B2C, Marketplace
#     company_stage: str = "Unknown"                    # startup, growth, mature, established
#     
#     # Geographic Data
#     headquarters_location: str = "Unknown"            # Structured HQ location
#     
#     # Qualitative Analysis
#     company_summary: str = ""                  # 3-sentence executive summary
#     key_strengths: List[str] = field(default_factory=list)
#     
#     # Confidence Scores
#     confidence_scores: Dict[str, ConfidenceLevel] = field(default_factory=dict)