"""
SQLAlchemy models for Form ADV data from SEC filings.

This module defines the database schema for storing and querying Form ADV data
from the SEC, combining information from both Part A and Part B filings.
"""

from datetime import datetime
from typing import Optional, Dict, List

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean, Text, 
    create_engine, MetaData, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class AdvFirm(Base):
    """
    Combined Form ADV data model representing investment advisory firms.
    
    This model combines key fields from both ADV Part A and Part B filings,
    focusing on the most relevant information for PE firm identification and matching.
    """
    
    __tablename__ = 'adv_firms'
    
    # Primary identifier
    id = Column(Integer, primary_key=True, autoincrement=True)
    filing_id = Column(String(50), unique=True, nullable=False, index=True)
    
    # Core firm identification
    firm_name = Column(String(500), nullable=False, index=True)  # 1A field
    business_name = Column(String(500), nullable=True)  # 1B field  
    legal_name = Column(String(500), nullable=True)  # 1C-Legal
    new_name = Column(String(500), nullable=True)  # 1C-New Name
    
    # SEC identifiers
    crd_number = Column(String(50), nullable=True, index=True)  # 1D
    cik_number = Column(String(50), nullable=True, index=True)  # 1N-CIK
    
    # Date information
    form_version = Column(String(20), nullable=True)  # FormVersion
    date_submitted = Column(DateTime, nullable=True)  # DateSubmitted
    
    # Primary address (1F1 fields)
    address_street1 = Column(String(200), nullable=True)
    address_street2 = Column(String(200), nullable=True)
    address_city = Column(String(100), nullable=True)
    address_state = Column(String(50), nullable=True)
    address_country = Column(String(50), nullable=True)
    address_postal = Column(String(20), nullable=True)
    
    # Contact information (1J fields)
    contact_name = Column(String(200), nullable=True)
    contact_title = Column(String(200), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    contact_fax = Column(String(50), nullable=True)
    contact_email = Column(String(200), nullable=True)
    
    # Business hours and availability
    business_hours = Column(String(100), nullable=True)  # 1F2-Hours
    phone_main = Column(String(50), nullable=True)  # 1F3
    fax_main = Column(String(50), nullable=True)  # 1F4
    
    # Employee information (5A, 5B fields)
    employee_range = Column(String(50), nullable=True)  # 5A-Range
    employee_number = Column(Integer, nullable=True)  # 5A-Number
    
    # Assets under management (5F fields)
    total_assets = Column(Float, nullable=True)  # 5F2a total
    discretionary_assets = Column(Float, nullable=True)  # 5F2b discretionary
    non_discretionary_assets = Column(Float, nullable=True)  # 5F2c non-discretionary
    
    # Client information (5C fields)
    client_range = Column(String(50), nullable=True)  # 5C-Range
    client_number = Column(Integer, nullable=True)  # 5C-Number
    
    # Business structure (from Part B)
    legal_structure = Column(String(100), nullable=True)  # 3A
    legal_structure_other = Column(String(200), nullable=True)  # 3A-Other
    fiscal_year_end = Column(String(20), nullable=True)  # 3B
    incorporation_state = Column(String(50), nullable=True)  # 3C-State
    incorporation_country = Column(String(50), nullable=True)  # 3C-Country
    
    # Advisory services (5D fields - key service types)
    provides_investment_advice = Column(Boolean, nullable=True)  # 5D1/5D1a
    provides_financial_planning = Column(Boolean, nullable=True)  # 5D2/5D1b
    provides_pension_consulting = Column(Boolean, nullable=True)  # 5D3/5D1c
    provides_selection_services = Column(Boolean, nullable=True)  # 5D4/5D1d
    
    # Registration information (2A fields - key states)
    registered_states = Column(Text, nullable=True)  # JSON string of registered states
    
    # Investment strategies and client types (5E fields)
    strategy_equity = Column(Boolean, nullable=True)  # 5E1
    strategy_fixed_income = Column(Boolean, nullable=True)  # 5E2
    strategy_commodity = Column(Boolean, nullable=True)  # 5E3
    strategy_mutual_funds = Column(Boolean, nullable=True)  # 5E4
    strategy_hedge_funds = Column(Boolean, nullable=True)  # 5E5
    strategy_private_funds = Column(Boolean, nullable=True)  # 5E6
    strategy_other = Column(Boolean, nullable=True)  # 5E7
    
    # Fee structure (5A-5B compensation ranges)
    fee_structure_notes = Column(Text, nullable=True)  # Combined fee information
    
    # Regulatory and compliance
    sec_registered = Column(Boolean, nullable=True)  # 1M
    state_registered = Column(Boolean, nullable=True)  # 1N
    
    # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Composite text field for semantic search
    searchable_text = Column(Text, nullable=True)  # Will be populated with combined text
    
    # Enhanced investment focus fields (added via parallel AI classification)
    focus_sector = Column(String(50), nullable=True)  # Industry sector focus
    stage_preference = Column(String(50), nullable=True)  # Investment stage preference
    check_size_range = Column(String(50), nullable=True)  # Typical investment amounts
    target_company_size = Column(String(50), nullable=True)  # Target company size
    
    # PE-specific classification fields
    is_pe_fund = Column(Boolean, nullable=True)  # Boolean PE fund identification
    pe_strategy = Column(String(50), nullable=True)  # PE strategy classification
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_firm_name_search', 'firm_name'),
        Index('idx_location_search', 'address_state', 'address_city'),
        Index('idx_assets_size', 'total_assets'),
        Index('idx_legal_structure', 'legal_structure'),
        Index('idx_registration', 'sec_registered', 'state_registered'),
    )
    
    def __repr__(self):
        return f"<AdvFirm(filing_id='{self.filing_id}', firm_name='{self.firm_name}')>"
    
    def to_dict(self):
        """Convert model instance to dictionary for API responses."""
        min_check, max_check = self.parse_check_size_range()
        
        return {
            'id': self.id,
            'filing_id': self.filing_id,
            'firm_name': self.firm_name,
            'business_name': self.business_name,
            'legal_name': self.legal_name,
            'crd_number': self.crd_number,
            'cik_number': self.cik_number,
            'address_city': self.address_city,
            'address_state': self.address_state,
            'address_country': self.address_country,
            'contact_name': self.contact_name,
            'contact_title': self.contact_title,
            'contact_phone': self.contact_phone,
            'contact_email': self.contact_email,
            'employee_range': self.employee_range,
            'employee_number': self.employee_number,
            'total_assets': self.total_assets,
            'client_range': self.client_range,
            'client_number': self.client_number,
            'legal_structure': self.legal_structure,
            'incorporation_state': self.incorporation_state,
            'incorporation_country': self.incorporation_country,
            'sec_registered': self.sec_registered,
            'state_registered': self.state_registered,
            
            # Enhanced investment focus fields
            'focus_sector': self.focus_sector,
            'stage_preference': self.stage_preference,
            'check_size_range': self.check_size_range,
            'target_company_size': self.target_company_size,
            
            # PE-specific classification fields
            'is_pe_fund': self.is_pe_fund,
            'pe_strategy': self.pe_strategy,
            
            # Computed fields from helper methods
            'pe_strategy_normalized': self.get_pe_strategy_mapping(),
            'compatible_stages': self.get_stage_alignment(),
            'min_check_size_millions': min_check,
            'max_check_size_millions': max_check,
            
            # Investment strategies (boolean flags)
            'strategy_equity': self.strategy_equity,
            'strategy_fixed_income': self.strategy_fixed_income,
            'strategy_private_funds': self.strategy_private_funds,
            'strategy_hedge_funds': self.strategy_hedge_funds,
            
            # Services provided
            'provides_investment_advice': self.provides_investment_advice,
            'provides_financial_planning': self.provides_financial_planning,
            'provides_pension_consulting': self.provides_pension_consulting,
        }
    
    def generate_searchable_text(self):
        """Generate composite text field for semantic search embedding."""
        parts = []
        
        # Core identifiers
        if self.firm_name:
            parts.append(f"Firm: {self.firm_name}")
        if self.business_name and self.business_name != self.firm_name:
            parts.append(f"Business: {self.business_name}")
        if self.legal_name and self.legal_name != self.firm_name:
            parts.append(f"Legal: {self.legal_name}")
            
        # Location
        location_parts = []
        if self.address_city:
            location_parts.append(self.address_city)
        if self.address_state:
            location_parts.append(self.address_state)
        if self.address_country:
            location_parts.append(self.address_country)
        if location_parts:
            parts.append(f"Location: {', '.join(location_parts)}")
            
        # Business structure
        if self.legal_structure:
            parts.append(f"Structure: {self.legal_structure}")
            
        # Size indicators
        if self.employee_range:
            parts.append(f"Employees: {self.employee_range}")
        if self.client_range:
            parts.append(f"Clients: {self.client_range}")
            
        # Services
        services = []
        if self.provides_investment_advice:
            services.append("Investment Advice")
        if self.provides_financial_planning:
            services.append("Financial Planning")
        if self.provides_pension_consulting:
            services.append("Pension Consulting")
        if services:
            parts.append(f"Services: {', '.join(services)}")
            
        # Investment strategies
        strategies = []
        if self.strategy_equity:
            strategies.append("Equity")
        if self.strategy_fixed_income:
            strategies.append("Fixed Income")
        if self.strategy_hedge_funds:
            strategies.append("Hedge Funds")
        if self.strategy_private_funds:
            strategies.append("Private Funds")
        if strategies:
            parts.append(f"Strategies: {', '.join(strategies)}")
            
        # Identifiers
        if self.crd_number:
            parts.append(f"CRD: {self.crd_number}")
        if self.cik_number:
            parts.append(f"CIK: {self.cik_number}")
            
        self.searchable_text = " | ".join(parts)
        return self.searchable_text
    
    def get_pe_strategy_mapping(self) -> Dict[str, str]:
        """Map PE strategy to standardized categories."""
        strategy_mappings = {
            'buyout': 'Buyout',
            'growth': 'Growth Equity', 
            'growth_equity': 'Growth Equity',
            'venture': 'Venture Capital',
            'venture_capital': 'Venture Capital',
            'distressed': 'Distressed',
            'special_situations': 'Special Situations',
            'mezzanine': 'Mezzanine',
            'real_estate': 'Real Estate',
            'infrastructure': 'Infrastructure',
            'fund_of_funds': 'Fund of Funds',
            'secondary': 'Secondary'
        }
        
        if self.pe_strategy:
            normalized = self.pe_strategy.lower().replace(' ', '_')
            return strategy_mappings.get(normalized, self.pe_strategy)
        return None
    
    def get_stage_alignment(self) -> str:
        """Get compatible investment stages based on PE strategy as comma-separated string."""
        stage_alignments = {
            'Buyout': ['mature', 'established', 'late_stage'],
            'Growth Equity': ['growth', 'expansion', 'late_stage'],
            'Venture Capital': ['seed', 'early_stage', 'growth'],
            'Distressed': ['turnaround', 'restructuring', 'mature'],
            'Mezzanine': ['growth', 'expansion', 'buyout'],
            'Special Situations': ['turnaround', 'carve_out', 'restructuring']
        }
        
        strategy = self.get_pe_strategy_mapping()
        if strategy in stage_alignments:
            return ",".join(stage_alignments[strategy])
        return ""
    
    def parse_check_size_range(self) -> tuple[Optional[float], Optional[float]]:
        """Parse check size range into min/max values in millions USD."""
        if not self.check_size_range:
            return None, None
            
        try:
            # Handle common formats like "$5M-$50M", "5-50M", "$5-50 million"
            text = self.check_size_range.lower().replace('$', '').replace('million', 'm').replace(' ', '')
            
            if '-' in text:
                parts = text.split('-')
                if len(parts) == 2:
                    min_val = self._parse_monetary_value(parts[0])
                    max_val = self._parse_monetary_value(parts[1])
                    return min_val, max_val
            else:
                # Single value, treat as minimum
                val = self._parse_monetary_value(text)
                return val, None
                
        except (ValueError, AttributeError):
            pass
            
        return None, None
    
    def _parse_monetary_value(self, value_str: str) -> Optional[float]:
        """Parse monetary value string to float in millions."""
        if not value_str:
            return None
            
        value_str = value_str.strip().replace(',', '')
        
        try:
            if value_str.endswith('m'):
                return float(value_str[:-1])
            elif value_str.endswith('k'):
                return float(value_str[:-1]) / 1000  # Convert K to M
            elif value_str.endswith('b'):
                return float(value_str[:-1]) * 1000  # Convert B to M
            else:
                # Assume raw number is in millions
                return float(value_str)
        except ValueError:
            return None
    
    def is_size_compatible(self, target_size_millions: float) -> bool:
        """Check if fund's check size range is compatible with target investment size."""
        min_check, max_check = self.parse_check_size_range()
        
        if min_check is None and max_check is None:
            return True  # No size constraints
            
        if min_check is not None and target_size_millions < min_check:
            return False
            
        if max_check is not None and target_size_millions > max_check:
            return False
            
        return True


class DatabaseManager:
    """Database connection and session management for ADV data."""
    
    def __init__(self, database_url: str = "sqlite:///adv_database.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        
    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()
        
    def drop_tables(self):
        """Drop all tables (use with caution)."""
        Base.metadata.drop_all(bind=self.engine)