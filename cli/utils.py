"""
Utility functions for the SMB Analyzer CLI
"""

import json
import re
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse


def validate_url(url: str) -> Optional[str]:
    """
    Validate and normalize a website URL
    
    Args:
        url: Input URL string
        
    Returns:
        Normalized URL if valid, None if invalid
    """
    if not url:
        return None
    
    # Add https:// if no protocol specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        
        # Basic validation
        if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.', parsed.netloc):
            return None
            
        return url
    except:
        return None


def save_results(
    results: Optional[Dict[str, Any]], 
    report_data: Optional[Dict[str, Any]], 
    filename: str,
    output_format: str = 'json'
) -> str:
    """
    Save analysis results and/or report to file
    
    Args:
        results: Analysis results from API
        report_data: Report data from API
        filename: Output filename (can include path)
        output_format: Output format ('json', 'markdown', 'table')
        
    Returns:
        Path to saved file
        
    Raises:
        Exception: If save operation fails
    """
    
    # Determine file extension if not provided
    file_path = Path(filename)
    if not file_path.suffix:
        if output_format == 'markdown' or (report_data and not results):
            file_path = file_path.with_suffix('.md')
        else:
            file_path = file_path.with_suffix('.json')
    
    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_path.suffix.lower() == '.md':
            # Save as markdown (report only)
            content = ""
            
            if report_data and 'report_markdown' in report_data:
                content = report_data['report_markdown']
            elif results:
                # Generate basic markdown from results
                content = _generate_basic_markdown(results)
            else:
                raise ValueError("No content to save as markdown")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        else:
            # Save as JSON
            save_data = {}
            
            if results:
                save_data['analysis_results'] = results
            
            if report_data:
                save_data['report'] = report_data
            
            if not save_data:
                raise ValueError("No data to save")
            
            # Add metadata
            save_data['metadata'] = {
                'saved_at': datetime.now().isoformat(),
                'cli_version': '1.0.0'
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
        
        return str(file_path.absolute())
        
    except Exception as e:
        raise Exception(f"Failed to save file: {e}")


def _generate_basic_markdown(results: Dict[str, Any]) -> str:
    """
    Generate basic markdown report from analysis results
    
    Args:
        results: Analysis results from API
        
    Returns:
        Markdown formatted string
    """
    
    if not results.get('success', False):
        return f"# Analysis Failed\n\nError: {results.get('error', 'Unknown error')}"
    
    smb_profile = results.get('smb_profile', {})
    matches = results.get('matches', [])
    
    # Extract company info
    company_name = smb_profile.get('company_name', 'Unknown Company')
    industry = smb_profile.get('primary_industry', 'Unknown')
    size_band = smb_profile.get('size_band', 'Unknown')
    revenue = smb_profile.get('estimated_revenue')
    hq = smb_profile.get('headquarters_location', 'Unknown')
    
    markdown = f"""# SMB Analysis Report: {company_name}

## Company Overview

- **Industry**: {industry}
- **Size**: {size_band}
- **Headquarters**: {hq}
"""
    
    if revenue:
        revenue_formatted = f"${revenue:,.0f}" if isinstance(revenue, (int, float)) else str(revenue)
        markdown += f"- **Estimated Revenue**: {revenue_formatted}\n"
    
    # Key strengths
    key_strengths = smb_profile.get('key_strengths', [])
    if key_strengths:
        markdown += "\n## Key Strengths\n\n"
        for strength in key_strengths:
            markdown += f"- {strength}\n"
    
    # PE fund matches
    if matches:
        markdown += "\n## PE Fund Matches\n\n"
        
        for i, match in enumerate(matches[:10], 1):
            fund = match.get('fund', {})
            reason = match.get('match_reason', 'Semantic similarity')
            distance = match.get('distance', 0)
            
            fund_name = fund.get('fund_name', 'Unknown')
            strategy = fund.get('strategy', '')
            aum = fund.get('aum_musd', 0)
            
            # Determine fit
            if distance < 1.0:
                fit = "High"
            elif distance < 1.5:
                fit = "Medium"
            else:
                fit = "Exploratory"
            
            markdown += f"### {i}. {fund_name} - {fit} Fit\n\n"
            
            if strategy:
                markdown += f"**Strategy**: {strategy}\n\n"
            if aum:
                markdown += f"**AUM**: ${aum}M\n\n"
            
            markdown += f"**Rationale**: {reason}\n\n"
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown += f"\n---\n*Generated by SMB Analyzer CLI on {timestamp}*\n"
    
    return markdown


def format_currency(amount: Optional[float]) -> str:
    """
    Format currency amount for display
    
    Args:
        amount: Currency amount
        
    Returns:
        Formatted currency string
    """
    if amount is None:
        return "Unknown"
    
    if amount >= 1_000_000_000:
        return f"${amount/1_000_000_000:.1f}B"
    elif amount >= 1_000_000:
        return f"${amount/1_000_000:.0f}M"
    elif amount >= 1_000:
        return f"${amount/1_000:.0f}K"
    else:
        return f"${amount:.0f}"


def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Truncate text to specified length with ellipsis
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."


def extract_domain(url: str) -> str:
    """
    Extract domain name from URL
    
    Args:
        url: Input URL
        
    Returns:
        Domain name
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return url