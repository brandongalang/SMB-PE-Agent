"""
Rich formatters for displaying SMB analysis results and reports
"""

from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.columns import Columns
from rich.box import ROUNDED


def format_analysis_results(console: Console, results: Dict[str, Any], analysis_time: float):
    """
    Format and display Phase 1 analysis results
    
    Args:
        console: Rich console instance
        results: API response from /match/pe endpoint
        analysis_time: Time taken for analysis in seconds
    """
    
    if not results.get('success', False):
        console.print(f"❌ [red]Analysis failed: {results.get('error', 'Unknown error')}[/red]")
        return
    
    smb_profile = results.get('smb_profile', {})
    matches = results.get('matches', [])
    
    # Company Analysis Section
    console.print()
    console.print("━" * 79)
    console.print("[bold blue]                              📊 COMPANY ANALYSIS[/bold blue]")
    console.print("━" * 79)
    console.print()
    
    # Company details
    company_name = smb_profile.get('company_name', 'Unknown')
    industry = smb_profile.get('primary_industry', 'Unknown')
    size_band = smb_profile.get('size_band', 'Unknown')
    revenue = smb_profile.get('estimated_revenue')
    hq = smb_profile.get('headquarters_location', 'Unknown')
    stage = smb_profile.get('company_stage', 'Unknown')
    
    console.print(f"🏢 Company: [bold cyan]{company_name}[/bold cyan]")
    console.print(f"🏭 Industry: {industry}")
    console.print(f"📏 Size: {size_band}")
    
    if revenue:
        revenue_formatted = f"~${revenue:,.0f}" if isinstance(revenue, (int, float)) else str(revenue)
        console.print(f"💰 Revenue: {revenue_formatted}")
    
    console.print(f"🏠 HQ: {hq}")
    console.print(f"📈 Stage: {stage}")
    
    # Key Strengths
    key_strengths = smb_profile.get('key_strengths', [])
    if key_strengths:
        console.print()
        console.print("💡 [bold]Key Strengths:[/bold]")
        for strength in key_strengths[:4]:  # Show top 4
            console.print(f"  • {strength}")
    
    # PE Fund Matches Section
    console.print()
    console.print("━" * 79)
    console.print("[bold blue]                           🎯 TOP PE FUND MATCHES[/bold blue]")
    console.print("━" * 79)
    console.print()
    
    if not matches:
        console.print("[yellow]No PE fund matches found[/yellow]")
        return
    
    # Create matches table
    table = Table(box=ROUNDED, show_header=True, header_style="bold blue")
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Fund Name", style="cyan", width=24)
    table.add_column("Fit", width=10)
    table.add_column("Focus & Rationale", width=35)
    
    for i, match in enumerate(matches[:10], 1):  # Show top 10
        fund = match.get('fund', {})
        distance = match.get('distance', 0)
        reason = match.get('match_reason', 'Semantic similarity')
        
        fund_name = fund.get('fund_name', 'Unknown')
        strategy = fund.get('strategy', '')
        sector = fund.get('focus_sector', '')
        aum = fund.get('aum_musd', 0)
        
        # Determine fit level based on distance
        if distance < 1.0:
            fit = "[green]High[/green]"
        elif distance < 1.5:
            fit = "[yellow]Medium[/yellow]"
        else:
            fit = "[red]Exploratory[/red]"
        
        # Format focus area
        focus_parts = []
        if strategy:
            focus_parts.append(strategy)
        if sector:
            focus_parts.append(sector)
        if aum:
            focus_parts.append(f"${aum}M AUM")
        
        focus_line = " | ".join(focus_parts)
        
        # Combine focus and rationale
        focus_and_reason = f"{focus_line}\n{reason}" if focus_line else reason
        
        table.add_row(
            str(i),
            fund_name,
            fit,
            focus_and_reason
        )
    
    console.print(table)
    
    # Analysis metadata
    console.print()
    
    # Format cost and confidence
    cost = smb_profile.get('estimated_cost_usd', 0)
    confidence_scores = smb_profile.get('confidence_scores', {})
    overall_confidence = 'MEDIUM'  # Default
    
    if confidence_scores:
        # Calculate overall confidence (simple heuristic)
        confidence_values = list(confidence_scores.values())
        if confidence_values:
            high_count = sum(1 for c in confidence_values if str(c).upper() == 'HIGH')
            if high_count >= len(confidence_values) // 2:
                overall_confidence = 'HIGH'
            elif any(str(c).upper() == 'LOW' for c in confidence_values):
                overall_confidence = 'LOW'
    
    confidence_color = {
        'HIGH': 'green',
        'MEDIUM': 'yellow', 
        'LOW': 'red'
    }.get(overall_confidence, 'yellow')
    
    metadata_text = f"💵 Analysis Cost: ${cost:.4f} | ⏱️  Time: {analysis_time:.1f}s | 🎯 Confidence: [{confidence_color}]{overall_confidence}[/{confidence_color}]"
    console.print(metadata_text)


def format_report(console: Console, report_data: Dict[str, Any]):
    """
    Format and display Phase 2 executive report
    
    Args:
        console: Rich console instance
        report_data: API response from /match/pe/report endpoint
    """
    
    if not report_data.get('success', False):
        console.print(f"❌ [red]Report generation failed: {report_data.get('error', 'Unknown error')}[/red]")
        return
    
    report_markdown = report_data.get('report_markdown', '')
    
    if not report_markdown:
        console.print("[yellow]No report content available[/yellow]")
        return
    
    # Report header
    console.print("━" * 79)
    console.print("[bold blue]                            📋 EXECUTIVE REPORT[/bold blue]")
    console.print("━" * 79)
    console.print()
    
    # Render markdown report
    try:
        md = Markdown(report_markdown)
        console.print(md)
    except Exception as e:
        # Fallback to plain text if markdown rendering fails
        console.print(report_markdown)
    
    console.print()
    console.print("[dim italic]Generated by SMB Analyzer CLI - Powered by Gemini 2.5 Flash[/dim italic]")
    console.print()
    
    # Interactive save prompt
    try:
        save_prompt = console.input("💾 Save report? [[y/N]]: ")
        if save_prompt.lower() in ['y', 'yes']:
            return True
    except (EOFError, KeyboardInterrupt):
        console.print()
    
    return False


def format_error_message(console: Console, error: str, suggestions: Optional[str] = None):
    """
    Format and display error messages with suggestions
    
    Args:
        console: Rich console instance
        error: Error message
        suggestions: Optional suggestions for fixing the error
    """
    console.print(f"❌ [red]Error: {error}[/red]")
    
    if suggestions:
        console.print(f"💡 [yellow]Suggestion: {suggestions}[/yellow]")


def format_success_message(console: Console, message: str):
    """
    Format and display success messages
    
    Args:
        console: Rich console instance
        message: Success message
    """
    console.print(f"✅ [green]{message}[/green]")