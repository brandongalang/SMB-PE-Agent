#!/usr/bin/env python3
"""
SMB Analyzer CLI - Main entry point
"""

import warnings
# Suppress urllib3 OpenSSL warnings
warnings.filterwarnings('ignore', message='urllib3.*', module='urllib3')

import click
import sys
from pathlib import Path
from typing import Optional
import time

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text

import os
# Add parent directory to path for package imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from cli.api_client import SMBAnalyzerClient, APIError
from cli.formatters import format_analysis_results, format_report
from cli.utils import save_results, validate_url

console = Console()


@click.command()
@click.argument('website_url', type=str)
@click.option('--company-name', '-n', type=str, help='Optional company name hint')
@click.option('--top-k', '-k', type=int, default=10, help='Number of PE funds to show in final output (default: 10)')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), default='table', help='Output format')
@click.option('--quick', is_flag=True, help='Quick mode (legacy option, now always enabled)')
@click.option('--report-only', is_flag=True, help='Report-only mode (no longer supported)')
@click.option('--save', type=str, help='Save results to file')
@click.option('--api-url', default='http://localhost:8000', help='FastAPI server URL')
@click.option('--timeout', type=int, default=120, help='Request timeout in seconds')
@click.option('--llm-analysis/--no-llm-analysis', default=True, help='Use LLM reasoning for sophisticated fund analysis (default: enabled)')
@click.option('--candidate-pool-size', '-c', type=int, default=50, help='Number of PE fund candidates for LLM to analyze (default: 50)')
@click.version_option(version='1.0.0')
def main(
    website_url: str,
    company_name: Optional[str],
    top_k: int,
    output_format: str,
    quick: bool,
    report_only: bool,
    save: Optional[str],
    api_url: str,
    timeout: int,
    llm_analysis: bool,
    candidate_pool_size: int
):
    """
    SMB Analyzer CLI - Analyze SMB websites and find PE fund matches
    
    WEBSITE_URL: Company website to analyze (e.g., https://company.com)
    """
    
    # Validate inputs
    website_url = validate_url(website_url)
    if not website_url:
        console.print("❌ [red]Error: Invalid website URL format[/red]")
        console.print("💡 [yellow]Suggestion: Use format 'https://company.com'[/yellow]")
        sys.exit(1)
    
    # Initialize API client
    client = SMBAnalyzerClient(base_url=api_url, timeout=timeout)
    
    # Check for LLM analysis mode
    if llm_analysis:
        console.print()
        header_text = Text("🧠 SMB ANALYZER CLI - LLM MODE", style="bold green")
        console.print(Panel(header_text, expand=False))
        console.print(f"🔍 Analyzing website: [cyan]{website_url}[/cyan]")
        console.print("🧠 [green]Using Gemini 2.5 Flash for sophisticated investment analysis[/green]")
        
        # Run LLM analysis pipeline
        try:
            import asyncio
            from cli.llm_integration import LLMAnalysisCLI, display_llm_results
            
            llm_cli = LLMAnalysisCLI()
            results = asyncio.run(llm_cli.run_llm_analysis(
                website_url=website_url,
                company_name=company_name,
                top_k=top_k,
                candidate_pool_size=candidate_pool_size
            ))
            
            # Display LLM results
            display_llm_results(results, output_format)
            
            # Save results if requested
            if save:
                save_results(results, save, output_format)
                console.print(f"[green]Results saved to: {save}[/green]")
            
            return
            
        except Exception as e:
            console.print(f"[red]LLM analysis failed: {e}[/red]")
            console.print("[yellow]Falling back to standard analysis...[/yellow]")
            # Continue to standard analysis below
    
    # Show header
    console.print()
    header_text = Text("SMB ANALYZER CLI", style="bold blue")
    console.print(Panel(header_text, expand=False))
    console.print(f"🔍 Analyzing website: [cyan]{website_url}[/cyan]")
    
    if not report_only:
        # Phase 1: Get analysis results
        console.print("⏳ [yellow]This may take 10-30 seconds...[/yellow]")
        console.print()
        
        start_time = time.time()
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Analyzing company and matching PE funds...", total=None)
                
                # Call the API - always use quick mode (no executive report)
                results = client.get_pe_matches(website_url, company_name, top_k=top_k)
                report_data = None
                
                progress.update(task, description="Analysis complete!")
        
        except APIError as e:
            console.print(f"❌ [red]API Error: {e}[/red]")
            if "timeout" in str(e).lower():
                console.print("💡 [yellow]Suggestion: Try increasing timeout with --timeout 180[/yellow]")
            elif "connection" in str(e).lower():
                console.print("💡 [yellow]Suggestion: Check if API server is running on {api_url}[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"❌ [red]Unexpected error: {e}[/red]")
            sys.exit(1)
        
        analysis_time = time.time() - start_time
        
        # Display Phase 1 results
        console.print()
        if output_format == 'json':
            import json
            console.print(json.dumps(results, indent=2, default=str))
        else:
            format_analysis_results(console, results, analysis_time)
        
        # Skip executive report generation - just show analysis results
            
        # Save results if requested
        if save:
            try:
                saved_path = save_results(results, None, save, output_format)
                console.print(f"✅ [green]Results saved to: {saved_path}[/green]")
            except Exception as e:
                console.print(f"⚠️  [yellow]Warning: Could not save results: {e}[/yellow]")
    
    else:
        # Report-only mode is no longer supported
        console.print("❌ [red]Report-only mode is no longer supported[/red]")
        console.print("💡 [yellow]Use normal mode without --report-only flag[/yellow]")
        sys.exit(1)


if __name__ == '__main__':
    main()