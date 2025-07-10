#!/usr/bin/env python3
"""
CLI integration for LLM reasoning layer.

This module provides CLI functions to use the LLM reasoning engine
for sophisticated PE fund analysis and ranking.
"""

import asyncio
import sys
import os
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_reasoning_engine import LLMRankingReasoner
from llm_reasoning_schema import LLMReasoningOutput, PEFundRanking
from api.agents.smb_agent import analyze_smb_website, SMBCompanyData
from enhanced_fund_matcher import EnhancedFundMatcher
from adv_semantic_layer import AdvSemanticSearcher

console = Console()

class LLMAnalysisCLI:
    """CLI interface for LLM-powered PE fund analysis."""
    
    def __init__(self):
        self.reasoner = None
        self.enhanced_matcher = None
        self.adv_searcher = None
    
    async def run_llm_analysis(
        self,
        website_url: str,
        company_name: Optional[str] = None,
        top_k: int = 10,
        candidate_pool_size: int = 50
    ) -> Dict[str, Any]:
        """
        Run complete LLM-powered analysis pipeline.
        
        Args:
            website_url: Company website to analyze
            company_name: Optional company name hint
            top_k: Number of final recommendations
            candidate_pool_size: Number of PE candidates to send to LLM
            
        Returns:
            Combined results with LLM rankings and full PE data
        """
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Step 1: SMB Analysis
            task1 = progress.add_task("🔍 Analyzing company website...", total=None)
            try:
                smb_data = await analyze_smb_website(website_url, company_name)
                progress.remove_task(task1)
                
                if not smb_data or smb_data.company_name in ["Analysis Failed", "Unknown", "", "Unable to determine due to tool failure."]:
                    console.print(f"❌ [red]SMB analysis failed - Company name: {smb_data.company_name if smb_data else 'None'}[/red]")
                    return {"success": False, "error": "SMB analysis failed"}
            except Exception as e:
                progress.remove_task(task1)
                console.print(f"❌ [red]SMB analysis exception: {e}[/red]")
                return {"success": False, "error": f"SMB analysis exception: {e}"}
            
            console.print(f"✅ [green]Company analyzed: {smb_data.company_name}[/green]")
            
            # Step 2: Get PE Fund Candidates
            task2 = progress.add_task("🔎 Finding PE fund candidates...", total=None)
            pe_candidates = await self._get_pe_fund_candidates(smb_data, candidate_pool_size)
            progress.remove_task(task2)
            
            if not pe_candidates:
                console.print("❌ [red]No PE fund candidates found[/red]")
                return {"success": False, "error": "No PE candidates found"}
            
            console.print(f"✅ [green]Found {len(pe_candidates)} unique PE fund candidates (after deduplication)[/green]")
            
            # Step 3: LLM Analysis
            task3 = progress.add_task("🧠 Running Gemini 2.5 Flash Lite w/Thinking analysis...", total=None)
            
            try:
                # Initialize LLM reasoner
                if not self.reasoner:
                    self.reasoner = LLMRankingReasoner()
                
                # Generate LLM analysis
                llm_output = await self.reasoner.analyze_and_rank_funds(
                    smb_data, pe_candidates, top_k
                )
                progress.remove_task(task3)
                
                console.print(f"✅ [green]LLM analysis completed with {llm_output.analysis_confidence}% confidence[/green]")
                
            except Exception as e:
                progress.remove_task(task3)
                console.print(f"❌ [red]LLM analysis failed: {e}[/red]")
                # Fall back to basic ranking
                return await self._fallback_to_basic_ranking(smb_data, pe_candidates, top_k)
            
            # Step 4: Combine LLM rankings with full PE data
            task4 = progress.add_task("📊 Combining results with PE fund data...", total=None)
            combined_results = self._combine_llm_with_pe_data(llm_output, pe_candidates)
            progress.remove_task(task4)
            
            console.print("✅ [green]Analysis complete![/green]")
            
            return {
                "success": True,
                "company_data": smb_data,
                "llm_analysis": llm_output,
                "enhanced_results": combined_results,
                "analysis_type": "llm_reasoning"
            }
    
    async def _get_pe_fund_candidates(
        self, 
        smb_data: SMBCompanyData, 
        candidate_pool_size: int
    ) -> List[Dict[str, Any]]:
        """Get PE fund candidates using enhanced matching."""
        
        try:
            # Initialize enhanced matcher if needed
            if not self.enhanced_matcher:
                self.enhanced_matcher = EnhancedFundMatcher(use_synthetic=True, use_adv=True)
            
            # Get enhanced matches directly with SMB data (larger pool for LLM to analyze)
            # Request more candidates to account for deduplication
            results = self.enhanced_matcher.match_company(
                company=smb_data,
                k_synthetic=candidate_pool_size // 2,
                k_adv=candidate_pool_size * 2,  # Get extra ADV results for deduplication
                total_k=candidate_pool_size * 3  # Over-fetch to ensure enough unique funds
            )
            
            # Deduplicate combined rankings by fund name
            combined_ranking = results.get("combined_ranking", [])
            deduplicated_funds = self._deduplicate_funds(combined_ranking)
            
            # Return exactly the requested number of unique funds
            return deduplicated_funds[:candidate_pool_size]
            
        except Exception as e:
            console.print(f"⚠️ [yellow]Enhanced matching failed: {e}. Using basic search...[/yellow]")
            
            # Fallback to basic ADV search
            if not self.adv_searcher:
                self.adv_searcher = AdvSemanticSearcher()
            
            query = f"{smb_data.primary_industry} {smb_data.business_model} {smb_data.growth_stage}"
            # Over-fetch for deduplication
            basic_results = self.adv_searcher.semantic_search(
                query=query,
                n_results=candidate_pool_size * 3,
                filters={"is_pe_fund": True},
                include_metadata=True
            )
            
            # Deduplicate basic results
            deduplicated_funds = self._deduplicate_funds(basic_results)
            return deduplicated_funds[:candidate_pool_size]
    
    def _deduplicate_funds(self, fund_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate funds by firm name, keeping the best scoring instance.
        
        Args:
            fund_list: List of fund candidates with potential duplicates
            
        Returns:
            Deduplicated list of funds, keeping highest scoring version of each
        """
        seen_names = {}
        deduplicated = []
        
        for fund_data in fund_list:
            # Extract fund name from different data structures
            if 'firm' in fund_data:
                firm = fund_data['firm']
                fund_name = firm.get('firm_name') or firm.get('fund_name', 'Unknown')
            else:
                firm = fund_data.get('fund', {})
                fund_name = firm.get('fund_name') or firm.get('firm_name', 'Unknown')
            
            # Normalize name for comparison (uppercase, strip whitespace)
            normalized_name = fund_name.upper().strip()
            
            # Check if we've seen this fund name before
            if normalized_name in seen_names:
                # Compare scores and keep the better one
                existing_score = seen_names[normalized_name].get('weighted_score', 0) or seen_names[normalized_name].get('confidence_score', 0) or (1.0 - seen_names[normalized_name].get('score', 1.0))
                current_score = fund_data.get('weighted_score', 0) or fund_data.get('confidence_score', 0) or (1.0 - fund_data.get('score', 1.0))
                
                if current_score > existing_score:
                    # Replace with better scoring version
                    idx = next(i for i, f in enumerate(deduplicated) if self._get_fund_name(f).upper().strip() == normalized_name)
                    deduplicated[idx] = fund_data
                    seen_names[normalized_name] = fund_data
                    # console.print(f"[dim]Replaced duplicate {fund_name} with higher score ({current_score:.3f} > {existing_score:.3f})[/dim]")
                else:
                    # console.print(f"[dim]Skipped duplicate {fund_name} with lower score ({current_score:.3f} <= {existing_score:.3f})[/dim]")
                    pass
            else:
                # First time seeing this fund
                seen_names[normalized_name] = fund_data
                deduplicated.append(fund_data)
        
        # console.print(f"[green]Deduplicated {len(fund_list)} funds to {len(deduplicated)} unique funds[/green]")
        return deduplicated
    
    def _get_fund_name(self, fund_data: Dict[str, Any]) -> str:
        """Extract fund name from fund data structure."""
        if 'firm' in fund_data:
            firm = fund_data['firm']
            return firm.get('firm_name') or firm.get('fund_name', 'Unknown')
        else:
            firm = fund_data.get('fund', {})
            return firm.get('fund_name') or firm.get('firm_name', 'Unknown')
    
    def _combine_llm_with_pe_data(
        self, 
        llm_output: LLMReasoningOutput, 
        pe_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine LLM rankings with full PE fund data.
        
        The LLM returns fund IDs with reasoning, we combine with original PE data.
        """
        
        # Create lookup for PE fund data by position ID (after deduplication)
        pe_data_lookup = {}
        for i, candidate in enumerate(pe_candidates, 1):
            # Use position-based ID to match what LLM sees
            fund_id = str(i)
            
            if 'firm' in candidate:
                firm = candidate['firm']
                fund_name = firm.get('firm_name', 'Unknown')
            else:
                firm = candidate.get('fund', {})
                fund_name = firm.get('fund_name', 'Unknown')
            
            pe_data_lookup[fund_id] = candidate
            # Debug log the mapping
            # console.print(f"[dim]Mapped position ID {fund_id} -> {fund_name}[/dim]")
        
        # Debug: Show available fund IDs
        # console.print(f"[dim]Total available fund IDs: {len(pe_data_lookup)}[/dim]")
        # console.print(f"[dim]First 10 IDs: {list(pe_data_lookup.keys())[:10]}[/dim]")
        
        # Combine LLM rankings with PE data
        combined_results = []
        
        # Debug: Show LLM requested fund IDs
        llm_fund_ids = [str(r.fund_id) for r in llm_output.fund_rankings]
        # console.print(f"[dim]LLM requested fund IDs: {llm_fund_ids}[/dim]")
        
        for llm_ranking in llm_output.fund_rankings:
            fund_id = str(llm_ranking.fund_id)
            
            # Find matching PE data
            pe_data = pe_data_lookup.get(fund_id)
            
            if pe_data:
                combined_result = {
                    "rank": llm_ranking.rank,
                    "llm_analysis": llm_ranking.dict(),
                    "pe_fund_data": pe_data,
                    "combined_score": llm_ranking.overall_score
                }
                combined_results.append(combined_result)
            else:
                console.print(f"⚠️ [yellow]Warning: LLM referenced fund ID {fund_id} not found in candidates[/yellow]")
        
        return combined_results
    
    async def _fallback_to_basic_ranking(
        self, 
        smb_data: SMBCompanyData, 
        pe_candidates: List[Dict[str, Any]], 
        top_k: int
    ) -> Dict[str, Any]:
        """Fallback to basic ranking when LLM fails."""
        
        # Sort by existing scores or semantic similarity
        sorted_candidates = sorted(
            pe_candidates[:top_k], 
            key=lambda x: x.get('weighted_score', 0) or (1.0 - x.get('score', 1.0)), 
            reverse=True
        )
        
        return {
            "success": True,
            "company_data": smb_data,
            "llm_analysis": None,
            "enhanced_results": [
                {
                    "rank": i + 1,
                    "llm_analysis": None,
                    "pe_fund_data": candidate,
                    "combined_score": candidate.get('weighted_score', 0) or (1.0 - candidate.get('score', 1.0))
                }
                for i, candidate in enumerate(sorted_candidates)
            ],
            "analysis_type": "fallback_basic",
            "warning": "LLM analysis failed - using basic ranking"
        }

def display_llm_results(results: Dict[str, Any], output_format: str = "table") -> None:
    """Display LLM analysis results in rich format."""
    
    if not results.get("success"):
        console.print(f"❌ [red]Analysis failed: {results.get('error', 'Unknown error')}[/red]")
        return
    
    company_data = results["company_data"]
    llm_analysis = results.get("llm_analysis")
    enhanced_results = results["enhanced_results"]
    
    # Company header
    console.print(Panel(
        f"[bold blue]{company_data.company_name}[/bold blue]\n"
        f"[dim]{company_data.business_description}[/dim]\n\n"
        f"Industry: {company_data.primary_industry} | "
        f"Stage: {company_data.growth_stage} | "
        f"Size: {company_data.size_band.value if company_data.size_band else 'Unknown'}",
        title="🏢 Company Analysis",
        border_style="blue"
    ))
    
    # LLM Analysis Summary
    if llm_analysis:
        console.print(Panel(
            f"[bold green]{llm_analysis.executive_summary}[/bold green]\n\n"
            f"Investment Attractiveness: [bold]{llm_analysis.investment_attractiveness.upper()}[/bold] | "
            f"Analysis Confidence: [bold]{llm_analysis.analysis_confidence}%[/bold]",
            title="🧠 LLM Investment Analysis",
            border_style="green"
        ))
    
    # Fund Rankings
    if output_format == "table":
        _display_llm_table(enhanced_results, llm_analysis)
    else:
        _display_llm_json(results)

def _display_llm_table(enhanced_results: List[Dict[str, Any]], llm_analysis: Optional[LLMReasoningOutput]) -> None:
    """Display LLM results in table format."""
    
    table = Table(title="🎯 LLM-Ranked PE Fund Recommendations", show_lines=True, padding=(1, 1))
    
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Fund Name", style="bold")
    table.add_column("Strategy", style="blue")
    table.add_column("Score", style="green", width=8)
    table.add_column("Investment Thesis", style="dim", width=40)
    table.add_column("Key Advantages", style="yellow", width=30)
    
    for result in enhanced_results:
        llm_data = result.get("llm_analysis")
        pe_data = result["pe_fund_data"]
        
        # Extract fund info
        if 'firm' in pe_data:
            firm = pe_data['firm']
            fund_name = firm.get('firm_name', 'Unknown')
            strategy = firm.get('pe_strategy', 'Unknown')
        else:
            firm = pe_data.get('fund', {})
            fund_name = firm.get('fund_name', 'Unknown')
            strategy = firm.get('strategy', 'Unknown')
        
        if llm_data:
            thesis = llm_data['investment_thesis']['headline']
            # Format advantages with bullet points for better readability
            advantages = "\n".join([f"• {adv}" for adv in llm_data['investment_thesis']['competitive_advantages'][:2]])
            score = f"{llm_data['overall_score']}/100"
        else:
            thesis = "Basic semantic match"
            advantages = "• Standard PE capabilities"
            score = f"{result['combined_score']:.1f}"
        
        table.add_row(
            str(result["rank"]),
            fund_name,
            strategy,
            score,
            thesis,
            advantages
        )
    
    console.print(table)
    
    # Process Recommendations
    if llm_analysis and llm_analysis.process_recommendations:
        proc_rec = llm_analysis.process_recommendations
        console.print(Panel(
            f"[bold]Recommended Approach:[/bold] {proc_rec.recommended_approach}\n\n"
            f"[bold]Sequencing Strategy:[/bold] {proc_rec.sequencing_strategy}\n\n"
            f"[bold]Timeline:[/bold] {proc_rec.timeline_recommendations}",
            title="📋 Strategic Recommendations",
            border_style="magenta"
        ))

def _display_llm_json(results: Dict[str, Any]) -> None:
    """Display results in JSON format."""
    import json
    
    # Convert complex objects to JSON-serializable format
    json_results = {
        "success": results["success"],
        "analysis_type": results["analysis_type"],
        "company": {
            "name": results["company_data"].company_name,
            "industry": results["company_data"].primary_industry,
            "description": results["company_data"].business_description
        },
        "llm_analysis": results["llm_analysis"].dict() if results.get("llm_analysis") else None,
        "fund_rankings": [
            {
                "rank": r["rank"],
                "combined_score": r["combined_score"],
                "llm_reasoning": r.get("llm_analysis"),
                "fund_data": r["pe_fund_data"]
            }
            for r in results["enhanced_results"]
        ]
    }
    
    console.print_json(json.dumps(json_results, indent=2, default=str))