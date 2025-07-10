"""
API client for communicating with the SMB Analyzer FastAPI backend
"""

import requests
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urljoin
import json


class APIError(Exception):
    """Custom exception for API-related errors"""
    pass


class SMBAnalyzerClient:
    """Client for interacting with the SMB Analyzer FastAPI endpoints"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the FastAPI server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the API
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            **kwargs: Additional arguments for requests
            
        Returns:
            Response JSON data
            
        Raises:
            APIError: If the request fails
        """
        url = urljoin(self.base_url + '/', endpoint.lstrip('/'))
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            raise APIError(f"Request timed out after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Could not connect to API server at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 422:
                try:
                    error_detail = response.json().get('detail', 'Validation error')
                    raise APIError(f"Validation error: {error_detail}")
                except:
                    raise APIError(f"Validation error: {e}")
            elif response.status_code == 500:
                try:
                    error_detail = response.json().get('detail', 'Internal server error')
                    raise APIError(f"Server error: {error_detail}")
                except:
                    raise APIError(f"Server error: {e}")
            else:
                raise APIError(f"HTTP {response.status_code}: {e}")
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {e}")
        except json.JSONDecodeError:
            raise APIError("Invalid JSON response from server")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API server is healthy
        
        Returns:
            Health status response
        """
        return self._make_request('GET', '/health')
    
    def get_pe_matches(
        self, 
        website_url: str, 
        company_name: Optional[str] = None, 
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Get SMB analysis and PE fund matches using unified SMB agent
        
        Args:
            website_url: Company website URL
            company_name: Optional company name hint
            k: Number of PE funds to return
            
        Returns:
            Analysis results with PE fund matches
        """
        params = {
            "website_url": website_url,
            "top_k": top_k,
            "include_report": False
        }
        
        if company_name:
            params["company_name"] = company_name
        
        # Use new unified endpoint for consistency
        response = self._make_request('POST', '/analysis/with-pe-matches', params=params)
        
        # Handle error responses
        if not response.get("success", True) or "error" in response:
            return {
                "success": False,
                "error": response.get("error", "Analysis failed"),
                "smb_profile": {"company_name": "Analysis Failed"},
                "matches": [],
                "confidence_level": "LOW"
            }
        
        # Extract company data safely
        company_data = response.get("company_data", {})
        
        # Convert to expected format for compatibility
        return {
            "success": response.get("success", True),
            "smb_profile": {
                "company_name": company_data.get("company_name", "Unknown"),
                "primary_industry": company_data.get("industry", "Unknown"),
                "size_band": company_data.get("size", "Unknown"),
                "headquarters_location": company_data.get("headquarters", "Unknown"),
                "business_description": company_data.get("business_description", ""),
                "key_strengths": company_data.get("key_strengths", []),
                "growth_stage": company_data.get("growth_stage", "Unknown"),
                "estimated_employees": company_data.get("estimated_employees"),
                "estimated_revenue": company_data.get("estimated_revenue"),
                # Add compatibility fields
                "business_model": company_data.get("business_model", "Unknown"),
                "confidence_scores": {"overall": response.get("confidence_level", "MEDIUM")},
                "estimated_cost_usd": response.get("estimated_cost", 0.0)
            },
            "matches": response.get("pe_matches", []),
            "confidence_level": response.get("confidence_level", "MEDIUM"),
            "total_tokens": response.get("total_tokens", 0),
            "estimated_cost": response.get("estimated_cost", 0.0),
            "execution_time": response.get("execution_time", 0.0)
        }
    
    def get_pe_matches_with_report(
        self, 
        website_url: str, 
        company_name: Optional[str] = None, 
        top_k: int = 10
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get SMB analysis, PE fund matches, and generated report using unified SMB agent
        
        Args:
            website_url: Company website URL
            company_name: Optional company name hint
            k: Number of PE funds to return
            
        Returns:
            Tuple of (analysis_results, report_data)
        """
        params = {
            "website_url": website_url,
            "top_k": top_k,
            "include_report": True
        }
        
        if company_name:
            params["company_name"] = company_name
        
        # Use new unified endpoint that uses SMB agent for both analysis and report
        full_response = self._make_request('POST', '/analysis/with-pe-matches', params=params)
        
        # Extract company data with better error handling
        company_data = full_response.get("company_data", {})
        
        # Handle case where response might not have expected structure
        if not company_data and "error" in full_response:
            # Return error response in expected format
            return {
                "success": False,
                "error": full_response.get("error", "Analysis failed"),
                "smb_profile": {"company_name": "Analysis Failed"},
                "matches": [],
                "confidence_level": "LOW"
            }, {
                "report_markdown": "Analysis failed - unable to generate report",
                "smb_profile": {"company_name": "Analysis Failed"},
                "matches": []
            }
        
        # Extract analysis results (compatible with existing format)
        analysis_results = {
            "success": full_response.get("success", True),
            "smb_profile": {
                "company_name": company_data.get("company_name", "Unknown"),
                "primary_industry": company_data.get("industry", "Unknown"), 
                "size_band": company_data.get("size", "Unknown"),
                "headquarters_location": company_data.get("headquarters", "Unknown"),
                "business_description": company_data.get("business_description", ""),
                "key_strengths": company_data.get("key_strengths", []),
                "growth_stage": company_data.get("growth_stage", "Unknown"),
                "estimated_employees": company_data.get("estimated_employees"),
                "estimated_revenue": company_data.get("estimated_revenue"),
                # Add additional fields that might be useful for compatibility
                "business_model": company_data.get("business_model", "Unknown"),
                "confidence_scores": {"overall": full_response.get("confidence_level", "MEDIUM")},
                "estimated_cost_usd": full_response.get("estimated_cost", 0.0),
                "company_stage": company_data.get("growth_stage", "Unknown")  # For formatter compatibility
            },
            "matches": full_response.get("pe_matches", []),
            "confidence_level": full_response.get("confidence_level", "MEDIUM"),
            "total_tokens": full_response.get("total_tokens", 0),
            "estimated_cost": full_response.get("estimated_cost", 0.0),
            "execution_time": full_response.get("execution_time", 0.0)
        }
        
        # Extract report data with better error handling
        executive_report = full_response.get("executive_report")
        if not executive_report or executive_report.strip() == "":
            executive_report = "# Executive Report\n\nReport generation is not available for this analysis."
        
        report_data = {
            "success": full_response.get("success", True),
            "report_markdown": executive_report,
            "smb_profile": analysis_results["smb_profile"],
            "matches": analysis_results["matches"]
        }
        
        return analysis_results, report_data
    
    def analyze_smb(
        self, 
        website_url: str, 
        company_name: Optional[str] = None,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze SMB website without PE fund matching using unified SMB agent
        
        Args:
            website_url: Company website URL
            company_name: Optional company name hint
            detailed: Whether to use detailed analysis endpoint (ignored, always detailed)
            
        Returns:
            SMB analysis results
        """
        # Use unified endpoint for consistency, just don't include PE matches in response
        params = {
            "website_url": website_url,
            "top_k": 0,  # No PE matches needed
            "include_report": False
        }
        
        if company_name:
            params["company_name"] = company_name
        
        response = self._make_request('POST', '/analysis/with-pe-matches', params=params)
        
        # Handle error responses
        if not response.get("success", True) or "error" in response:
            return {
                "success": False,
                "error": response.get("error", "Analysis failed"),
                "company_data": {"company_name": "Analysis Failed"},
                "confidence_level": "LOW"
            }
        
        # Return just the company data part for compatibility
        return {
            "success": response.get("success", True),
            "company_data": response.get("company_data", {}),
            "confidence_level": response.get("confidence_level", "MEDIUM"),
            "total_tokens": response.get("total_tokens", 0),
            "estimated_cost": response.get("estimated_cost", 0.0),
            "execution_time": response.get("execution_time", 0.0)
        }