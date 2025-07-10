# SMB-PE Agent

An intelligent AI agent system for analyzing small-to-medium businesses (SMBs) and matching them with relevant private equity (PE) funds. Built with Google Gemini 2.0 Flash and PydanticAI, the system provides comprehensive company analysis and intelligent fund matching through semantic search.

## Features

- **AI-Powered SMB Analysis**: Uses Google Gemini 2.0 Flash with PydanticAI for comprehensive company analysis
- **LLM Investment Analysis**: Sophisticated investment attractiveness scoring and strategic recommendations
- **Semantic PE Fund Matching**: ChromaDB vector search over both synthetic and real SEC Form ADV data
- **CLI Interface**: Command-line tool with `./analyze` wrapper for quick company analysis
- **REST API**: FastAPI-based backend with async support for programmatic access
- **Confidence Scoring**: Multi-level confidence assessment for data reliability
- **Cost Tracking**: Token usage and API cost monitoring for budget management

## Quick Start

### Prerequisites

- Python 3.9+
- Google API key for Gemini

### Installation

```bash
# Clone repository
git clone https://github.com/brandongalang/SMB-PE-Agent.git
cd SMB-PE-Agent

# Install dependencies
pip install -r requirements.txt
cd api && pip install -r requirements.txt
cd ../cli && pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env
```

### Usage

#### CLI Analysis (Recommended)
```bash
# Basic company analysis with PE matching
./analyze https://company.com

# Advanced LLM analysis with customized results
./analyze https://dempstereye.com --top-k 3 --candidate-pool-size 20

# With company name hint for better analysis
./analyze https://company.com --company-name "Acme Corporation"

# Save results to file
./analyze https://company.com --top-k 15 --format json --save results.json

# Disable LLM analysis for faster results
./analyze https://company.com --no-llm-analysis
```

#### CLI Analysis (Manual)
```bash
# Alternative method requiring PYTHONPATH
PYTHONPATH=. python3 cli/main.py https://company.com --top-k 10
```

#### API Server
```bash
# Start API server
PYTHONPATH=. python3 -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Access documentation
open http://localhost:8000/docs
```

#### API Usage
```bash
# Complete SMB analysis with PE matches
curl -X POST "http://localhost:8000/analysis/with-pe-matches?website_url=https://company.com&top_k=10" \
     -H "accept: application/json"

# Health check
curl http://localhost:8000/health
```

## Architecture

### Core Components

**SMB Analysis Agent** (`api/agents/smb_agent.py`)
- Built with PydanticAI framework
- Uses Google Gemini 2.0 Flash with native search grounding
- Provides structured SMBCompanyData output with confidence scoring
- Comprehensive error handling and fallback mechanisms

**Semantic Search Layer** (`pe_semantic_layer.py`, `pe_fund_matcher.py`)
- ChromaDB vector database for semantic search
- SQLite database with PE fund profiles
- Intelligent company-to-fund matching algorithms

**API Layer** (`api/`)
- FastAPI with async support
- Unified analysis endpoints with confidence scoring
- Token usage tracking and cost estimation

**CLI Layer** (`cli/`)
- Rich console interface with progress indicators
- Flexible output formats (table, JSON)
- File saving and batch processing support

### Data Flow

1. **Input**: Company website URL (+ optional company name hint)
2. **Analysis**: SMB agent analyzes website using Google Search grounding
3. **Matching**: Semantic search finds relevant PE funds from database
4. **Output**: Structured results with confidence scores and cost tracking

## API Endpoints

### Analysis Endpoints
- `POST /analysis/with-pe-matches` - **Unified SMB analysis with PE matching (recommended)**
- `POST /analysis/detailed` - Detailed company analysis with confidence scores
- `POST /analysis/summary` - Quick analysis for rapid insights
- `POST /analysis/bulk` - Multiple company analysis

### Utility Endpoints
- `GET /health` - Service health check and API key validation
- `GET /docs` - Interactive API documentation
- `GET /` - API information and available endpoints

## Data Models

### SMBCompanyData
Comprehensive SMB analysis including:
- Company identification and business description
- Industry classification (NAICS codes, keywords)
- Size estimation (employees, revenue, size band)
- Business model and revenue streams
- Growth indicators and competitive advantages
- PE attractiveness factors and defensibility
- Confidence scores per field

### PE Fund Matching
- Semantic similarity scoring
- Investment stage alignment
- Sector focus matching
- Geographic relevance
- Match reasoning explanation

## Configuration

### Environment Variables
```env
GEMINI_API_KEY=your_gemini_api_key_here  # Required
CHROMA_DB_PATH=.chroma_pe_funds         # Optional: ChromaDB storage location
DATABASE_URL=sec_adv_synthetic.db       # Optional: SQLite database path
```

### Dependencies
- **PydanticAI**: AI agent framework
- **Google Generative AI**: Gemini models with search grounding
- **ChromaDB**: Vector database for semantic search
- **FastAPI**: High-performance web framework
- **SQLAlchemy**: SQL toolkit and ORM
- **Rich**: Console formatting for CLI
- **Click**: CLI framework

## Database Schema

### SQLite Tables (`sec_adv_synthetic.db`)
- `pe_funds`: PE fund profiles with investment criteria and focus areas
- `firms`: Firm information with geographic and sector data

### ChromaDB Collections
- `pe_funds`: Vector embeddings of PE fund descriptions for semantic search
- Persistent storage in `.chroma_pe_funds/` directory

## Performance

- **Analysis Time**: 10-30 seconds for complex company analysis
- **Concurrent Requests**: Async FastAPI supports multiple simultaneous analyses
- **Cost Efficiency**: Token usage tracking helps monitor API costs
- **Scalability**: ChromaDB vector search scales with database size

## Development

### Project Structure
```
├── analyze                 # CLI wrapper script (recommended entry point)
├── api/                    # FastAPI backend
│   ├── agents/            # AI analysis agents (SMB agent)
│   ├── models/            # Data models and schemas
│   ├── routers/           # API route definitions
│   └── main.py           # FastAPI application
├── cli/                   # Command-line interface
│   ├── main.py           # CLI entry point
│   ├── llm_integration.py # LLM analysis pipeline
│   └── formatters.py     # Output formatting
├── pe_semantic_layer.py   # ChromaDB integration for synthetic data
├── adv_semantic_layer.py  # ChromaDB integration for SEC Form ADV data
├── enhanced_fund_matcher.py # Combined matching over multiple data sources
├── pe_fund_matcher.py     # Matching algorithms
├── data_models.py         # Shared data structures
├── sec_adv_synthetic.db   # PE fund database (excluded from repo)
├── adv_database.db       # SEC Form ADV database (~349K firms, excluded from repo)
└── .env.example          # Environment configuration template
```

### Testing
```bash
# Test core functionality
python pe_semantic_layer.py "healthcare services"
python pe_fund_matcher.py

# Test API endpoints
curl http://localhost:8000/health

# Test CLI (recommended method)
./analyze https://example.com --top-k 3

# Test CLI (manual method)
PYTHONPATH=. python3 cli/main.py https://example.com --no-llm-analysis
```

## Troubleshooting

### Common Issues

**API Server Won't Start**
```bash
export PYTHONPATH=.
export GEMINI_API_KEY=your_key_here
python3 -c "from api.main import app; print('Imports OK')"
```

**CLI Import Errors**
```bash
# Ensure relative imports work
cd cli && python3 -c "from .main import main; print('CLI OK')"
```

**ChromaDB Collection Issues**
```bash
# Collections auto-create if missing
# Check for .chroma_* directories in project root
ls -la .chroma_*
```

### Performance Optimization
- Use `--top-k` to limit PE fund results for faster responses
- Set appropriate timeouts for complex analyses
- Monitor token usage with cost tracking features
- Use batch processing for multiple companies

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For issues and feature requests, please check the documentation in `CLAUDE.md` or create an issue in the repository.