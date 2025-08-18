# Simple Usage Guide

## Quick Start

### 1. Start the API Server (once per session)
```bash
cd "/Users/brandongalang/Documents/PE-SMB Matcher POC"
./start-server
```

### 2. Analyze Companies
```bash
# Basic usage - analyzes company and finds top 10 PE fund matches with LLM analysis
./analyze https://company.com

# Examples
./analyze https://dempstereye.com
./analyze https://hubspot.com
./analyze https://example-smb.com
```

## Options

```bash
# Get more or fewer PE fund matches in output (default is 10)
./analyze https://company.com --top-k 20

# Analyze more PE fund candidates before ranking (default is 50)
./analyze https://company.com --candidate-pool-size 100

# Combine both for comprehensive analysis
./analyze https://company.com --top-k 30 --candidate-pool-size 150

# Skip LLM analysis for faster basic results
./analyze https://company.com --no-llm-analysis

# Save results to a file
./analyze https://company.com --save results.json

# Output as JSON instead of formatted table
./analyze https://company.com --format json

# Provide company name hint if website detection fails
./analyze https://company.com --company-name "Acme Corp"
```

### Understanding the Parameters

- **`--top-k`**: How many PE funds to show in the final output (after LLM ranking)
- **`--candidate-pool-size`**: How many PE funds the LLM analyzes before selecting the top-k

For example:
- `--top-k 20 --candidate-pool-size 100` means: Analyze 100 funds, show the best 20
- Default (`--top-k 10 --candidate-pool-size 50`) means: Analyze 50 funds, show the best 10

## What You Get

### With Default LLM Analysis (recommended):
- 🏢 Comprehensive company profile with investment-relevant data
- 🧠 Sophisticated investment analysis and PE fund rankings
- 🎯 Top 10 PE funds ranked by strategic fit (from 50 analyzed)
- 📋 Strategic recommendations for fundraising approach
- 💰 Detailed investment thesis for each fund

### Without LLM Analysis (--no-llm-analysis):
- 🏢 Company profile and key metrics
- 🎯 Top PE fund matches based on semantic similarity
- 💰 Basic fit assessment

## Features

- **Rich Company Analysis**: Captures scalability factors, defensibility, growth indicators, and more
- **Smart PE Matching**: Matches based on complete company context, not just basic info
- **LLM Investment Reasoning**: Uses Google Gemini to provide sophisticated investment analysis
- **Real SEC Data**: Searches both synthetic and real Form ADV data for PE funds
- **Fast Results**: Typically completes in 30-60 seconds

## Troubleshooting

If you get "Could not connect to API server":
1. Make sure you started the server with `./start-server`
2. Check that port 8000 is not in use
3. Try `killall python3` and restart the server

For other issues, check the logs in the terminal where you started the server.