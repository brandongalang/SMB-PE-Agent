# 🚀 PE-SMB Matcher - Quick Start Demo

**Ready-to-run demo in 3 simple steps!**

## Prerequisites
- Python 3.9+
- Google API key (free from [ai.google.dev](https://ai.google.dev/))

## Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd PE-SMB-Matcher-POC
./setup-demo.sh
```

### 2. Add API Key
Edit `.env` file and add:
```env
GOOGLE_API_KEY=your_api_key_here
```

### 3. Run Demo
```bash
# Start API server (in one terminal)
./start-server

# Analyze companies (in another terminal)
./analyze https://dempstereye.com
./analyze https://hubspot.com --top-k 5
```

## What You Get

**Immediate Demo Capability:**
- ✅ Works out-of-the-box with synthetic PE database (11MB)
- ✅ Full company analysis with AI reasoning
- ✅ PE fund matching and investment recommendations
- ✅ Rich CLI output with progress indicators

**Full Dataset (Optional):**
- 📥 Setup script downloads real SEC Form ADV data (349K firms)
- 🗂️ Builds comprehensive semantic search indexes
- 💰 Production-ready matching over 349K real PE/advisory firms

## Demo Companies to Try

```bash
# Healthcare/Medical Companies
./analyze https://dempstereye.com
./analyze https://modernhealthcare.com

# Technology Companies  
./analyze https://hubspot.com
./analyze https://salesforce.com

# Small Business Examples
./analyze https://example-smb.com --company-name "Local Services Co"
```

## Key Features Demonstrated

- 🧠 **AI-Powered Analysis**: Google Gemini 2.0 extracts investment-relevant data
- 🎯 **Smart Matching**: Semantic search finds relevant PE funds
- 💡 **Investment Reasoning**: Sophisticated analysis of strategic fit
- 📊 **Rich Output**: Confidence scores, cost tracking, detailed recommendations
- ⚡ **Fast Results**: Complete analysis in 30-60 seconds

## Output Example

```
🏢 COMPANY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Company: Dempster Eye Center 
Industry: Healthcare Services, Ophthalmology
Size: Small (10-50 employees, $1-10M revenue)
Growth Stage: Established

💪 Investment Strengths:
• Specialized healthcare niche with defensible market position
• Recurring patient relationships and procedure revenue
• Scalable service model with expansion potential

🎯 PE FUND MATCHES (Ranked by Strategic Fit)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🥇 HealthTech Growth Partners (92% Match)
   Focus: Healthcare Services, Medical Devices
   Investment Thesis: Strong strategic alignment with healthcare 
   specialization and growth potential...
```

## Troubleshooting

**"Could not connect to API":** Start server with `./start-server`

**"Invalid API key":** Add GOOGLE_API_KEY to `.env` file

**"No results found":** Run full setup with `./setup-demo.sh`

**Need help?** Check `README.md` or `CLAUDE.md` for detailed documentation.

---
*This demo showcases AI-powered SMB analysis and PE matching. Perfect for investors, consultants, and business development professionals.*