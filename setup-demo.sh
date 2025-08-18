#!/bin/bash
# PE-SMB Matcher Demo Setup Script
# Downloads large datasets and prepares system for full functionality

set -e

echo "🚀 Setting up PE-SMB Matcher Demo Environment..."

# Check if Python 3.9+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3.9+ is required. Please install Python first."
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
cd api && pip install -r requirements.txt
cd ../cli && pip install -r requirements.txt
cd ..

# Check for API key
if [ ! -f ".env" ]; then
    echo "⚙️  Setting up environment configuration..."
    cp .env.example .env
    echo ""
    echo "🔑 IMPORTANT: Edit .env file and add your GOOGLE_API_KEY"
    echo "   You can get a free API key from: https://ai.google.dev/"
    echo ""
    read -p "Press Enter after adding your API key to .env..."
fi

# Check if large databases exist
if [ ! -f "adv_database.db" ]; then
    echo "📥 Downloading SEC Form ADV data (this may take a few minutes)..."
    
    # Download ADV CSV files if they don't exist
    if [ ! -f "ADV_Base_A_20150315.csv" ]; then
        echo "   Downloading Part A data (406MB)..."
        wget -O ADV_Base_A_20150315.csv "https://www.sec.gov/files/structureddata/data/form-adv/2015-q1/ADV_Base_A_20150315.csv" || {
            echo "❌ Download failed. You can manually download from:"
            echo "   https://www.sec.gov/structureddata/form-adv-data.html"
            echo "   Place files in project directory and run: python process_adv_csvs.py"
            exit 1
        }
    fi
    
    if [ ! -f "ADV_Base_B_20150315.csv" ]; then
        echo "   Downloading Part B data (52MB)..."
        wget -O ADV_Base_B_20150315.csv "https://www.sec.gov/files/structureddata/data/form-adv/2015-q1/ADV_Base_B_20150315.csv" || {
            echo "❌ Download failed. You can manually download from:"
            echo "   https://www.sec.gov/structureddata/form-adv-data.html"  
            exit 1
        }
    fi
    
    echo "🔄 Processing ADV data into database..."
    python process_adv_csvs.py --part-a ADV_Base_A_20150315.csv --part-b ADV_Base_B_20150315.csv
    
    echo "🗂️  Building semantic search index..."
    python adv_semantic_layer.py "test query" --rebuild
    
    echo "✅ Full database setup complete!"
else
    echo "✅ Database already exists, skipping download."
fi

# Verify ChromaDB collections exist
echo "🔍 Verifying semantic search collections..."
python -c "
import os
if not os.path.exists('.chroma_pe_funds'):
    print('Building PE funds search index...')
    os.system('python pe_semantic_layer.py \"healthcare\" > /dev/null 2>&1')
if not os.path.exists('.chroma_adv_firms'):
    print('Building ADV firms search index...')  
    os.system('python adv_semantic_layer.py \"test\" --rebuild > /dev/null 2>&1')
print('✅ Search indexes ready!')
"

echo ""
echo "🎉 Setup Complete! You can now:"
echo "   1. Start the server: ./start-server"
echo "   2. Analyze companies: ./analyze https://company.com"
echo ""
echo "💡 For help: ./analyze --help"
echo "📚 Full documentation: README.md"