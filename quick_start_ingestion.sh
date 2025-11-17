#!/bin/bash
# PHOENIX Google Takeout Quick Start
# Run this script to begin the full ingestion process

set -e

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║     PHOENIX - Google Takeout Integration Quick Start         ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if running from PHOENIX directory
if [ ! -f "phoenix.py" ]; then
    echo "❌ Error: Please run this script from the PHOENIX directory"
    echo "   cd /home/user/PHOENIX"
    echo "   ./quick_start_ingestion.sh"
    exit 1
fi

echo "📋 This script will guide you through:"
echo "   1. Downloading your 148GB Google Takeout data"
echo "   2. Extracting the archives"
echo "   3. Ingesting into PHOENIX"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✅ Python 3 found"

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."

if ! python3 -c "import cryptography" 2>/dev/null; then
    echo "⚠️  Installing cryptography..."
    pip install cryptography
fi

if ! python3 -c "import networkx" 2>/dev/null; then
    echo "⚠️  Installing networkx..."
    pip install networkx
fi

echo "✅ Dependencies OK"

# Check disk space
echo ""
echo "💾 Checking disk space..."
available_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

if [ "$available_space" -lt 150 ]; then
    echo "⚠️  Warning: Less than 150GB free space"
    echo "   Available: ${available_space}GB"
    echo "   Recommended: 150GB+"
    read -p "Continue anyway? (y/n): " continue_choice
    if [ "$continue_choice" != "y" ]; then
        exit 1
    fi
else
    echo "✅ Sufficient disk space: ${available_space}GB"
fi

# Show the menu
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "What would you like to do?"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "1) Download and extract Google Takeout (guided)"
echo "2) Ingest already-extracted Takeout data"
echo "3) View ingestion progress/status"
echo "4) Read the full guide"
echo "5) Test personal data system"
echo "6) Exit"
echo ""

read -p "Choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 Starting download helper..."
        python3 download_google_takeout.py
        ;;
    2)
        echo ""
        read -p "Enter path to extracted Takeout folder: " takeout_path

        if [ ! -d "$takeout_path" ]; then
            echo "❌ Directory not found: $takeout_path"
            exit 1
        fi

        echo ""
        echo "🚀 Starting ingestion..."
        python3 ingest_takeout.py "$takeout_path"
        ;;
    3)
        echo ""
        echo "📊 Ingestion Progress:"
        echo "─────────────────────────────────────────────────────────────"

        progress_file="$HOME/.phoenix_vault/ingestion_progress.json"

        if [ -f "$progress_file" ]; then
            cat "$progress_file" | python3 -m json.tool
        else
            echo "No ingestion in progress or completed yet"
        fi

        echo ""
        echo "📁 Log files:"
        ls -lh "$HOME/.phoenix_vault/logs/" 2>/dev/null || echo "No logs yet"
        ;;
    4)
        echo ""
        echo "📖 Opening guide..."

        if command -v less &> /dev/null; then
            less GOOGLE_TAKEOUT_GUIDE.md
        else
            cat GOOGLE_TAKEOUT_GUIDE.md
        fi
        ;;
    5)
        echo ""
        echo "🧪 Running personal data system test..."
        python3 test_personal_data_ingestion.py
        ;;
    6)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ Done!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📚 For detailed instructions, see: GOOGLE_TAKEOUT_GUIDE.md"
echo "🚀 To use your personalized AI: python phoenix.py"
echo ""
