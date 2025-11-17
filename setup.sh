#!/bin/bash
# PHOENIX Setup Script
# This script helps you set up everything needed to run PHOENIX

echo "════════════════════════════════════════════"
echo "         PHOENIX AI SYSTEM SETUP            "
echo "════════════════════════════════════════════"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}!${NC} Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install basic requirements
echo ""
echo "Installing basic requirements..."
pip install --quiet --upgrade pip
pip install --quiet pyyaml psutil ollama

echo -e "${GREEN}✓${NC} Basic dependencies installed"

# Check for Ollama
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is installed"

    # Check if Ollama is running
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✓${NC} Ollama service is running"
    else
        echo -e "${YELLOW}!${NC} Ollama is not running. Starting it now..."
        ollama serve &> /dev/null &
        sleep 3
    fi
else
    echo -e "${YELLOW}!${NC} Ollama not found. Installing..."
    echo "Would you like to install Ollama now? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
        echo -e "${GREEN}✓${NC} Ollama installed"
        # Start Ollama
        ollama serve &> /dev/null &
        sleep 3
    else
        echo -e "${YELLOW}!${NC} You'll need to install Ollama manually:"
        echo "    curl -fsSL https://ollama.com/install.sh | sh"
    fi
fi

# Check for required model
echo ""
echo "Checking for Qwen 2.5 model..."
if ollama list 2>/dev/null | grep -q "qwen2.5:14b"; then
    echo -e "${GREEN}✓${NC} Qwen 2.5 14B model is available"
else
    echo -e "${YELLOW}!${NC} Qwen 2.5 14B model not found"
    echo "This is an 8GB download. Would you like to download it now? (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        echo "Downloading Qwen 2.5 14B..."
        ollama pull qwen2.5:14b
        echo -e "${GREEN}✓${NC} Model downloaded"
    else
        echo "You can download it later with: ollama pull qwen2.5:14b"
        echo "Trying a smaller model instead..."
        echo "Downloading Llama 3.2 3B (smaller, faster)..."
        ollama pull llama3.2:3b
        echo -e "${GREEN}✓${NC} Llama 3.2 3B downloaded as fallback"
    fi
fi

# Make scripts executable
chmod +x phoenix.py

echo ""
echo "════════════════════════════════════════════"
echo -e "${GREEN}        Setup Complete!${NC}"
echo "════════════════════════════════════════════"
echo ""
echo "To start PHOENIX:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run PHOENIX:"
echo "     python phoenix.py"
echo ""
echo "Or run tests first:"
echo "     python phoenix.py --test"
echo ""
echo "For help:"
echo "     python phoenix.py --help"
echo ""