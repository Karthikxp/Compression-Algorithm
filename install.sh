#!/bin/bash
# Installation script for SAAC

echo "=========================================="
echo "SAAC Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Found Python $python_version"

# Check if FFmpeg is installed
echo ""
echo "[2/5] Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version | head -n1)
    echo "  ✓ $ffmpeg_version"
else
    echo "  ✗ FFmpeg not found!"
    echo ""
    echo "  Please install FFmpeg:"
    echo "    macOS:   brew install ffmpeg"
    echo "    Ubuntu:  sudo apt-get install ffmpeg libx265-dev"
    echo "    Windows: choco install ffmpeg"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment (optional)
echo ""
echo "[3/5] Setting up Python environment..."
read -p "Create a virtual environment? (recommended) (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        echo "  Creating virtual environment..."
        python3 -m venv venv
        echo "  ✓ Virtual environment created"
    else
        echo "  ✓ Virtual environment already exists"
    fi
    
    echo "  Activating virtual environment..."
    source venv/bin/activate
fi

# Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "[5/5] Creating directories..."
mkdir -p models
mkdir -p examples/images
mkdir -p output
echo "  ✓ Directories created"

# Download YOLO model (first run will auto-download)
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run the demo:       python3 examples/demo.py"
echo "  2. Try basic usage:    python3 examples/basic_usage.py"
echo "  3. See advanced usage: python3 examples/advanced_usage.py"
echo ""
echo "Note: YOLO model will be automatically downloaded on first run."
echo ""

