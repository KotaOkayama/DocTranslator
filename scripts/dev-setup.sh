#!/bin/bash

echo "🚀 Setting up DocTranslator development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install

# Copy environment variables file
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment variables file..."
    cp .env.example .env
    echo "📝 Please edit .env file and set GenAI Hub API key"
fi

# Create necessary directories
mkdir -p uploads downloads logs

echo "✅ Development environment setup completed!"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env file and set GenAI Hub API key"
echo "2. Activate virtual environment with 'source venv/bin/activate'"
echo "3. Start development server with 'uvicorn app.main:app --reload'"
echo "4. Access http://localhost:8000 in your browser"
echo ""
echo "🐳 Using Docker:"
echo "docker-compose -f docker-compose.dev.yml up --build"
