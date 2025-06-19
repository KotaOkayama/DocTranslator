#!/bin/bash

echo "🚀 Starting DocTranslator..."

# Check .env file existence
if [ ! -f ".env" ]; then
    echo "❌ .env file not found"
    echo "💡 Copy .env.example to .env and set your API key"
    exit 1
fi

# Check if Docker Desktop is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker Desktop is not running"
    echo "💡 Please start Docker Desktop and try again"
    exit 1
fi

# Start application
echo "🐳 Starting Docker containers..."
docker-compose up --build

echo "✅ DocTranslator is now running!"
echo "🌐 Access http://localhost:8000 in your browser"
