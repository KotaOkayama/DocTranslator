#!/bin/bash

echo "🧪 Running DocTranslator test suite..."

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Code formatting check
echo "📝 Checking code formatting..."
if ! black --check app/ tests/; then
    echo "❌ Code formatting errors found"
    echo "💡 Run 'black app/ tests/' to fix"
    exit 1
fi

# Lint check
echo "🔍 Running lint check..."
if ! flake8 app/ tests/; then
    echo "❌ Lint errors found"
    exit 1
fi

# Type checking
echo "🔍 Running type checking..."
if ! mypy app/; then
    echo "❌ Type errors found"
    exit 1
fi

# Run tests
echo "🧪 Running tests..."
if ! pytest --cov=app tests/ --cov-report=html --cov-report=term; then
    echo "❌ Tests failed"
    exit 1
fi

echo "✅ All checks completed successfully!"
echo "📊 Coverage report: htmlcov/index.html"
echo "🎉 Code quality looks great!"
