# Development Guide

## 🛠 Development Environment Setup

### Prerequisites
- Docker Desktop
- Python 3.11 or later
- Git
- VSCode (recommended)
- LibreOffice (for PDF conversion)

### Docker Development Environment (Recommended)

1. **Initial Setup**
```bash
# Clone repository
git clone https://github.com/CS-Japan-SE/DocTranslator.git
cd DocTranslator

# Create environment file
cp .env.example .env
# Edit .env and add your GenAI Hub API key

# Start development environment
docker-compose -f docker-compose.dev.yml up --build
```

2. **Development Commands**
```bash
# Start development server
docker-compose -f docker-compose.dev.yml up

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Run tests
docker-compose -f docker-compose.dev.yml exec document-translator pytest

# Format code
docker-compose -f docker-compose.dev.yml exec document-translator black .

# Run linter
docker-compose -f docker-compose.dev.yml exec document-translator flake8
```

### Local Development Environment

1. **Setup Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

2. **Start Development Server**
```bash
uvicorn app.main:app --reload
```

## 📁 Project Structure

```
DocTranslator/
├── app/                     # Application code
│   ├── core/               # Core business logic
│   │   ├── __init__.py
│   │   └── translator.py   # Translation logic
│   ├── static/             # Static files
│   │   ├── css/
│   │   ├── js/
│   │   └── index.html
│   └── utils/              # Utility functions
├── docker/                 # Docker configuration
│   ├── Dockerfile         # Production Dockerfile
│   ├── Dockerfile.dev     # Development Dockerfile
│   └── nginx/             # Nginx configuration
├── docs/                   # Documentation
├── scripts/               # Development scripts
├── tests/                 # Test files
├── downloads/             # Downloaded files
├── uploads/              # Uploaded files
└── logs/                 # Log files
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_translator.py

# Run with coverage
pytest --cov=app tests/

# Run with verbose output
pytest -v
```

### Writing Tests
```python
# Example test
def test_translate_text():
    result = translate_text("Hello", "en", "ja")
    assert result == "こんにちは"
```

## 🎨 Code Style

### Code Formatting
- Use `black` for code formatting
- Maximum line length: 88 characters
- Use double quotes for strings

### Type Hints
```python
from typing import Optional, List

def translate_text(
    text: str,
    source_lang: str,
    target_lang: str,
    api_key: Optional[str] = None
) -> str:
    ...
```

### Documentation
- Use docstrings for all functions and classes
- Follow Google style docstrings
```python
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Translated text

    Raises:
        ValueError: If language codes are invalid
    """
    ...
```

## 🔧 Development Tools

### VSCode Extensions
- Python
- Docker
- Black Formatter
- GitLens
- Python Test Explorer

### VSCode Settings
```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true
}
```

## 📦 Dependencies Management

### Adding New Dependencies
1. Add to appropriate requirements file:
   - `requirements.txt` for production
   - `requirements-dev.txt` for development

2. Update Docker environment:
```bash
docker-compose -f docker-compose.dev.yml build --no-cache
```

### Updating Dependencies
```bash
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-dev.txt
```

## 🔄 Git Workflow

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make Changes**
- Write code
- Add tests
- Update documentation

3. **Commit Changes**
```bash
git add .
git commit -m "feat: add new feature"
```

4. **Push Changes**
```bash
git push origin feature/your-feature-name
```

5. **Create Pull Request**
- Describe changes
- Reference issues
- Request review

## 🐛 Debug Tips

### Logging
```python
import logging
logger = logging.getLogger(__name__)

logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

### Debugging in Docker
```bash
# Access container shell
docker-compose -f docker-compose.dev.yml exec document-translator bash

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Install debugging tools
pip install ipdb

# Use interactive debugger
python -m ipdb your_script.py
```

## 🔒 Security Best Practices

1. **Environment Variables**
- Never commit sensitive data
- Use .env for local development
- Use secure secrets management in production

2. **Input Validation**
- Validate all user inputs
- Sanitize file names
- Check file types and sizes

3. **Error Handling**
- Don't expose internal errors to users
- Log errors appropriately
- Provide user-friendly error messages

## 📈 Performance Tips

1. **Optimization**
- Use async/await for I/O operations
- Implement caching where appropriate
- Optimize database queries

2. **Resource Management**
- Clean up temporary files
- Close file handles properly
- Monitor memory usage

## 📚 Learning Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [Docker Documentation](https://docs.docker.com/)
- [pytest Documentation](https://docs.pytest.org/)

## 🤝 Contributing Guidelines

### Code Review Process

1. **Before Submitting**
- Run all tests
- Format code with black
- Update documentation
- Check for security issues

2. **Pull Request Requirements**
- Clear description
- Test coverage
- Documentation updates
- Clean commit history

3. **Review Criteria**
- Code quality
- Test coverage
- Documentation
- Security considerations

### Development Workflow

1. **Planning**
- Create issue
- Discuss implementation
- Define acceptance criteria

2. **Implementation**
- Create feature branch
- Write tests
- Implement feature
- Update documentation

3. **Review**
- Submit pull request
- Address feedback
- Update changes

4. **Merge**
- Squash commits
- Merge to main
- Delete feature branch

## 🔧 Maintenance

### Regular Tasks

1. **Code Cleanup**
```bash
# Format code
black .

# Remove unused imports
autoflake --recursive --in-place --remove-all-unused-imports .

# Sort imports
isort .
```

2. **Dependency Updates**
```bash
# Check for updates
pip list --outdated

# Update dependencies
pip install --upgrade -r requirements.txt
pip install --upgrade -r requirements-dev.txt
```

3. **Test Coverage**
```bash
# Run coverage report
pytest --cov=app tests/ --cov-report=html
```

### Documentation Updates

1. **API Documentation**
- Update API endpoints
- Update request/response examples
- Update error messages

2. **Development Guide**
- Update setup instructions
- Update troubleshooting guide
- Update best practices

3. **README Updates**
- Update features
- Update requirements
- Update installation steps

## 🔍 Code Review Checklist

### General
- [ ] Code follows style guide
- [ ] Documentation is updated
- [ ] Tests are added/updated
- [ ] Error handling is appropriate
- [ ] Performance considerations
- [ ] Security considerations

### Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Edge cases covered
- [ ] Error cases tested
- [ ] Performance tested

### Security
- [ ] Input validation
- [ ] Error handling
- [ ] Secure configuration
- [ ] File handling
- [ ] API security

### Documentation
- [ ] Code comments
- [ ] API documentation
- [ ] README updates
- [ ] Change log
- [ ] License compliance
```