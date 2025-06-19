.PHONY: help dev-setup start test lint format clean

help:
	@echo "Available commands:"
	@echo "  make dev-setup  - Set up development environment"
	@echo "  make start      - Start development server"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean up temporary files"

dev-setup:
	@echo "Setting up development environment..."
	docker-compose -f docker-compose.dev.yml build --no-cache
	@echo "Creating necessary directories..."
	mkdir -p uploads downloads logs
	touch uploads/.gitkeep downloads/.gitkeep logs/.gitkeep
	@echo "Development environment setup complete."

start:
	@echo "Starting development server..."
	docker-compose -f docker-compose.dev.yml up

test:
	@echo "Checking if container is running..."
	@if ! docker-compose -f docker-compose.dev.yml ps | grep -q "document-translator.*running"; then \
		echo "Starting container in detached mode..."; \
		docker-compose -f docker-compose.dev.yml up -d; \
		echo "Waiting for container to be ready..."; \
		sleep 15; \
	fi
	@echo "Running tests..."
	docker-compose -f docker-compose.dev.yml exec -T -e TESTING=true document-translator python -m pytest -v
	@echo "Tests completed."
	@echo "Restoring original .env file..."
	@if [ -f .env.backup ]; then \
		cp .env.backup .env; \
		rm .env.backup; \
	fi

lint:
	@echo "Checking if container is running..."
	@if ! docker-compose -f docker-compose.dev.yml ps | grep -q "document-translator.*running"; then \
		echo "Starting container in detached mode..."; \
		docker-compose -f docker-compose.dev.yml up -d; \
		echo "Waiting for container to be ready..."; \
		sleep 10; \
	fi
	@echo "Running linter..."
	docker-compose -f docker-compose.dev.yml exec -T document-translator flake8 . --ignore=E501,W503,F401,E402
	@echo "Linting completed."

format:
	@echo "Checking if container is running..."
	@if ! docker-compose -f docker-compose.dev.yml ps | grep -q "document-translator.*running"; then \
		echo "Starting container in detached mode..."; \
		docker-compose -f docker-compose.dev.yml up -d; \
		echo "Waiting for container to be ready..."; \
		sleep 10; \
	fi
	@echo "Formatting code..."
	docker-compose -f docker-compose.dev.yml exec -T document-translator black .
	@echo "Formatting completed."

clean:
	@echo "Cleaning up temporary files..."
	docker-compose -f docker-compose.dev.yml down -v
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.pyd" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name "*.egg" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -r {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -r {} + 2>/dev/null || true
	@echo "Cleanup completed."
