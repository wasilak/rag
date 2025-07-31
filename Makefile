# RAG Project Makefile

.PHONY: help install install-web build-web dev-web clean-web web test lint format

# Default target
help: ## Show this help message
	@echo "RAG Project - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Python backend commands
install: ## Install Python dependencies
	uv sync

dev: ## Run development server (TUI chat)
	python main.py chat

data-fill: ## Fill database with sample data (requires SOURCE variable)
	python main.py data-fill $(SOURCE) --cleanup

search: ## Search in database (requires QUERY variable)
	python main.py search "$(QUERY)"

list-models: ## List available models (requires PROVIDER variable)
	python main.py list-models $(PROVIDER)

# Web frontend commands
install-web: ## Install web dependencies
	cd web && yarn install

build-web: ## Build web frontend for production
	cd web && yarn build

dev-web: ## Start web development server
	cd web && yarn start

clean-web: ## Clean web build artifacts
	rm -rf web/build web/node_modules

# Combined web interface
web: install-web build-web ## Build web frontend and start web server
	python main.py web

web-dev: ## Start web server in development mode (requires separate 'make dev-web')
	python main.py web --debug --port 5000

# Testing and quality
test: ## Run tests
	python -m pytest tests/ -v

lint: ## Run linters
	python -m ruff check .
	cd web && yarn lint

lint-fix: ## Fix linting issues
	python -m ruff check . --fix
	cd web && yarn lint:fix

format: ## Format code
	python -m ruff format .

type-check: ## Run type checking
	python -m mypy .
	cd web && yarn type-check

# Docker commands
docker-build: ## Build Docker image
	docker build -t rag-app .

docker-run: ## Run Docker container
	docker run -p 5000:5000 rag-app

# Cleanup
clean: clean-web ## Clean all build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

# Development setup
setup: install install-web ## Setup development environment
	@echo "✅ Development environment setup complete!"
	@echo "To start TUI chat: make dev"
	@echo "To start web interface: make web"
	@echo "To start web development: make dev-web (in one terminal) and make web-dev (in another)"

# Production build
build: build-web ## Build production assets
	@echo "✅ Production build complete!"
	@echo "To start production web server: make web"

# Examples
example-fill: ## Fill database with example documents
	python main.py data-fill examples/ --cleanup --clean-content

example-search: ## Search example documents
	python main.py search "How can I learn Rust effectively?"

example-chat: ## Start chat with example documents
	python main.py chat

example-web: ## Start web interface with example documents
	python main.py web
