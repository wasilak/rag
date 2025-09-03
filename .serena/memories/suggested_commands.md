# Suggested Commands for RAG Project

## Development Commands

### Setup
- `make setup` - Setup complete development environment (Python + Node.js)
- `uv sync` - Install Python dependencies
- `cd web && yarn install` - Install web dependencies

### Running the Application
- `python main.py chat` - Start TUI chat interface
- `python main.py web` - Start web interface (production)
- `make dev` - Start TUI development mode
- `make web` - Build and start web interface
- `make dev-web` - Start React dev server (port 3000)
- `make web-dev` - Start Flask backend in debug mode (port 5000)

### Data Management
- `python main.py data-fill <source>` - Add documents to knowledge base
- `python main.py search "<query>"` - Search documents
- `python main.py list-models <provider>` - List available models

### Quality Assurance
- `make lint` - Run linters for both Python and web
- `make lint-fix` - Fix linting issues automatically
- `make format` - Format Python code with ruff
- `make type-check` - Run type checking (mypy + TypeScript)
- `make test` - Run Python tests with pytest

### Docker
- `docker build -t rag-app .` - Build Docker image
- `docker run -p 5000:5000 rag-app` - Run in container