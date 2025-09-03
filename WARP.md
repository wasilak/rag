# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a **RAG (Retrieval-Augmented Generation) system** that transforms documents into an intelligent, conversational AI assistant. It supports multiple LLM providers (Ollama, OpenAI, Gemini), processes various document types (Markdown, URLs, PDFs), and provides both TUI and web interfaces for interaction.

## Essential Commands

### Development Setup
```bash
# Install Python dependencies (requires Python 3.12+)
uv sync

# Setup full development environment (Python + web)
make setup

# Install only web dependencies
make install-web
```

### Core Operations
```bash
# Fill knowledge base with documents
python main.py data-fill examples/ --cleanup --clean-content
python main.py data-fill https://example.com --source-type url --clean-content

# Search documents
python main.py search "your query here"

# Start TUI chat interface
python main.py chat

# Start web interface (production)
make web

# Development mode (two terminals)
make dev-web     # React dev server on :3000
make web-dev     # Flask backend on :8080
```

### Testing & Quality
```bash
# Linting
make lint
make lint-fix

# Format code
make format

# Type checking
make type-check
```

### Docker
```bash
# Build and run with Docker
docker build -t rag .
docker run -it --rm -p 8080:8080 rag web --host 0.0.0.0
```

## High-Level Architecture

### Core Components

**Main Entry Point**: `main.py` - CLI dispatcher that routes to specific command processors

**Document Processing Pipeline**:
- `libs/commands/data_fill/` - Document ingestion, cleaning, chunking, and embedding
- `libs/commands/data_fill/documents_types/` - Type-specific loaders (file, url, markdown)
- `libs/commands/data_fill/embedding.py` - Multi-provider embedding functions
- `libs/commands/data_fill/cleaning.py` - LLM-powered content cleaning

**RAG Core**:
- `libs/rag_orchestrator.py` - Advanced iterative search with self-improving queries
- `libs/search_orchestrator.py` - Search coordination and result formatting
- `libs/commands/search/search.py` - Basic search implementation

**LLM Integration**:
- `libs/models.py` - Model validation, defaults, and provider management
- `libs/models_gemini.py` - Google Gemini specific implementations
- `libs/utils.py` - OpenAI client creation and tokenization

**Interfaces**:
- `libs/chat.py` - Terminal chat interface using Textual
- `libs/web.py` - Flask web server with Socket.IO for real-time streaming
- `web/` - React frontend with Material-UI components

**Storage**:
- ChromaDB for vector storage (persistent or HTTP client)
- SQLite for chat history (`libs/chat_storage.py`)
- S3 integration for document backup

### Key Architectural Patterns

**Multi-Provider LLM Support**: The system abstracts LLM interactions through a common interface supporting Ollama (local), OpenAI, and Gemini with automatic model validation and fallback.

**Iterative Search Refinement**: `RAGOrchestrator` implements self-improving search that uses LLMs to evaluate result quality and refine queries across multiple iterations until sufficient relevance is achieved.

**Document Processing Pipeline**: Modular document loaders handle different source types, with optional LLM-powered cleaning to remove navigation/ads, and Fabric AI integration for wisdom extraction.

**Dual Interface Architecture**: Both terminal (Textual) and web (React + Flask) interfaces share the same RAG backend, with real-time streaming via WebSocket for responsive user experience.

**Flexible Configuration**: Extensive environment variable support and command-line arguments for all aspects (models, hosts, ports, processing options).

## Development Context

### Technology Stack
- **Backend**: Python 3.12+, Flask, ChromaDB, LangChain
- **Frontend**: React 19, TypeScript, Material-UI, Socket.IO
- **Package Management**: uv (Python), Yarn (Node.js)
- **Deployment**: Docker with multi-stage builds

### Key Dependencies
- `chromadb` - Vector database for embeddings
- `textual` - Terminal UI framework
- `langchain-community` - Document loading and processing
- `ollama`, `openai`, `google-generativeai` - LLM providers
- `flask-socketio` - Real-time web communication

### Environment Variables
Key variables for configuration (see `libs/args.py` for complete list):
- `RAG_LLM`, `RAG_MODEL` - LLM provider and model
- `RAG_EMBEDDING_LLM`, `RAG_EMBEDDING_MODEL` - Embedding configuration  
- `RAG_CHROMADB_HOST`, `RAG_CHROMADB_PORT` - Vector DB connection
- `OPENAI_API_KEY`, `GEMINI_API_KEY` - API credentials

### Model Selection System
The system includes intelligent model validation (`libs/models.py`) that:
- Validates requested models against provider APIs
- Falls back to known-good defaults if validation fails
- Handles Ollama tag variations (e.g., `model:latest` vs `model`)
- Provides `list-models` command for discovery

### Web Development Workflow
- React dev server watches source files for hot reload
- Flask backend serves API and built static files
- Production builds static React app into `web/build/`
- Docker multi-stage build optimizes final image size

## Common Development Tasks

### Adding New Document Types
1. Create loader in `libs/commands/data_fill/documents_types/`
2. Register in `libs/commands/data_fill/documents.py`
3. Add any new CLI arguments to `libs/args.py`

### Adding New LLM Providers
1. Add provider to `libs/models.py` defaults and listing
2. Update `libs/commands/data_fill/embedding.py` for embeddings
3. Add to `libs/utils.py` for client creation
4. Update CLI choices in `libs/args.py`

### Modifying Search Behavior
- Basic search: `libs/commands/search/search.py`
- Advanced iterative search: `libs/rag_orchestrator.py`
- Result formatting: `libs/utils.py` `format_footnotes()`

### Web Interface Changes
- React components in `web/src/`
- Backend API endpoints in `libs/web.py`
- Real-time messaging via Socket.IO events

The codebase emphasizes modularity, multi-provider support, and user experience through both programmatic and interactive interfaces.
