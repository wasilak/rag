# RAG Project Overview

## Purpose
A powerful, flexible RAG (Retrieval-Augmented Generation) system that transforms markdown files, web content, and documentation into an interactive AI-powered knowledge base. Users can chat with their documents, search intelligently, and get accurate answers with proper source citations.

## Key Features
- Multi-LLM support (Ollama, OpenAI, Google Gemini)
- Flexible document ingestion (markdown, web URLs, batch processing)
- Interactive chat interfaces (terminal TUI and web UI)
- Intelligent search with semantic similarity
- Vector database with ChromaDB
- Document cleaning and processing capabilities

## Tech Stack
- **Backend**: Python 3.12+, Flask, FastAPI
- **Frontend**: React 19, Material-UI, TypeScript
- **AI/ML**: LangChain, ChromaDB, OpenAI, Ollama, Google Gemini
- **UI**: Textual (TUI), React with Material-UI (web)
- **Document Processing**: Unstructured, Trafilatura, BeautifulSoup
- **Package Management**: uv for Python, yarn for Node.js

## Main Entrypoints
- `main.py` - Primary CLI interface with commands: data-fill, search, chat, web, list-models
- Web interface at `web/` - React-based frontend
- Library modules in `libs/` - Core functionality modules

## project uses `uv` so all commands should be run via it, e.g. `uv run <command>`
