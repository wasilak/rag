import argparse
import os


def get_env_default(env_var, default=None):
    """Get value from environment variable with fallback to default"""
    return os.getenv(env_var, default)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with environment variable support

    Environment Variables:
    - RAG_COLLECTION: Collection name (default: "RAG")
    - RAG_CHROMADB_PATH: Database path (default: "")
    - RAG_CHROMADB_HOST: ChromaDB host (default: "127.0.0.1")
    - RAG_CHAT_DB_PATH: Path to SQLite chat history database (default: "chat_history.db")
    - RAG_CHROMADB_PORT: ChromaDB port (default: 8000)
    - RAG_LOG_LEVEL: Log level (default: "INFO")
    - RAG_EMBEDDING_MODEL: Embedding model (default: "nomic-embed-text")
    - RAG_EMBEDDING_LLM: LLM provider for embedding function (default: "ollama")
    - RAG_EMBEDDING_OLLAMA_HOST: Ollama host for embedding (default: "127.0.0.1")
    - RAG_EMBEDDING_OLLAMA_PORT: Ollama port for embedding (default: 11434)
    - RAG_LLM: LLM provider (default: "ollama")
    - RAG_MODEL: LLM model for search (default: "qwen3:8b")
    - RAG_OLLAMA_HOST: Ollama host for embedding (default: "127.0.0.1")
    - RAG_OLLAMA_PORT: Ollama port for embedding (default: 11434)
    - RAG_MODE: Processing mode for data-fill (default: "single")
    - RAG_BUCKET_NAME: S3 bucket name for uploading markdown files (optional)
    - RAG_BUCKET_PATH: S3 bucket path for uploading markdown files (optional)
    - RAG_UPLOAD_TO_S3: Upload parsed markdown to S3 (default: "false")
    - RAG_ENABLE_CLEANING: Enable document cleaning (default: "false")
    - RAG_EXTRACT_WISDOM: Enable Fabric wisdom extraction (default: "false")
    - RAG_FABRIC_COMMAND: Fabric command name (default: "fabric")
    - RAG_FABRIC_PATTERN: Fabric pattern to use for wisdom extraction (default: "create_micro_summary")
    - RAG_CHUNK_SIZE: Size of text chunks for splitting (default: 600)
    - RAG_CHUNK_OVERLAP: Overlap between chunks (default: 200)
    - RAG_WEB_PORT: Web server port (default: 8080)
    - RAG_WEB_HOST: Web server host (default: "127.0.0.1")
    - RAG_WEB_DEBUG: Enable web debug mode (default: "false")
    - RAG_WEB_BROWSER: Auto-open browser when starting web interface (default: false)
    - RAG_WEB_CORS_ORIGINS: Comma-separated CORS origins (optional)
    - RAG_WEB_SECRET_KEY: Flask secret key (default: "rag-web-secret-key")
    - RAG_WEB_MAX_HISTORY: Max conversation history (default: 50)
    - RAG_WEB_TIMEOUT: Request timeout in seconds (default: 300)
    - RAG_WEB_WORKERS: Number of worker processes (default: 1)
    """
    # Create a parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--collection",
        type=str,
        default=get_env_default("RAG_COLLECTION", "RAG"),
        help="Name of the collection to use or create (env: RAG_COLLECTION)",
    )
    parent_parser.add_argument(
        "--chromadb-path",
        type=str,
        default=get_env_default("RAG_CHROMADB_PATH", ""),
        help="Path to the ChromaDB database (env: RAG_CHROMADB_PATH)",
    )
    parent_parser.add_argument(
        "--chromadb-host",
        type=str,
        default=get_env_default("RAG_CHROMADB_HOST", "127.0.0.1"),
        help="Host for the ChromaDB server (env: RAG_CHROMADB_HOST)",
    )
    parent_parser.add_argument(
        "--chromadb-port",
        type=int,
        default=int(get_env_default("RAG_CHROMADB_PORT", 8000)),
        help="Port for the ChromaDB server (env: RAG_CHROMADB_PORT)",
    )
    parent_parser.add_argument(
        "--chat-db-path",
        type=str,
        default=get_env_default("RAG_CHAT_DB_PATH", "chat_history.db"),
        help="Path to SQLite chat history database (env: RAG_CHAT_DB_PATH)",
    )
    parent_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode without making changes",
    )
    parent_parser.add_argument(
        "--log-level",
        type=str,
        default=get_env_default("RAG_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level (env: RAG_LOG_LEVEL)",
    )
    parent_parser.add_argument(
        "--embedding-model",
        type=str,
        default=get_env_default("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
        help="Embedding model to use (env: RAG_EMBEDDING_MODEL)",
    )
    parent_parser.add_argument(
        "--embedding-llm",
        type=str,
        default=get_env_default("RAG_EMBEDDING_LLM", "ollama"),
        choices=["openai", "ollama", "gemini"],
        help="LLM provider for embedding function (env: RAG_EMBEDDING_LLM)",
    )
    parent_parser.add_argument(
        "--embedding-ollama-host",
        type=str,
        default=get_env_default("RAG_EMBEDDING_OLLAMA_HOST", "127.0.0.1"),
        help="Ollama host for embedding (env: RAG_EMBEDDING_OLLAMA_HOST)",
    )
    parent_parser.add_argument(
        "--embedding-ollama-port",
        type=int,
        default=int(get_env_default("RAG_EMBEDDING_OLLAMA_PORT", 11434)),
        help="Ollama port for embedding (env: RAG_EMBEDDING_OLLAMA_PORT)",
    )
    parent_parser.add_argument(
        "--ollama-host",
        type=str,
        default=get_env_default("RAG_OLLAMA_HOST", "127.0.0.1"),
        help="Ollama host for LLM (env: RAG_OLLAMA_HOST)",
    )
    parent_parser.add_argument(
        "--ollama-port",
        type=int,
        default=int(get_env_default("RAG_OLLAMA_PORT", 11434)),
        help="Ollama port for LLM (env: RAG_OLLAMA_PORT)",
    )
    parent_parser.add_argument(
        "--llm",
        type=str,
        default=get_env_default("RAG_LLM", "ollama"),
        choices=["openai", "ollama", "gemini"],
        help="LLM to use for search (env: RAG_LLM)",
    )

    parser = argparse.ArgumentParser(
        description="Run ChromaDB with Ollama Embedding Function",
        parents=[parent_parser],
    )

    subparsers = parser.add_subparsers(dest="subparser")

    # data fill subcommand
    data_subparser = subparsers.add_parser(
        "data-fill", help="Data fill subcommand", parents=[parent_parser]
    )
    data_subparser.add_argument(
        "source_path",
        type=str,
        nargs="+",
        help="Path(s) to the source data file(s) or URL(s). URLs must start with http:// or https://",
    )
    data_subparser.add_argument(
        "--mode",
        type=str,
        default=get_env_default("RAG_MODE", "single"),
        choices=["single", "elements"],
        help="Mode for processing the data: 'single' for single file, 'elements' for multiple elements in a file (env: RAG_MODE)",
    )
    data_subparser.add_argument(
        "--no-insert-into-chroma",
        dest="insert_into_chroma",
        action="store_false",
        default=(get_env_default("RAG_INSERT_INTO_CHROMA", "true") or "true").lower() == "true",
        help="Do not insert data into ChromaDB (env: RAG_INSERT_INTO_CHROMA, default: true)",
    )
    data_subparser.add_argument(
        "--cleanup",
        action="store_true",
        default=(get_env_default("RAG_CLEANUP", "false") or "false").lower() == "true",
        help="Cleanup the collection before filling it",
    )
    data_subparser.add_argument(
        "--bucket-name",
        type=str,
        default=get_env_default("RAG_BUCKET_NAME", ""),
        help="S3 bucket name for uploading markdown files (optional)",
    )
    data_subparser.add_argument(
        "--bucket-path",
        type=str,
        default=get_env_default("RAG_BUCKET_PATH", ""),
        help="S3 bucket path for uploading markdown files (optional)",
    )
    data_subparser.add_argument(
        "--upload-to-s3",
        action="store_true",
        default=(get_env_default("RAG_UPLOAD_TO_S3", "false") or "false").lower() == "true",
        help="Upload parsed markdown to S3 (env: RAG_UPLOAD_TO_S3, default: false)",
    )
    data_subparser.add_argument(
        "--upload-to-open-webui",
        action="store_true",
        default=(get_env_default("RAG_UPLOAD_TO_OPEN_WEBUI", "false") or "false").lower() == "true",
        help="Upload parsed markdown to Open WebUI via API (env: RAG_UPLOAD_TO_OPEN_WEBUI, default: false)",
    )
    data_subparser.add_argument(
        "--open-webui-url",
        type=str,
        default=get_env_default("RAG_OPEN_WEBUI_URL", "http://localhost:3000"),
        help="Open WebUI API base URL (env: RAG_OPEN_WEBUI_URL)",
    )
    data_subparser.add_argument(
        "--open-webui-api-key",
        type=str,
        default=get_env_default("RAG_OPEN_WEBUI_API_KEY", ""),
        help="Open WebUI API key (env: RAG_OPEN_WEBUI_API_KEY)",
    )
    data_subparser.add_argument(
        "--open-webui-knowledge-id",
        type=str,
        default=get_env_default("RAG_OPEN_WEBUI_KNOWLEDGE_ID", ""),
        help="Open WebUI knowledge collection ID (env: RAG_OPEN_WEBUI_KNOWLEDGE_ID, optional)",
    )
    data_subparser.add_argument(
        "--clean-content",
        action="store_true",
        default=((get_env_default("RAG_CLEAN_CONTENT", "false") or "false").lower() == "true"),
        help="Clean document content by removing navigation, ads, and UI clutter before processing (env: RAG_CLEAN_CONTENT)",
    )
    data_subparser.add_argument(
        "--extract-wisdom",
        action="store_true",
        default=((get_env_default("RAG_EXTRACT_WISDOM", "false") or "false").lower() == "true"),
        help="Extract wisdom from content using Fabric (requires Fabric to be installed) (env: RAG_EXTRACT_WISDOM)",
    )
    data_subparser.add_argument(
        "--fabric-command",
        type=str,
        default=get_env_default("RAG_FABRIC_COMMAND", "fabric"),
        help="Fabric command name (e.g., 'fabric' or 'fabric-ai') (env: RAG_FABRIC_COMMAND)",
    )
    data_subparser.add_argument(
        "--fabric-pattern",
        type=str,
        default=get_env_default("RAG_FABRIC_PATTERN", "create_micro_summary"),
        help="Fabric pattern to use for wisdom extraction (default: 'create_micro_summary') (env: RAG_FABRIC_PATTERN)",
    )
    data_subparser.add_argument(
        "--chunk-size",
        type=int,
        default=int(get_env_default("RAG_CHUNK_SIZE", "600")),
        help="Size of text chunks for splitting (env: RAG_CHUNK_SIZE)",
    )
    data_subparser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(get_env_default("RAG_CHUNK_OVERLAP", "200")),
        help="Overlap between chunks (env: RAG_CHUNK_OVERLAP)",
    )
    data_subparser.add_argument(
        "--convert-to-markdown",
        action="store_true",
        default=False,
        help="Convert HTML files to Markdown using the same rules as URL sources (file-based flow only)",
    )

    # search subcommand
    search_parser = subparsers.add_parser(
        "search", help="Search for documents in the collection", parents=[parent_parser]
    )
    search_parser.add_argument(
        "query", type=str, help="Query text to search for in the collection"
    )
    search_parser.add_argument(
        "--model",
        type=str,
        default=get_env_default("RAG_MODEL", "qwen3:8b"),
        help="Model to use for the LLM (env: RAG_MODEL)",
    )
    # model="gpt-4o",
    # model="qwen3:8b",
    # model="deepseek-r1:14b",
    # "gemini-2.5-flash"
    # "gemini-2.5-pro"

    # chat subcommand
    chat_parser = subparsers.add_parser(
        "chat", help="Interactive chat with documents", parents=[parent_parser]
    )
    chat_parser.add_argument(
        "--model",
        type=str,
        default=get_env_default("RAG_MODEL", "qwen3:8b"),
        help="Model to use for the LLM (env: RAG_MODEL)",
    )

    # list-models subcommand
    list_models_parser = subparsers.add_parser(
        "list-models", help="List available models for a specific LLM provider"
    )
    list_models_parser.add_argument(
        "provider",
        type=str,
        choices=["openai", "ollama", "gemini"],
        help="LLM provider to list models for",
    )

    # web subcommand
    web_parser = subparsers.add_parser(
        "web", help="Start web interface for interactive chat", parents=[parent_parser]
    )
    web_parser.add_argument(
        "--model",
        type=str,
        default=get_env_default("RAG_MODEL", "qwen3:8b"),
        help="Model to use for the LLM (env: RAG_MODEL)",
    )
    web_parser.add_argument(
        "--port",
        type=int,
        default=int(get_env_default("RAG_WEB_PORT", 8080)),
        help="Port for web server (env: RAG_WEB_PORT)",
    )
    web_parser.add_argument(
        "--host",
        type=str,
        default=get_env_default("RAG_WEB_HOST", "127.0.0.1"),
        help="Host for web server (env: RAG_WEB_HOST)",
    )
    web_parser.add_argument(
        "--debug",
        action="store_true",
        default=(get_env_default("RAG_WEB_DEBUG", "false") or "false").lower() == "true",
        help="Enable debug mode for web server (env: RAG_WEB_DEBUG)",
    )
    web_parser.add_argument(
        "--browser",
        action="store_true",
        default=False,
        help="Automatically open browser when starting web interface",
    )
    web_parser.add_argument(
        "--cors-origins",
        type=str,
        default=get_env_default("RAG_WEB_CORS_ORIGINS", ""),
        help="Comma-separated list of allowed CORS origins (env: RAG_WEB_CORS_ORIGINS)",
    )
    web_parser.add_argument(
        "--secret-key",
        type=str,
        default=get_env_default("RAG_WEB_SECRET_KEY", "rag-web-secret-key"),
        help="Flask secret key for sessions (env: RAG_WEB_SECRET_KEY)",
    )
    web_parser.add_argument(
        "--max-history",
        type=int,
        default=int(get_env_default("RAG_WEB_MAX_HISTORY", 50)),
        help="Maximum conversation history length (env: RAG_WEB_MAX_HISTORY)",
    )
    web_parser.add_argument(
        "--timeout",
        type=int,
        default=int(get_env_default("RAG_WEB_TIMEOUT", 300)),
        help="Request timeout in seconds (env: RAG_WEB_TIMEOUT)",
    )
    web_parser.add_argument(
        "--workers",
        type=int,
        default=int(get_env_default("RAG_WEB_WORKERS", 1)),
        help="Number of worker processes (env: RAG_WEB_WORKERS)",
    )

    return parser.parse_args()
