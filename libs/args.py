import argparse
import os


def get_env_default(env_var, default=None):
    """Get value from environment variable with fallback to default"""
    return os.getenv(env_var, default)


def parse_arguments():
    """Parse command line arguments with environment variable support
    
    Environment Variables:
    - RAG_COLLECTION: Collection name (default: "RAG")
    - RAG_DB_PATH: Database path (default: "db")
    - RAG_LOG_LEVEL: Log level (default: "INFO")
    - RAG_EMBEDDING_MODEL: Embedding model (default: "nomic-embed-text")
    - RAG_LLM: LLM provider (default: "ollama")
    - RAG_MODEL: LLM model for search (default: "qwen3:8b")
    - RAG_SOURCE_TYPE: Source type for data-fill (default: "file")
    - RAG_MODE: Processing mode for data-fill (default: "single")
    """
    # Create a parent parser for shared arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--collection", type=str, 
                             default=get_env_default("RAG_COLLECTION", "RAG"), 
                             help="Name of the collection to use or create (env: RAG_COLLECTION)")
    parent_parser.add_argument("--db-path", type=str, 
                             default=get_env_default("RAG_DB_PATH", "db"), 
                             help="Path to the ChromaDB database (env: RAG_DB_PATH)")
    parent_parser.add_argument("--dry-run", action='store_true', 
                             help="Run in dry-run mode without making changes")
    parent_parser.add_argument("--log-level", type=str, 
                             default=get_env_default("RAG_LOG_LEVEL", "INFO"), 
                             choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                             help="Log level (env: RAG_LOG_LEVEL)")
    parent_parser.add_argument("--embedding-model", type=str, 
                             default=get_env_default("RAG_EMBEDDING_MODEL", "nomic-embed-text"), 
                             help="Embedding model to use (must be consistent for data-fill and search) (env: RAG_EMBEDDING_MODEL)")
    parent_parser.add_argument("--llm", type=str, 
                             default=get_env_default("RAG_LLM", "ollama"), 
                             choices=["openai", "ollama", "gemini"], 
                             help="LLM to use for embeddings and search (env: RAG_LLM)")

    parser = argparse.ArgumentParser(description="Run ChromaDB with Ollama Embedding Function", parents=[parent_parser])

    subparsers = parser.add_subparsers(dest='subparser')

    # data fill subcommand
    data_subparser = subparsers.add_parser("data-fill", help="Data fill subcommand", parents=[parent_parser])
    data_subparser.add_argument("source_path", type=str, nargs='+', help="Path(s) to the source data file(s) or URL(s)")
    data_subparser.add_argument("--source-type", type=str, 
                             default=get_env_default("RAG_SOURCE_TYPE", "file"), 
                             choices=["file", "url"], 
                             help="Type of the source data: 'file' for local file, 'url' for remote URL (env: RAG_SOURCE_TYPE)")
    data_subparser.add_argument("--mode", type=str, 
                             default=get_env_default("RAG_MODE", "single"), 
                             choices=["single", "elements"], 
                             help="Mode for processing the data: 'single' for single file, 'elements' for multiple elements in a file (env: RAG_MODE)")
    data_subparser.add_argument("--cleanup", action='store_true', 
                             help="Cleanup the collection before filling it")

    # search subcommand
    search_parser = subparsers.add_parser("search", help="Search for documents in the collection", parents=[parent_parser])
    search_parser.add_argument("query", type=str, help="Query text to search for in the collection")
    search_parser.add_argument("--model", type=str, 
                             default=get_env_default("RAG_MODEL", "qwen3:8b"), 
                             help="Model to use for the LLM (env: RAG_MODEL)")
    # model="gpt-4o",
    # model="qwen3:8b",
    # model="deepseek-r1:14b",
    # "gemini-2.5-flash"
    # "gemini-2.5-pro"

    # chat subcommand
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with documents", parents=[parent_parser])
    chat_parser.add_argument("--model", type=str, 
                           default=get_env_default("RAG_MODEL", "qwen3:8b"), 
                           help="Model to use for the LLM (env: RAG_MODEL)")

    return parser.parse_args() 
