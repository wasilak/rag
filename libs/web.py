import os
import logging
import webbrowser
from typing import List, Dict, Any, Optional, Callable, TypeVar, cast
from functools import wraps
from flask import Flask, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from openai import OpenAI
import tiktoken
from engineio.payload import Payload
from chromadb.api import ClientAPI
from .embedding import set_embedding_function
from .utils import format_footnotes
from .models import get_best_model

logger = logging.getLogger("RAG")


class WebChatManager:
    """Manages web chat sessions and state"""

    def __init__(
        self,
        client: ClientAPI,
        collection_name: str,
        llm: str,
        model: str,
        embedding_model: str,
        embedding_llm: str,
        embedding_ollama_host: str,
        embedding_ollama_port: int,
        ollama_host: str,
        ollama_port: int,
        max_history: int = 50,
    ):
        self.client = client
        self.collection_name = collection_name
        self.llm = llm
        # Validate and get best available model
        self.model = get_best_model(
            llm, ollama_host, ollama_port, model, "chat"
        )
        self.embedding_model = embedding_model
        self.embedding_llm = embedding_llm
        self.embedding_ollama_host = embedding_ollama_host
        self.embedding_ollama_port = embedding_ollama_port
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.max_history = max_history

        self.llm_client = self._create_llm_client()
        self.embedding_function = set_embedding_function(
            embedding_llm, embedding_model, embedding_ollama_host, embedding_ollama_port
        )
        self.conversation_history: List[Dict[str, str]] = []
        self.tokenizer = self._get_tokenizer()

    def _create_llm_client(self) -> OpenAI:
        """Create LLM client based on the provider"""
        if self.llm == "ollama":
            logger.debug("Using Ollama as LLM")
            return OpenAI(
                base_url=f"http://{self.ollama_host}:{self.ollama_port}/v1",
                api_key="ollama",  # required, but unused
            )
        elif self.llm == "gemini":
            logger.debug("Using Gemini as LLM")
            return OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            logger.debug("Using OpenAI as LLM")
            return OpenAI()

    def _get_tokenizer(self):
        """Get appropriate tokenizer for the model"""
        try:
            if self.model.startswith("gpt-"):
                return tiktoken.encoding_for_model(self.model)
            elif self.model.startswith("claude-"):
                return tiktoken.get_encoding("cl100k_base")  # Claude uses cl100k_base
            else:
                # Default to GPT-4 tokenizer for other models
                return tiktoken.encoding_for_model("gpt-4")
        except Exception:
            # Fallback to GPT-4 tokenizer
            return tiktoken.encoding_for_model("gpt-4")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4

    def count_conversation_tokens(self) -> Dict[str, int]:
        """Count tokens in current conversation"""
        total_tokens = 0
        user_tokens = 0
        assistant_tokens = 0

        for message in self.conversation_history:
            tokens = self._count_tokens(message["content"])
            total_tokens += tokens
            if message["role"] == "user":
                user_tokens += tokens
            else:
                assistant_tokens += tokens

        return {
            "total": total_tokens,
            "user": user_tokens,
            "assistant": assistant_tokens,
            "messages": len(self.conversation_history),
        }

    def _build_system_prompt(self, results: Dict[str, Any]) -> str:
        """Build system prompt with document context"""
        system_prompt = """
        ### Example
        User: How can a coding buddy help with learning Rust?

        Answer:
        A coding buddy can help you improve by providing feedback through code reviews and engaging in pair programming sessions [1]. They can also help you solidify your understanding of Rust by encouraging you to explain concepts aloud [2].

        Footnotes:
        [1] "Find A Coding Buddy" in "Flattening Rust's Learning Curve | corrode Rust Consulting", `markdown.md`
        [2] "Explain Rust Code To Non-Rust Developers" in "Flattening Rust's Learning Curve | corrode Rust Consulting", `markdown.md`

        ---

        You are a helpful assistant that answers questions strictly based on the provided documents.

        Each document includes:
        - `text`: the paragraph or chunk content
        - `resolved_title`: the section this chunk belongs to
        - `page_title`: the title of the entire markdown document
        - `source`: the filename (e.g., markdown.md)

        When answering:
        - Use only the provided document text — do not invent or guess.
        - Use full sentences and clearly explained reasoning.
        - Every factual statement must be annotated with a footnote reference like [1], [2], etc.
        - Maintain conversation context from previous messages.

        Footnotes must:
        - Be deduplicated
        - Follow the format:
          [1] "{section}" in "{page}", `{source}`

        VERY IMPORTANT:
        - Your response must be nicely formatted in valid Markdown, with footnotes at the end of the answer
        - Use text formatting like bold, italics, and code blocks as needed.
        - Use quotes for direct citations from the documents.
        - Use headings, lists, and other Markdown features to improve readability.

        If the documents do not answer the question, respond with: I don't know.
        """

        footnotes_metadata = []

        # Check if documents and metadatas are not None
        if results["documents"] is not None and results["metadatas"] is not None:
            for i, item in enumerate(results["documents"][0]):
                metadata_entry = results["metadatas"][0][i]
                footnotes_metadata.append(metadata_entry)

                system_prompt += f'{i + 1}. "{item}"\n'
                system_prompt += f"metadata: {metadata_entry}\n\n"

        # Format footnotes from metadata
        footnotes = format_footnotes(footnotes_metadata)
        system_prompt += f"Footnotes:\n{footnotes}\n"

        return system_prompt

    def generate_response_stream(self, user_message: str):
        """Generate streaming response using RAG"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self._trim_conversation_history()

            # Search for relevant documents
            collection = self.client.get_or_create_collection(
                name=self.collection_name, embedding_function=self.embedding_function
            )

            results = collection.query(query_texts=[user_message], n_results=4)

            # Convert ChromaDB query results to dict
            results_dict: Dict[str, Any] = {
                "documents": results["documents"],
                "metadatas": results["metadatas"],
                "ids": results["ids"],
                "distances": results["distances"]
            }

            # Build system prompt with context
            system_prompt = self._build_system_prompt(results_dict)

            # Prepare messages for LLM
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)

            # Generate streaming response
            response = self.llm_client.chat.completions.create(
                model=self.model, messages=messages, stream=True  # type: ignore
            )

            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunk_content = chunk.choices[0].delta.content
                    full_response += chunk_content
                    yield chunk_content

            # Add complete response to conversation history
            if full_response:
                self.conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )
                self._trim_conversation_history()

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}"

    def _trim_conversation_history(self):
        """Trim conversation history to max_history length"""
        if len(self.conversation_history) > self.max_history:
            # Keep the most recent messages, maintaining pairs when possible
            excess = len(self.conversation_history) - self.max_history
            self.conversation_history = self.conversation_history[excess:]
            logger.debug(f"Trimmed conversation history to {len(self.conversation_history)} messages")

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "model": self.model,
            "llm": self.llm,
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_llm": self.embedding_llm,
        }


# Global chat manager instance
chat_manager: Optional[WebChatManager] = None
# Type alias for route functions
RouteFunc = TypeVar('RouteFunc', bound=Callable[..., Any])

def _require_chat_manager(f: RouteFunc) -> RouteFunc:
    """Decorator to ensure chat manager is initialized"""
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        assert chat_manager is not None, "Chat manager must be initialized"
        return f(*args, **kwargs)
    return cast(RouteFunc, wrapper)


def _configure_cors(app: Flask, cors_origins: str, port: int, host: str) -> None:
    """Configure CORS settings for the Flask app"""
    if cors_origins:
        # Use custom CORS origins if provided
        allowed_origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    else:
        # Default CORS origins - allow React dev server (3000) and actual server port
        allowed_origins = [
            "http://localhost:3000", "http://127.0.0.1:3000",  # React dev server
            f"http://localhost:{port}", f"http://127.0.0.1:{port}",  # Local access
            f"http://{host}:{port}",  # Specified host
            "*"  # Allow all origins when binding to 0.0.0.0
        ]

    CORS(app, origins=allowed_origins, supports_credentials=True)

def _initialize_chat_manager(
    client: ClientAPI,
    collection_name: str,
    llm: str,
    model: str,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    ollama_host: str,
    ollama_port: int,
    max_history: int,
) -> None:
    """Initialize the global chat manager"""
    global chat_manager
    chat_manager = WebChatManager(
        client, collection_name, llm, model, embedding_model,
        embedding_llm, embedding_ollama_host, embedding_ollama_port,
        ollama_host, ollama_port, max_history
    )

def setup_routes(app: Flask) -> None:
    """Set up Flask routes"""
    @app.route("/")
    def serve_index() -> Any:
        """Serve React app"""
        if app.static_folder is None:
            raise RuntimeError("Static folder not configured")
        return send_from_directory(app.static_folder, "index.html")

    @app.route("/api/config")
    @_require_chat_manager
    def get_config() -> Any:
        """Get current configuration"""
        assert chat_manager is not None
        return jsonify(chat_manager.get_config())

    @app.route("/api/tokens")
    @_require_chat_manager
    def get_tokens() -> Any:
        """Get token statistics"""
        assert chat_manager is not None
        return jsonify(chat_manager.count_conversation_tokens())

    @app.route("/api/history")
    @_require_chat_manager
    def get_history() -> Any:
        """Get conversation history"""
        assert chat_manager is not None
        return jsonify({"history": chat_manager.conversation_history})

    @app.route("/api/clear", methods=["POST"])
    @_require_chat_manager
    def clear_chat() -> Any:
        """Clear conversation history"""
        assert chat_manager is not None
        chat_manager.clear_conversation()
        return jsonify({"status": "cleared"})

def create_app(
    client: ClientAPI,
    collection_name: str,
    llm: str,
    model: str,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    ollama_host: str,
    ollama_port: int,
    port: int = 8080,
    host: str = "127.0.0.1",
    cors_origins: str = "",
    secret_key: str = "rag-web-secret-key",
    max_history: int = 50,
    timeout: int = 300,
    debug: bool = False,
) -> Flask:
    """Create and configure Flask app"""
    app = Flask(__name__, static_folder="../web/build", static_url_path="")
    app.config["SECRET_KEY"] = secret_key
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0  # Disable caching in development
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max request size
    app.config["PROPAGATE_EXCEPTIONS"] = True
    if debug:
        app.config["DEBUG"] = True  # Enable Flask debug mode when requested

    # Configure CORS
    _configure_cors(app, cors_origins, port, host)

    # Initialize SocketIO with permissive CORS for 0.0.0.0 binding
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        ping_timeout=timeout,
        ping_interval=25,
        max_http_buffer_size=16 * 1024 * 1024,  # 16MB max WebSocket message size
        always_connect=True,
        logger=True if debug else False,  # Only enable socket logging in debug mode
        engineio_logger=True if debug else False,  # Only enable engine logging in debug mode
        manage_session=False  # Disable session management for development
    )

    # Initialize chat manager
    _initialize_chat_manager(
        client, collection_name, llm, model, embedding_model,
        embedding_llm, embedding_ollama_host, embedding_ollama_port,
        ollama_host, ollama_port, max_history
    )

    # Set up routes
    setup_routes(app)

    @socketio.on("send_message")
    def handle_message(data):
        """Handle incoming chat message with streaming response"""
        user_message = data.get("message", "").strip()
        if not user_message:
            return

        # Emit that we're processing
        emit("message_status", {"status": "processing"})

        # Generate streaming response
        try:
            assert chat_manager is not None, "Chat manager must be initialized"
            for chunk in chat_manager.generate_response_stream(user_message):
                emit("message_chunk", {"chunk": chunk})

            # Emit completion with token stats
            token_stats = chat_manager.count_conversation_tokens()
            emit("message_complete", {
                "status": "complete",
                "tokens": token_stats
            })
        except Exception as e:
            logger.error(f"Error in message handling: {e}")
            emit("message_error", {"error": str(e)})

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection"""
        logger.info("Client connected to WebSocket")
        emit("connected", {"status": "connected"})

    @socketio.on("disconnect")
    def handle_disconnect():
        """Handle client disconnection"""
        logger.info("Client disconnected from WebSocket")

    @socketio.on_error()
    def handle_error(e):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {str(e)}")
        emit("error", {"error": str(e)})

    @socketio.on("error")
    def handle_client_error(error):
        """Handle client-side WebSocket errors"""
        logger.error(f"Client WebSocket error: {error}")

    # Store socketio instance in app for access
    setattr(app, 'socketio', socketio)

    return app


def process_web(
    client: ClientAPI,
    collection: str,
    llm: str,
    model: str,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    ollama_host: str,
    ollama_port: int,
    port: int = 8080,
    host: str = "127.0.0.1",
    debug: bool = False,
    browser: bool = False,
    cors_origins: str = "",
    secret_key: str = "rag-web-secret-key",
    max_history: int = 50,
    timeout: int = 300,
    workers: int = 1,
) -> None:
    """Process web command"""
    # Check if web interface is available
    import os
    web_build_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'build')
    if not os.path.exists(web_build_path):
        logger.warning("Web interface build not found. Please ensure the web interface is built.")
        logger.info("Web interface will not be available")
        return

    logger.info(f"Starting web interface for collection '{collection}'")

    app = create_app(
        client, collection, llm, model, embedding_model,
        embedding_llm, embedding_ollama_host, embedding_ollama_port,
        ollama_host, ollama_port, port, host, cors_origins, secret_key, max_history, timeout,
        debug
    )

    # Start the server
    logger.info(f"Web interface starting at http://{host}:{port}")

    # Open browser if requested
    if browser:
        # Always use localhost for browser, regardless of host binding
        url = f"http://localhost:{port}"
        logger.debug(f"Opening browser to {url}")
        webbrowser.open(url)

    # Configure Engine.IO for development
    Payload.max_decode_packets = 1000  # Increased for development to help with debugging

    # Run the app with SocketIO
    if hasattr(app, 'socketio'):
        getattr(app, 'socketio').run(
            app,
            host=host,
            port=port,
            debug=True if debug else False,
            allow_unsafe_werkzeug=True,
            log_output=True if debug else False  # Only show logs in debug mode
        )
    else:
        raise RuntimeError("SocketIO not properly initialized")
