import os
import logging
import webbrowser
from typing import List, Dict, Any, Optional, Callable, TypeVar, cast, Generator
from functools import wraps
from flask import Flask, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from openai import OpenAI
import tiktoken
from engineio.payload import Payload
from chromadb.api import ClientAPI
from .commands.data_fill.embedding import set_embedding_function
from .utils import format_footnotes
from .models import get_best_model
from .chat_storage import ChatStorage

logger = logging.getLogger("RAG")

# Per-session chat state (sid -> {"current_chat_id": str|None, "conversation_history": list})
chat_sessions = {}


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
        chat_db_path: Optional[str] = None,
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
        self.tokenizer = self._get_tokenizer()
        self.storage = ChatStorage(chat_db_path) if chat_db_path else None

        # Remove global current_chat_id and conversation_history
        # Use per-session state in handlers below

        # Auto-load latest chat from storage if available (for legacy, not used in web)
        # (No-op for per-session)

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
        - `sanitized_title`: the section this chunk belongs to
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

    def generate_response_stream(self, user_message: str) -> Generator[str, None, Optional[str]]:
        """Generate streaming response using RAG. Returns new_chat_id if created."""
        new_chat_id = None
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            if self.storage:
                new_chat_id = self._save_message("user", user_message)

            # Search for relevant documents
            if not self.client:
                raise ValueError("ChromaDB client not initialized")

            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
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
                self._save_message("assistant", full_response)
                self._trim_conversation_history()

            return new_chat_id

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"Error: {str(e)}"
            return None

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
        if self.storage and self.current_chat_id:
            self.current_chat_id = None

    def _load_chat(self, chat_id: str) -> bool:
        """Load chat from storage"""
        if not self.storage:
            return False

        chat = self.storage.get_chat(chat_id)
        if not chat:
            return False

        self.conversation_history = [
            {"role": msg.role, "content": msg.content}
            for msg in chat.messages
        ]
        self.current_chat_id = chat_id
        return True

    def list_chats(self) -> List[Dict[str, Any]]:
        """List all available chats"""
        if not self.storage:
            return []

        chats = self.storage.list_chats()
        return [{
            "id": chat.id,
            "title": chat.title,
            "created_at": chat.created_at.isoformat(),
            "last_updated": chat.last_updated.isoformat()
        } for chat in chats]

    def _save_message(self, role: str, content: str) -> Optional[str]:
        """Save message to storage if enabled. Returns new chat_id if created."""
        if not self.storage:
            return None

        new_chat_id = None
        if not self.current_chat_id:
            # This shouldn't happen with the new approach, but keep for safety
            trimmed = content.strip().replace('\n', ' ')
            if len(trimmed) > 100:
                title = trimmed[:100].rstrip() + "..."
            else:
                title = trimmed
            self.current_chat_id = self.storage.create_chat(title)
            new_chat_id = self.current_chat_id
        else:
            # Update chat title with first user message if it's still "New Chat"
            if role == "user":
                chat = self.storage.get_chat(self.current_chat_id)
                if chat and chat.title == "New Chat":
                    trimmed = content.strip().replace('\n', ' ')
                    if len(trimmed) > 100:
                        title = trimmed[:100].rstrip() + "..."
                    else:
                        title = trimmed
                    self.storage.update_chat_title(self.current_chat_id, title)

        self.storage.add_message(
            self.current_chat_id,
            role,
            content
        )
        return new_chat_id

    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return {
            "model": self.model,
            "llm": self.llm,
            "collection": self.collection_name,
            "embedding_model": self.embedding_model,
            "embedding_llm": self.embedding_llm,
        }

    def _generate_chat_summary(self, messages: List[Any]) -> str:
        """Generate a summary of the chat conversation using LLM"""
        if not messages:
            return "No messages to summarize."

        # Build conversation text for summarization
        conversation_text = ""
        for msg in messages:
            role = "User" if msg.role == "user" else "Assistant"
            conversation_text += f"{role}: {msg.content}\n\n"

        # Create prompt for summarization
        summary_prompt = f"""Please provide a comprehensive summary of the following conversation. The summary should capture the key points, decisions made, and important information discussed. Make it concise but informative.

Conversation:
{conversation_text}

Summary:"""

        try:
            # Generate summary using LLM
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise but comprehensive summaries of conversations."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            summary = response.choices[0].message.content
            summary = summary.strip() if summary else None
            return summary if summary else "Unable to generate summary."

        except Exception as e:
            logger.error(f"Error generating chat summary: {e}")
            return f"Error generating summary: {str(e)}"


# Global chat manager instance
chat_manager: Optional[WebChatManager] = None
# Type alias for route functions
RouteFunc = TypeVar('RouteFunc', bound=Callable[..., Any])


def _require_chat_manager(f: RouteFunc) -> RouteFunc:
    """Decorator to ensure chat manager is initialized"""
    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if chat_manager is None:
            from flask import jsonify
            return jsonify({"error": "Chat manager not initialized"}), 500
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
    chat_db_path: Optional[str] = None,
) -> None:
    """Initialize the global chat manager"""
    global chat_manager
    chat_manager = WebChatManager(
        client, collection_name, llm, model, embedding_model,
        embedding_llm, embedding_ollama_host, embedding_ollama_port,
        ollama_host, ollama_port, max_history, chat_db_path
    )


def setup_routes(app: Flask) -> None:  # noqa: C901
    """Set up Flask routes"""

    @app.route("/api/chats", methods=["GET"])
    @_require_chat_manager
    def list_chats():
        """List all available chats"""
        return jsonify(chat_manager.list_chats())  # type: ignore[attr-defined]

    @app.route("/api/chats/<chat_id>", methods=["GET"])
    @_require_chat_manager
    def load_chat(chat_id: str):
        """Load specific chat"""
        if chat_manager._load_chat(chat_id):  # type: ignore[attr-defined]
            return jsonify({"success": True})
        return jsonify({"success": False}), 404

    @app.route("/api/chats", methods=["POST"])
    @_require_chat_manager
    def create_chat():
        """Create a new chat"""
        if not chat_manager.storage:  # type: ignore[attr-defined]
            return jsonify({"success": False, "error": "Storage not available"}), 500

        # Create a new chat with a placeholder title
        chat_id = chat_manager.storage.create_chat("New Chat")  # type: ignore[attr-defined]
        chat = chat_manager.storage.get_chat(chat_id)  # type: ignore[attr-defined]
        if chat:
            return jsonify({
                "id": chat.id,
                "title": chat.title,
                "created_at": chat.created_at.isoformat(),
                "last_updated": chat.last_updated.isoformat()
            })
        else:
            return jsonify({"success": False, "error": "Failed to create chat"}), 500

    @app.route("/api/chats/<chat_id>", methods=["DELETE"])
    @_require_chat_manager
    def delete_chat(chat_id: str):
        """Delete a chat and its messages"""
        if chat_manager.storage and chat_manager.storage.delete_chat(chat_id):  # type: ignore[attr-defined]
            return jsonify({"success": True})
        return jsonify({"success": False}), 404

    @app.route("/api/config")
    @_require_chat_manager
    def get_config() -> Any:
        """Get current configuration"""
        return jsonify(chat_manager.get_config())  # type: ignore[attr-defined]

    @app.route("/api/tokens")
    @_require_chat_manager
    def get_tokens() -> Any:
        chat_id = request.args.get("chat_id")
        if not chat_id or not chat_manager.storage:  # type: ignore[attr-defined]
            return jsonify({"total": 0, "user": 0, "assistant": 0, "messages": 0})
        chat = chat_manager.storage.get_chat(chat_id)  # type: ignore[attr-defined]
        if not chat:
            return jsonify({"total": 0, "user": 0, "assistant": 0, "messages": 0})
        # Count tokens
        total_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        for msg in chat.messages:
            tokens = chat_manager._count_tokens(msg.content)  # type: ignore[attr-defined]
            total_tokens += tokens
            if msg.role == "user":
                user_tokens += tokens
            else:
                assistant_tokens += tokens
        return jsonify({
            "total": total_tokens,
            "user": user_tokens,
            "assistant": assistant_tokens,
            "messages": len(chat.messages)
        })

    @app.route("/api/history")
    @_require_chat_manager
    def get_history() -> Any:
        chat_id = request.args.get("chat_id")
        if not chat_id or not chat_manager.storage:  # type: ignore[attr-defined]
            return jsonify({"history": []})
        chat = chat_manager.storage.get_chat(chat_id)  # type: ignore[attr-defined]
        if not chat:
            return jsonify({"history": []})
        return jsonify({"history": [
            {"role": msg.role, "content": msg.content}
            for msg in chat.messages
        ]})

    @app.route("/api/clear", methods=["POST"])
    @_require_chat_manager
    def clear_chat() -> Any:
        """Clear conversation history"""
        chat_manager.clear_conversation()  # type: ignore[attr-defined]
        return jsonify({"status": "cleared"})

    @app.route("/api/chats/<chat_id>/summarize", methods=["POST"])
    @_require_chat_manager
    def summarize_chat(chat_id: str) -> Any:
        """Summarize and compact a chat conversation"""
        if not chat_manager.storage:  # type: ignore[attr-defined]
            return jsonify({"success": False, "error": "Storage not available"}), 500

        chat = chat_manager.storage.get_chat(chat_id)  # type: ignore[attr-defined]
        if not chat:
            return jsonify({"success": False, "error": "Chat not found"}), 404

        if len(chat.messages) < 2:
            return jsonify({"success": False, "error": "Not enough messages to summarize"}), 400

        try:
            # Generate summary using LLM
            summary = chat_manager._generate_chat_summary(chat.messages)  # type: ignore[attr-defined]

            # Replace all messages with the summary
            success = chat_manager.storage.replace_with_summary(chat_id, summary)  # type: ignore[attr-defined]

            if not success:
                return jsonify({"success": False, "error": "Failed to replace messages with summary"}), 500

            return jsonify({
                "success": True,
                "history": [{"role": "assistant", "content": summary}]
            })
        except Exception as e:
            logger.error(f"Error summarizing chat: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    # React app routes - must come after API routes
    @app.route("/")
    def serve_index() -> Any:
        """Serve React app"""
        # index.html is in the parent directory of the static folder
        if app.static_folder is None:
            raise RuntimeError("Static folder not configured")
        index_folder = os.path.dirname(app.static_folder)
        print(f"Serving index.html from: {index_folder}")
        return send_from_directory(index_folder, "index.html")

    @app.route("/<path:path>")
    def serve_react_app(path: str) -> Any:
        """Serve React app for all other routes (client-side routing)"""
        # index.html is in the parent directory of the static folder
        if app.static_folder is None:
            raise RuntimeError("Static folder not configured")
        index_folder = os.path.dirname(app.static_folder)
        return send_from_directory(index_folder, "index.html")


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
    chat_db_path: Optional[str] = None,
) -> Flask:  # noqa: C901
    """Create and configure Flask app"""
    import os
    # Get the absolute path to the static folder relative to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    static_folder_path = os.path.join(project_root, "web", "build", "static")
    print(f"Static folder path: {static_folder_path}")
    print(f"Static folder exists: {os.path.exists(static_folder_path)}")
    app = Flask(__name__, static_folder=static_folder_path, static_url_path="/static")
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
        client=client,
        collection_name=collection_name,
        llm=llm,
        model=model,
        embedding_model=embedding_model,
        embedding_llm=embedding_llm,
        embedding_ollama_host=embedding_ollama_host,
        embedding_ollama_port=embedding_ollama_port,
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        max_history=max_history,
        chat_db_path=chat_db_path
    )

    # Set up routes
    setup_routes(app)

    @socketio.on("reset_chat")
    def handle_reset_chat():
        from flask import request  # type: ignore[import]
        sid = request.sid  # type: ignore[attr-defined]
        if sid not in chat_sessions:
            chat_sessions[sid] = {"current_chat_id": None, "conversation_history": []}
        else:
            chat_sessions[sid]["current_chat_id"] = None
            chat_sessions[sid]["conversation_history"] = []
        emit("chat_reset", {"status": "reset"})

    @socketio.on("switch_chat")
    def handle_switch_chat(data):
        """Switch to a different chat for current session"""
        from flask import request  # type: ignore[import]
        sid = request.sid  # type: ignore[attr-defined]
        chat_id = data.get("chat_id")

        if not chat_id:
            return

        # Load the chat into the session
        if chat_manager and chat_manager._load_chat(chat_id):
            chat_sessions[sid] = {
                "current_chat_id": chat_manager.current_chat_id,
                "conversation_history": chat_manager.conversation_history
            }
            logger.info(f"Switched session {sid} to chat {chat_id}")
        else:
            logger.error(f"Failed to load chat {chat_id} for session {sid}")

    @socketio.on("send_message")
    def handle_message(data):
        """Handle incoming chat message with streaming response"""
        from flask import request  # type: ignore[import]
        sid = request.sid  # type: ignore[attr-defined]
        if sid not in chat_sessions:
            chat_sessions[sid] = {"current_chat_id": None, "conversation_history": []}

        user_message = data.get("message", "").strip()
        if not user_message:
            return

        # Emit that we're processing
        emit("message_status", {"status": "processing"})

        # Generate streaming response
        try:
            if chat_manager is None:
                emit("message_error", {"error": "Chat manager not initialized"})
                return

            # Patch chat_manager to use per-session state
            chat_manager.current_chat_id = chat_sessions[sid]["current_chat_id"]
            chat_manager.conversation_history = chat_sessions[sid]["conversation_history"]

            stream = chat_manager.generate_response_stream(user_message)
            new_chat_id = None
            for chunk in stream:
                emit("message_chunk", {"chunk": chunk})
            # After streaming is done, get the new_chat_id if any (from generator return value)
            try:
                stream.send(None)
            except StopIteration as e:
                if hasattr(e, 'value'):
                    new_chat_id = e.value

            # Save back per-session state
            chat_sessions[sid]["current_chat_id"] = chat_manager.current_chat_id
            chat_sessions[sid]["conversation_history"] = chat_manager.conversation_history

            token_stats = chat_manager.count_conversation_tokens()
            emit("message_complete", {
                "status": "complete",
                "tokens": token_stats,
                "newChatId": new_chat_id
            })

            # If we updated a chat title, emit a refresh signal
            if new_chat_id is None and chat_manager.current_chat_id:
                # Check if the current chat title was updated
                chat = chat_manager.storage.get_chat(chat_manager.current_chat_id) if chat_manager.storage else None
                if chat and chat.title != "New Chat":
                    emit("chat_title_updated", {"chatId": chat_manager.current_chat_id, "title": chat.title})
        except Exception as e:
            logger.error(f"Error in message handling: {e}")
            emit("message_error", {"error": str(e)})

    @socketio.on("connect")
    def handle_connect():
        """Handle client connection"""
        logger.info("Client connected to WebSocket")
        emit("connected", {"status": "connected"})

    @socketio.on("disconnect")
    def handle_disconnect(_unused=None):
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
    collection_name: str,
    llm: str,
    model: str,
    embedding_model: str,
    embedding_llm: str,
    embedding_ollama_host: str,
    embedding_ollama_port: int,
    ollama_host: str,
    ollama_port: int,
    host: str = "127.0.0.1",
    port: int = 8080,
    debug: bool = False,
    browser: bool = False,
    cors_origins: str = "",
    secret_key: str = "rag-web-secret-key",
    max_history: int = 50,
    timeout: int = 300,
    workers: int = 1,
    chat_db_path: Optional[str] = None,
) -> None:
    """Process web command"""
    # Check if web interface is available
    import os
    web_build_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'build')
    if not os.path.exists(web_build_path):
        logger.warning("Web interface build not found. Please ensure the web interface is built.")
        logger.info("Web interface will not be available")
        return

    logger.info(f"Starting web interface for collection '{collection_name}'")

    app = create_app(
        client=client,
        collection_name=collection_name,
        llm=llm,
        model=model,
        embedding_model=embedding_model,
        embedding_llm=embedding_llm,
        embedding_ollama_host=embedding_ollama_host,
        embedding_ollama_port=embedding_ollama_port,
        ollama_host=ollama_host,
        ollama_port=ollama_port,
        port=port,
        host=host,
        cors_origins=cors_origins,
        secret_key=secret_key,
        max_history=max_history,
        timeout=timeout,
        debug=debug,
        chat_db_path=chat_db_path
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
