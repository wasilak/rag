import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Horizontal, ScrollableContainer
from textual.widgets import Header, Footer, Static, Button, Label, TextArea
from textual import work
from textual.binding import Binding
from openai import OpenAI
import tiktoken
from .embedding import set_embedding_function
from .utils import format_footnotes
from .models import get_best_model
from chromadb.api import ClientAPI

logger = logging.getLogger("RAG")


class ChatMessage(Static):
    """A widget to display a chat message"""

    DEFAULT_CSS = """
    ChatMessage {
        border: none;
        padding: 0;
    }
    """

    def __init__(self, content: str, is_user: bool = True, model_name: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.is_user = is_user
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")


    def on_mount(self) -> None:
        """Set up the message display"""
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.style import Style

        # Create header with timestamp
        user_info = "👤 User" if self.is_user else f"🤖 {self.model_name}" if self.model_name else "🤖 Bot"
        header = f"{self.timestamp} {user_info}"

        # Create markdown rendered content
        style = Style(color="white")
        markdown_content = Markdown(
            self.content,
            code_theme="monokai",
            style=style
        )

        # Combine header with markdown content in a panel
        panel = Panel(
            markdown_content,
            title=header,
            border_style="bright_blue" if self.is_user else "green",
            padding=(0, 1)
        )
        self.update(panel)


class ChatHistory(ScrollableContainer):
    """Scrollable container for chat messages"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.can_focus = True  # Allow focus for scrolling

    def add_message(self, content: str, is_user: bool = True, model_name: str = "") -> None:
        """Add a new message to the chat history"""
        message = ChatMessage(content, is_user, model_name)
        self.mount(message)
        # Schedule scroll to end after the message is rendered
        self.call_after_refresh(self.scroll_end)


class ChatApp(App):
    """Main chat application"""

    # Force dark mode and ensure proper rendering
    CSS_PATH = None  # Use inline CSS
    TITLE = "RAG Chat"

    # Add key bindings for scrolling
    BINDINGS = [
        Binding("up", "scroll_up", "Scroll up", show=False),
        Binding("down", "scroll_down", "Scroll down", show=False),
        Binding("pageup", "page_up", "Page up", show=False),
        Binding("pagedown", "page_down", "Page down", show=False),
        Binding("home", "scroll_home", "Scroll to top", show=False),
        Binding("end", "scroll_end", "Scroll to bottom", show=False),
        Binding("ctrl+c", "exit", "Exit"),
        Binding("ctrl+l", "clear_chat", "Clear chat"),
        Binding("ctrl+i", "show_info", "Show info"),
    ]

    CSS = """
    #chat-history {
        height: 1fr;
        border: solid #7aa2f7;
        padding: 1;
        margin: 1;
        scrollbar-size-horizontal: 1;
        scrollbar-size-vertical: 1;
        overflow-y: auto;
    }

    #chat-history:focus {
        border: solid #bb9af7;
        background: $surface-lighten-1;
    }

    #status {
        background: transparent;
        color: $text;
        padding: 0 1;
        margin: 1;
        border: solid #7aa2f7;
        text-align: center;
    }

    #token-info {
        background: transparent;
        color: $text-muted;
        padding: 0 1;
        margin: 0 1;
        text-align: center;
        height: 1;
    }

    ChatMessage {
        margin: 1;
        padding: 1;
    }

    .user-message {
        background: transparent;
        border: dashed #7aa2f7;
        color: $text;
    }

    .bot-message {
        background: $surface;
        border: dashed #7aa2f7;
        color: $text;
    }

    #input-container {
        height: 10;
        border-top: solid #7aa2f7;
        padding: 0 1;
        background: $surface;
        margin-bottom: 2;
    }

    #message-input {
        height: 8;
        min-height: 8;
        max-height: 8;
    }

    #send-button {
        height: 8;
        min-height: 8;
        max-height: 8;
        margin-left: 1;
    }
    """

    def __init__(
      self,
      client,
      collection_name: str,
      llm: str,
      model: str,
      embedding_model: str,
      embedding_llm: str,
      embedding_ollama_host: str,
      embedding_ollama_port: int,
    ):
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.llm = llm
        # Validate and get best available model
        self.model = get_best_model(llm, embedding_ollama_host, embedding_ollama_port, model, "chat")
        self.embedding_model = embedding_model
        self.embedding_llm = embedding_llm
        self.llm_client = None
        self.embedding_function = None
        self.conversation_history: List[Dict[str, str]] = []
        self.tokenizer = self._get_tokenizer()
        self.total_tokens_used = 0
        self.embedding_ollama_host = embedding_ollama_host
        self.embedding_ollama_port = embedding_ollama_port

    def action_scroll_up(self) -> None:
        """Scroll chat history up"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_up()

    def action_scroll_down(self) -> None:
        """Scroll chat history down"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_down()

    def action_page_up(self) -> None:
        """Page up in chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_page_up()

    def action_page_down(self) -> None:
        """Page down in chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_page_down()

    def action_scroll_home(self) -> None:
        """Scroll to top of chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_home()

    def action_scroll_end(self) -> None:
        """Scroll to bottom of chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.scroll_end()

    def action_clear_chat(self) -> None:
        """Clear the chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.remove_children()
        self.conversation_history.clear()
        self.total_tokens_used = 0
        self._update_token_display()
        self.query_one("#status", Static).update(f"Chat cleared. Ready to chat with {self.model}!")

    def action_show_info(self) -> None:
        """Show chat information"""
        token_counts = self._count_conversation_tokens()
        info = f"Model: {self.model}\nLLM: {self.llm}\nCollection: {self.collection_name}\nMessages: {len(self.conversation_history)}\n\nToken Usage:\nTotal: {token_counts['total']}\nUser: {token_counts['user']}\nAssistant: {token_counts['assistant']}"
        self.notify(info, title="Chat Info")

    def action_exit_chat(self) -> None:
        """Exit the chat application"""
        self.exit()

    def action_tokyo_night(self) -> None:
        """Switch to Tokyo Night theme"""
        self.dark = True
        self.notify("Switched to Tokyo Night theme", title="Theme Changed")

    def action_light_theme(self) -> None:
        """Switch to light theme"""
        self.dark = False
        self.notify("Switched to light theme", title="Theme Changed")

    def action_switch_theme(self, theme_name: str) -> None:
        """Switch to a specific theme"""
        if theme_name.lower() in ["dark", "tokyo", "tokyo-night"]:
            self.dark = True
            self.notify(f"Switched to {theme_name} theme", title="Theme Changed")
        elif theme_name.lower() in ["light", "default"]:
            self.dark = False
            self.notify(f"Switched to {theme_name} theme", title="Theme Changed")

    def action_list_themes(self) -> None:
        """List available themes"""
        themes = ["dark", "light", "default"]
        theme_list = "\n".join(themes)
        self.notify(f"Available themes:\n{theme_list}", title="Available Themes")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        yield Header()
        yield Label("", id="status")
        yield ChatHistory(id="chat-history")
        yield Label("", id="token-info")
        with Horizontal(id="input-container"):
            yield TextArea(id="message-input")
            yield Button("Send", id="send-button")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the application when it starts"""
        # Set Tokyo Night theme as default
        self.theme = "tokyo-night"

        self.llm_client = self._create_llm_client()
        self.embedding_function = set_embedding_function(self.embedding_llm, self.embedding_model, self.embedding_ollama_host, self.embedding_ollama_port)

        # Set text content after mounting
        self.query_one("#status", Static).update(f"🤖 Model: {self.model} | 📚 Collection: {self.collection_name} | 💬 Ready to chat! Press Ctrl+Enter to send")

        # Initialize token display
        self._update_token_display()

        # Focus on input field
        text_area = self.query_one("#message-input")
        text_area.focus()

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

    def _count_conversation_tokens(self) -> Dict[str, int]:
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
            "assistant": assistant_tokens
        }

    def _update_token_display(self):
        """Update the token info display"""
        token_counts = self._count_conversation_tokens()
        token_info = f"💬 Messages: {len(self.conversation_history)} | 🎯 Total Tokens: {token_counts['total']} | 👤 User: {token_counts['user']} | 🤖 Assistant: {token_counts['assistant']}"
        self.query_one("#token-info", Label).update(token_info)

    def _create_llm_client(self) -> OpenAI:
        """Create LLM client based on the provider"""
        if self.llm == "ollama":
            logger.debug("Using Ollama as LLM")
            return OpenAI(
                base_url=f'http://{self.embedding_ollama_host}:{self.embedding_ollama_port}/v1',
                api_key='ollama',  # required, but unused
            )
        elif self.llm == "gemini":
            logger.debug("Using Gemini as LLM")
            return OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            logger.debug("Using OpenAI as LLM")
            return OpenAI()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes"""
        # Keep fixed height for input area
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        # Handle send button
        if event.button.id == "send-button":
            text_area = self.query_one("#message-input", TextArea)
            message = text_area.text.strip()
            if message:
                self._process_message(message)
                # Consume this event
                event.stop()

    def on_key(self, event) -> None:
        """Handle key events"""
        if event.key == "ctrl+c":
            self.exit()
        elif event.key == "cmd+p":
            self.action_command_palette()
        elif event.key == "ctrl+enter":
            # For TextArea, we need to handle Ctrl+Enter manually
            text_area = self.query_one("#message-input", TextArea)
            message = text_area.text.strip()
            if message:
                self._process_message(message)
                # Prevent the event from propagating to the TextArea
                event.stop()
        # Let other keys be handled by the bindings (up, down, pageup, pagedown, etc.)

    def _process_message(self, message: str) -> None:
        """Process a user message"""
        # Clear input
        text_area = self.query_one("#message-input", TextArea)
        text_area.text = ""

        # Add user message to chat
        chat_history = self.query_one("#chat-history", ChatHistory)
        chat_history.add_message(message, is_user=True)

        # Update token display
        self._update_token_display()

        # Update status
        self.query_one("#status", Static).update(f"🤔 Thinking with {self.model}...")

        # Process the message asynchronously
        self._generate_response(message)

    @work(thread=True)
    def _generate_response(self, user_message: str) -> None:
        """Generate response using RAG"""
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})

            # Search for relevant documents
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )

            results = collection.query(
                query_texts=[user_message],
                n_results=4
            )

            # Build system prompt with context
            system_prompt = self._build_system_prompt(results)

            # Prepare messages for LLM
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(self.conversation_history)

            # Check if llm_client is None
            if self.llm_client is None:
                self.call_from_thread(self._update_chat_with_response, "Error: LLM client not initialized.")
                return

            # Generate response
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages  # type: ignore
            )

            if response.choices:
                assistant_message = response.choices[0].message.content

                # Ensure assistant_message is not None
                if assistant_message is not None:
                    # Add to conversation history
                    self.conversation_history.append({"role": "assistant", "content": assistant_message})

                    # Update UI in main thread
                    self.call_from_thread(self._update_chat_with_response, assistant_message)
                else:
                    self.call_from_thread(self._update_chat_with_response, "Sorry, I received an empty response.")
            else:
                self.call_from_thread(self._update_chat_with_response, "Sorry, I couldn't generate a response.")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self.call_from_thread(self._update_chat_with_response, f"Error: {str(e)}")

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

                system_prompt += f"{i + 1}. \"{item}\"\n"
                system_prompt += f"metadata: {metadata_entry}\n\n"

        # Format footnotes from metadata
        footnotes = format_footnotes(footnotes_metadata)
        system_prompt += f"Footnotes:\n{footnotes}\n"

        return system_prompt

    def _update_chat_with_response(self, response: str) -> None:
        """Update chat with assistant response"""
        chat_history = self.query_one("#chat-history", ChatHistory)
        chat_history.add_message(response, is_user=False, model_name=self.model)

        # Update token display
        self._update_token_display()

        self.query_one("#status", Static).update(f"🤖 Model: {self.model} | 📚 Collection: {self.collection_name} | 💬 Ready to chat! Press Ctrl+Enter to send")


def process_chat(
      client: ClientAPI,
      collection: str,
      llm: str,
      model: str,
      embedding_model: str,
      embedding_llm: str,
      embedding_ollama_host: str,
      embedding_ollama_port: int
    ) -> None:
    """Process chat operation"""
    # Disable logging during chat to prevent UI interference
    logging.getLogger().handlers.clear()
    logging.getLogger().setLevel(logging.CRITICAL)

    logger.debug(f"Starting chat interface for collection '{collection}'")

    app = ChatApp(client, collection, llm, model, embedding_model, embedding_llm, embedding_ollama_host, embedding_ollama_port)
    app.run()

    logger.debug("Chat session ended")
