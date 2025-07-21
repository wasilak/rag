import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Input, Static, Button, Label, TextArea
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from openai import OpenAI
from .embedding import set_embedding_function
from .utils import format_footnotes

logger = logging.getLogger("RAG")


class ChatMessage(Static):
    """A widget to display a chat message"""
    
    def __init__(self, content: str, is_user: bool = True, model_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.content = content
        self.is_user = is_user
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Add CSS class based on message type
        if self.is_user:
            self.add_class("user-message")
        else:
            self.add_class("bot-message")
        
    def on_mount(self) -> None:
        """Set up the message display"""
        # Create header with timestamp
        user_info = "User" if self.is_user else f"Bot ({self.model_name})" if self.model_name else "Bot"
        header = f"[{self.timestamp}] {user_info}"
        
        # For now, just display the content as plain text to avoid formatting issues
        # TODO: Implement better Rich formatting that works in terminal
        message_content = f"{header}\n\n{self.content}"
        self.update(message_content)


class ChatHistory(Vertical):
    """Container for chat messages"""
    
    def add_message(self, content: str, is_user: bool = True, model_name: str = "") -> None:
        """Add a new message to the chat history"""
        message = ChatMessage(content, is_user, model_name)
        self.mount(message)
        self.scroll_end()





class ChatApp(App):
    """Main chat application"""
    
    # Force dark mode and ensure proper rendering
    CSS_PATH = None  # Use inline CSS
    TITLE = "RAG Chat"
    
    CSS = """
    #status {
        background: transparent;
        color: $text;
        padding: 0 1;
        margin: 1;
        border: solid $accent;
        text-align: center;
    }
    
    ChatMessage {
        margin: 1;
        padding: 1;
    }
    
    .user-message {
        background: transparent;
        border: dashed $primary-lighten-1;
        color: $text;
    }
    
    .bot-message {
        background: $surface;
        border: dashed $secondary;
        color: $text;
    }
    
    #input-label {
        color: $text;
        text-align: center;
        margin: 1;
    }
    
    .auto-grow {
        min-height: 3;
        max-height: 10;
    }
    """
    
    def __init__(self, client, collection_name: str, llm: str, model: str, embedding_model: str):
        super().__init__()
        self.client = client
        self.collection_name = collection_name
        self.llm = llm
        self.model = model
        self.embedding_model = embedding_model
        self.llm_client = None
        self.embedding_function = None
        self.conversation_history: List[Dict[str, str]] = []
        
    def action_clear_chat(self) -> None:
        """Clear the chat history"""
        chat_history = self.query_one("#chat-history")
        chat_history.remove_children()
        self.conversation_history.clear()
        self.query_one("#status").update(f"Chat cleared. Ready to chat with {self.model}!")
        
    def action_show_info(self) -> None:
        """Show chat information"""
        info = f"Model: {self.model}\nLLM: {self.llm}\nCollection: {self.collection_name}\nMessages: {len(self.conversation_history)}"
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
        yield Label("", id="input-label")
        with Horizontal():
            yield TextArea(id="message-input", classes="auto-grow")
            yield Button("Send", id="send-button")
        yield Footer()
        
    def on_mount(self) -> None:
        """Set up the application when it starts"""
        # Set Tokyo Night theme as default
        self.theme = "tokyo-night"
        
        self.llm_client = self._create_llm_client()
        self.embedding_function = set_embedding_function(self.llm, self.embedding_model)
        
        # Set text content after mounting
        self.query_one("#status").update(f"🤖 Model: {self.model} | 📚 Collection: {self.collection_name} | 💬 Ready to chat! Press Ctrl+Enter to send")
        self.query_one("#input-label").update("Type your message below and press Ctrl+Enter to send")
        

        
        # Focus on input field
        text_area = self.query_one("#message-input")
        text_area.focus()
        
    def _create_llm_client(self) -> OpenAI:
        """Create LLM client based on the provider"""
        if self.llm == "ollama":
            logger.info(f"Using Ollama as LLM")
            return OpenAI(
                base_url='http://localhost:11434/v1',
                api_key='ollama',  # required, but unused
            )
        elif self.llm == "gemini":
            logger.info(f"Using Gemini as LLM")
            return OpenAI(
                api_key=os.getenv("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            )
        else:
            logger.info(f"Using OpenAI as LLM")
            return OpenAI()
            
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Handle text area changes for auto-growing"""
        # Auto-grow the text area based on content
        text_area = event.text_area
        lines = len(text_area.text.split('\n'))
        new_height = max(3, min(lines + 1, 10))  # Min 3, max 10 lines
        text_area.styles.height = new_height
        

        
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id == "send-button":
            text_area = self.query_one("#message-input")
            message = text_area.text.strip()
            if message:
                self._process_message(message)
                
    def on_key(self, event) -> None:
        """Handle key events"""
        if event.key == "ctrl+c":
            self.exit()
        elif event.key == "cmd+p":
            self.action_command_palette()
        elif event.key == "ctrl+enter":
            # For TextArea, we need to handle Ctrl+Enter manually
            text_area = self.query_one("#message-input")
            message = text_area.text.strip()
            if message:
                self._process_message(message)
                # Prevent the event from propagating to the TextArea
                event.stop()
                
    def _process_message(self, message: str) -> None:
        """Process a user message"""
        # Clear input
        text_area = self.query_one("#message-input")
        text_area.text = ""
        text_area.styles.height = 3  # Reset height
        
        # Add user message to chat
        chat_history = self.query_one("#chat-history")
        chat_history.add_message(message, is_user=True)
        
        # Update status
        self.query_one("#status").update(f"🤔 Thinking with {self.model}...")
        
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
            
            # Generate response
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            if response.choices:
                assistant_message = response.choices[0].message.content
                
                # Add to conversation history
                self.conversation_history.append({"role": "assistant", "content": assistant_message})
                
                # Update UI in main thread
                self.call_from_thread(self._update_chat_with_response, assistant_message)
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
        chat_history = self.query_one("#chat-history")
        chat_history.add_message(response, is_user=False, model_name=self.model)
        self.query_one("#status").update(f"🤖 Model: {self.model} | 📚 Collection: {self.collection_name} | 💬 Ready to chat! Press Ctrl+Enter to send")


def process_chat(client, collection, llm, model, embedding_model):
    """Process chat operation"""
    logger.info(f"Starting chat interface for collection '{collection}'")
    
    app = ChatApp(client, collection, llm, model, embedding_model)
    app.run()
    
    logger.info(f"Chat session ended") 
