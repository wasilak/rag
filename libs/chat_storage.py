import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class StoredMessage:
    id: str
    chat_id: str
    role: str  # 'user' or 'assistant'
    content: str
    created_at: datetime
    metadata: Optional[dict] = None


@dataclass
class StoredChat:
    id: str
    title: str
    created_at: datetime
    last_updated: datetime
    messages: List[StoredMessage]
    summary: Optional[str] = None


class ChatStorage:
    def __init__(self, db_path: str):
        """Initialize chat storage with SQLite database path"""
        self.db_path = Path(db_path)
        self._initialize_db()

    def _initialize_db(self):
        """Create database tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,  -- JSON string for additional data
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def create_chat(self, title: str) -> str:
        """Create a new chat and return its ID"""
        chat_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO chats (id, title) VALUES (?, ?)",
                (chat_id, title)
            )
            conn.commit()
        return chat_id

    def add_message(self, chat_id: str, role: str, content: str, metadata: Optional[dict] = None) -> str:
        """Add a message to a chat and return message ID"""
        message_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            # Update chat's last_updated timestamp
            conn.execute(
                "UPDATE chats SET last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (chat_id,)
            )
            # Insert message
            conn.execute(
                "INSERT INTO messages (id, chat_id, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
                (message_id, chat_id, role, content, str(metadata) if metadata else None)
            )
            conn.commit()
        return message_id

    def get_chat(self, chat_id: str) -> Optional[StoredChat]:
        """Get chat by ID including all its messages"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get chat
            chat_row = conn.execute(
                "SELECT * FROM chats WHERE id = ?",
                (chat_id,)
            ).fetchone()

            if not chat_row:
                return None

            # Get messages
            messages = []
            message_rows = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at",
                (chat_id,)
            ).fetchall()

            for msg in message_rows:
                messages.append(StoredMessage(
                    id=msg['id'],
                    chat_id=msg['chat_id'],
                    role=msg['role'],
                    content=msg['content'],
                    created_at=datetime.fromisoformat(msg['created_at']),
                    metadata=eval(msg['metadata']) if msg['metadata'] else None
                ))

            return StoredChat(
                id=chat_row['id'],
                title=chat_row['title'],
                created_at=datetime.fromisoformat(chat_row['created_at']),
                last_updated=datetime.fromisoformat(chat_row['last_updated']),
                messages=messages,
                summary=chat_row['summary']
            )

    def list_chats(self) -> List[StoredChat]:
        """List all chats"""
        chats = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute("SELECT * FROM chats ORDER BY last_updated DESC"):
                # Get messages for each chat
                messages = []
                message_rows = conn.execute(
                    "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at",
                    (row['id'],)
                ).fetchall()

                for msg in message_rows:
                    messages.append(StoredMessage(
                        id=msg['id'],
                        chat_id=msg['chat_id'],
                        role=msg['role'],
                        content=msg['content'],
                        created_at=datetime.fromisoformat(msg['created_at']),
                        metadata=eval(msg['metadata']) if msg['metadata'] else None
                    ))

                chats.append(StoredChat(
                    id=row['id'],
                    title=row['title'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    last_updated=datetime.fromisoformat(row['last_updated']),
                    messages=messages,
                    summary=row['summary']
                ))
        return chats

    def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            conn.commit()
            return cursor.rowcount > 0

    def update_summary(self, chat_id: str, summary: str) -> bool:
        """Update chat's summary"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE chats SET summary = ? WHERE id = ?",
                (summary, chat_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_chat_title(self, chat_id: str, title: str) -> bool:
        """Update chat's title"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE chats SET title = ?, last_updated = CURRENT_TIMESTAMP WHERE id = ?",
                (title, chat_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_messages_since(self, chat_id: str, timestamp: datetime) -> List[StoredMessage]:
        """Get messages for a chat since given timestamp"""
        messages = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? AND created_at > ? ORDER BY created_at",
                (chat_id, timestamp.isoformat())
            ).fetchall()

            for row in rows:
                messages.append(StoredMessage(
                    id=row['id'],
                    chat_id=row['chat_id'],
                    role=row['role'],
                    content=row['content'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    metadata=eval(row['metadata']) if row['metadata'] else None
                ))
        return messages

    def clear_messages(self, chat_id: str) -> bool:
        """Clear all messages for a chat"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.commit()
            return cursor.rowcount > 0

    def replace_with_summary(self, chat_id: str, summary: str) -> bool:
        """Replace all messages with a single summary message"""
        try:
            # Clear existing messages
            self.clear_messages(chat_id)
            
            # Add the summary as a single assistant message
            self.add_message(chat_id, "assistant", summary)
            
            # Update the summary field
            self.update_summary(chat_id, summary)
            
            return True
        except Exception as e:
            logger.error(f"Error replacing messages with summary: {e}")
            return False
