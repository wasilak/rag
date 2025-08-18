# RAG Rust Migration: Phase-by-Phase Plan

## Phase 1: Base/Core Foundation (3-4 weeks)

### Project Structure
```
rag-rust/
├── Cargo.toml
├── src/
│   ├── main.rs                 // CLI entry point
│   ├── lib.rs                  // Library exports
│   ├── config/
│   │   ├── mod.rs              // Configuration module
│   │   ├── args.rs             // CLI arguments (replaces args.py)
│   │   └── env.rs              // Environment variables
│   ├── models/
│   │   ├── mod.rs              // LLM provider abstraction
│   │   ├── traits.rs           // Common LLM traits
│   │   ├── openai.rs           // OpenAI client
│   │   ├── ollama.rs           // Ollama client
│   │   ├── gemini.rs           // Gemini client
│   │   └── validation.rs       // Model validation (replaces models.py)
│   ├── storage/
│   │   ├── mod.rs              // Storage abstractions
│   │   └── vector.rs           // Vector DB interface
│   ├── error.rs                // Centralized error handling
│   └── utils/
│       ├── mod.rs
│       ├── tokenizer.rs        // Token counting
│       └── formatting.rs       // Output formatting
├── tests/
│   ├── integration/
│   └── unit/
└── examples/
```

### Core Dependencies (Cargo.toml)
```toml
[package]
name = "rag"
version = "0.7.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
description = "RAG (Retrieval-Augmented Generation) in Rust"

[dependencies]
# Async runtime
tokio = { version = "1.0", features = ["full"] }

# CLI and config
clap = { version = "4.4", features = ["derive", "env"] }
dotenvy = "0.15"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# HTTP client
reqwest = { version = "0.11", features = ["json", "stream"] }

# Vector database
qdrant-client = { version = "1.7", optional = true }

# LLM clients
async-openai = { version = "0.17", optional = true }
ollama-rs = { version = "0.1", optional = true }

# Tokenization
tiktoken-rs = "0.5"

[features]
default = ["qdrant", "openai", "ollama"]
qdrant = ["qdrant-client"]
openai = ["async-openai"]
ollama = ["ollama-rs"]
```

### Key Implementation Files

#### src/config/args.rs - CLI Interface
```rust
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug)]
#[command(name = "rag")]
#[command(about = "RAG (Retrieval-Augmented Generation) system")]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,

    /// Log level
    #[arg(long, default_value = "info", env = "RAG_LOG_LEVEL")]
    pub log_level: String,

    /// LLM provider
    #[arg(long, default_value = "ollama", env = "RAG_LLM")]
    pub llm: String,

    /// Model name
    #[arg(long, env = "RAG_MODEL")]
    pub model: Option<String>,

    /// Vector database URL
    #[arg(long, default_value = "http://localhost:6333", env = "RAG_VECTOR_URL")]
    pub vector_url: String,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Fill knowledge base with documents
    DataFill {
        /// Source paths or URLs
        sources: Vec<String>,
        
        /// Clean content using LLM
        #[arg(long)]
        clean_content: bool,

        /// Cleanup existing collection
        #[arg(long)]
        cleanup: bool,
    },
    /// Search the knowledge base
    Search {
        /// Search query
        query: String,
    },
    /// Start interactive chat interface
    Chat,
    /// Start web interface
    Web {
        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}
```

#### src/models/traits.rs - LLM Abstraction
```rust
use async_trait::async_trait;
use serde_json::Value;

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn generate_completion(
        &self,
        messages: Vec<ChatMessage>,
        stream: bool,
    ) -> anyhow::Result<LlmResponse>;

    async fn generate_embedding(&self, text: &str) -> anyhow::Result<Vec<f32>>;

    fn supports_streaming(&self) -> bool;
    fn model_name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub enum LlmResponse {
    Complete(String),
    Stream(Box<dyn futures::Stream<Item = String> + Send + Unpin>),
}
```

### Testing Strategy for Phase 1
```bash
# Unit tests for core components
cargo test unit_

# Integration tests for LLM providers
cargo test integration_llm

# Benchmark against Python version
cargo bench --bench core_benchmark
```

## Phase 2: Data-Fill Subcommand (4-5 weeks)

### Enhanced Project Structure
```
src/
├── documents/
│   ├── mod.rs
│   ├── types.rs                // Document structures
│   ├── loaders/
│   │   ├── mod.rs
│   │   ├── file.rs             // File system loader
│   │   ├── url.rs              // Web scraping
│   │   ├── pdf.rs              // PDF processing
│   │   └── markdown.rs         // Markdown processing
│   ├── processors/
│   │   ├── mod.rs
│   │   ├── chunking.rs         // Text chunking
│   │   ├── cleaning.rs         // Content cleaning
│   │   └── embedding.rs        // Embedding generation
│   └── pipeline.rs             // Processing orchestration
```

### Key Implementation: Document Processing Pipeline
```rust
// src/documents/pipeline.rs
use crate::documents::{Document, DocumentLoader, DocumentProcessor};
use futures::stream::{self, StreamExt};
use tokio::sync::Semaphore;

pub struct DocumentPipeline {
    loaders: Vec<Box<dyn DocumentLoader>>,
    processors: Vec<Box<dyn DocumentProcessor>>,
    concurrency_limit: usize,
}

impl DocumentPipeline {
    pub async fn process_sources(
        &self,
        sources: Vec<String>,
        cleanup: bool,
    ) -> anyhow::Result<ProcessingResult> {
        let semaphore = Semaphore::new(self.concurrency_limit);
        
        let results = stream::iter(sources)
            .map(|source| {
                let sem = &semaphore;
                async move {
                    let _permit = sem.acquire().await?;
                    self.process_single_source(source).await
                }
            })
            .buffer_unordered(self.concurrency_limit)
            .collect::<Vec<_>>()
            .await;

        // Aggregate results and store in vector DB
        self.store_documents(results, cleanup).await
    }
}
```

### Performance Focus Areas
- **Parallel document loading**: Use `rayon` for CPU-bound tasks
- **Streaming processing**: Process large documents without loading entirely into memory  
- **Efficient chunking**: Zero-copy string slicing where possible
- **Batch embedding**: Send multiple texts to embedding API in single request

## Phase 3: Search CLI Subcommand (3-4 weeks)

### Core Search Implementation
```rust
// src/search/mod.rs
use crate::models::LlmProvider;
use crate::storage::VectorStore;

pub struct SearchEngine {
    vector_store: Box<dyn VectorStore>,
    llm_provider: Box<dyn LlmProvider>,
    max_iterations: usize,
    min_relevance_score: f32,
}

impl SearchEngine {
    pub async fn search(&self, query: &str) -> anyhow::Result<SearchResult> {
        let mut current_query = query.to_string();
        let mut best_result = None;
        let mut iterations = Vec::new();

        for iteration in 0..self.max_iterations {
            let results = self.vector_store
                .similarity_search(&current_query, 4)
                .await?;

            let (score, analysis, refined_query) = self
                .evaluate_results(&results, &current_query)
                .await?;

            iterations.push(SearchIteration {
                query: current_query.clone(),
                score,
                analysis,
                results: results.clone(),
            });

            if score >= self.min_relevance_score || iteration == self.max_iterations - 1 {
                best_result = Some(results);
                break;
            }

            current_query = refined_query;
        }

        Ok(SearchResult {
            best_results: best_result.unwrap(),
            iterations,
            final_query: current_query,
            original_query: query.to_string(),
        })
    }
}
```

### Validation Against Python
Create integration tests that compare Rust vs Python results:
```rust
#[tokio::test]
async fn test_search_parity_with_python() {
    // Run same query against both implementations
    let rust_result = rust_search_engine.search("test query").await?;
    let python_result = run_python_search("test query").await?;
    
    // Compare relevance scores and result quality
    assert_similarity(rust_result, python_result, 0.95);
}
```

## Phase 4: TUI Interface (2-3 weeks)

### TUI with ratatui
```rust
// src/tui/app.rs
use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, Paragraph},
    Frame,
};

pub struct ChatApp {
    pub messages: Vec<ChatMessage>,
    pub input: String,
    pub search_engine: SearchEngine,
}

impl ChatApp {
    pub fn draw<B: Backend>(&mut self, f: &mut Frame<B>) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(f.size());

        // Chat history
        let messages: Vec<ListItem> = self
            .messages
            .iter()
            .map(|msg| {
                ListItem::new(format!("{}: {}", msg.role, msg.content))
                    .style(match msg.role.as_str() {
                        "user" => Style::default().fg(Color::Yellow),
                        _ => Style::default().fg(Color::Green),
                    })
            })
            .collect();

        let messages_widget = List::new(messages)
            .block(Block::default().borders(Borders::ALL).title("Chat"));
        f.render_widget(messages_widget, chunks[0]);

        // Input field
        let input_widget = Paragraph::new(self.input.as_ref())
            .block(Block::default().borders(Borders::ALL).title("Input"));
        f.render_widget(input_widget, chunks[1]);
    }

    pub async fn handle_input(&mut self, input: String) -> anyhow::Result<()> {
        if !input.trim().is_empty() {
            self.messages.push(ChatMessage {
                role: "user".to_string(),
                content: input.clone(),
            });

            let response = self.search_engine.chat(&input).await?;
            
            self.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: response,
            });
        }
        Ok(())
    }
}
```

## Phase 5: Web UI Backend (4-5 weeks)

### Axum Web Server with WebSocket Streaming
```rust
// src/web/mod.rs
use axum::{
    extract::{ws::WebSocket, WebSocketUpgrade, State},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use tokio::sync::RwLock;
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub search_engine: SearchEngine,
    pub sessions: Arc<RwLock<HashMap<Uuid, ChatSession>>>,
}

pub fn create_router() -> Router<AppState> {
    Router::new()
        .route("/api/chat", post(chat_handler))
        .route("/api/search", post(search_handler))
        .route("/ws", get(websocket_handler))
        .route("/", get(serve_frontend))
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(|socket| handle_websocket(socket, state))
}

async fn handle_websocket(socket: WebSocket, state: AppState) {
    let session_id = Uuid::new_v4();
    let session = ChatSession::new(session_id);
    
    {
        let mut sessions = state.sessions.write().await;
        sessions.insert(session_id, session.clone());
    }

    let (mut sender, mut receiver) = socket.split();

    // Handle incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let response_stream = state.search_engine
                    .chat_stream(&text, session_id)
                    .await;

                // Stream response back to client
                tokio::spawn(async move {
                    let mut stream = response_stream;
                    while let Some(chunk) = stream.next().await {
                        if sender.send(Message::Text(chunk)).await.is_err() {
                            break;
                        }
                    }
                });
            }
            _ => {}
        }
    }

    // Cleanup session
    let mut sessions = state.sessions.write().await;
    sessions.remove(&session_id);
}
```

## Migration Validation Strategy

### For Each Phase:
1. **Functionality Parity Tests**: Ensure Rust version produces same results as Python
2. **Performance Benchmarks**: Measure speed and memory improvements
3. **Integration Tests**: Verify compatibility with existing data and configs
4. **User Acceptance**: Test with real workflows before moving to next phase

### Rollback Plan:
- Keep Python version running in parallel
- Feature flags to switch between implementations
- Gradual user migration with fallback options

## Timeline Summary

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1. Base/Core | 3-4 weeks | CLI + LLM abstraction working |
| 2. Data-Fill | 4-5 weeks | Document ingestion pipeline |
| 3. Search CLI | 3-4 weeks | Basic search functionality |
| 4. TUI | 2-3 weeks | Terminal chat interface |
| 5. Web UI | 4-5 weeks | Full web application |

**Total: 16-21 weeks (4-5 months)**

This approach lets you:
- ✅ Validate each component thoroughly before building on it
- ✅ Maintain working Python fallback throughout migration  
- ✅ Get early performance wins from core components
- ✅ Learn Rust patterns progressively
- ✅ Minimize risk with incremental rollout
