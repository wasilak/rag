import React, { useState, useEffect, useRef } from "react";
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
  Fade,
  Divider,
  Button,
} from "@mui/material";
import { Send, Stop, Clear } from "@mui/icons-material";
import ChatMessage from "./ChatMessage";
import { webSocketService, Message } from "../services/websocket";

interface ChatInterfaceProps {
  onClearChat?: () => void;
  messages: Message[];
  onSendMessage: (message: string) => void;
  isProcessing?: boolean;
  streamingContent?: string;
  pendingUserMessage?: string | null;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  onClearChat,
  messages,
  onSendMessage,
  isProcessing = false,
  streamingContent = "",
  pendingUserMessage = null,
}) => {
  const [currentMessage, setCurrentMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textFieldRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Focus input on mount
  useEffect(() => {
    textFieldRef.current?.focus();
  }, []);

  // Global keyboard listener for Ctrl+C
  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.key === "c" && event.ctrlKey) {
        event.preventDefault();
        if (onClearChat) onClearChat();
      }
    };

    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => {
      document.removeEventListener("keydown", handleGlobalKeyDown);
    };
  }, [onClearChat]);

  const handleSendMessage = () => {
    const message = currentMessage.trim();
    if (!message || isProcessing) return;
    onSendMessage(message);
    setCurrentMessage("");
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === "Enter" && event.ctrlKey) {
      event.preventDefault();
      handleSendMessage();
    } else if (event.key === "c" && event.ctrlKey) {
      event.preventDefault();
      handleClearChat();
    }
    // Allow Enter without Ctrl to create new lines
  };

  const handleClearChat = () => {
    if (onClearChat) {
      onClearChat();
    }
    // Also clear local state
  };

  const handleStopGeneration = () => {
    // In a real implementation, you might want to send a stop signal
  };

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      {/* Messages Area */}
      <Box
        sx={{
          flex: 1,
          overflow: "auto",
          p: 2,
          display: "flex",
          flexDirection: "column",
          gap: 2,
        }}
      >
        {messages.length === 0 && !isProcessing && (
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              textAlign: "center",
              opacity: 0.6,
            }}
          >
            <Typography variant="h5" gutterBottom>
              Welcome to RAG Chat! 🤖
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Ask questions about your documents and get AI-powered responses.
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Press Ctrl+Enter or click Send to submit your message.
            </Typography>
          </Box>
        )}

        {messages.map((message) => (
          <ChatMessage key={message.id} message={message} />
        ))}

        {/* Pending user message for new chat */}
        {pendingUserMessage && (
          <ChatMessage
            key="pending-user"
            message={{
              id: "pending-user",
              content: pendingUserMessage,
              isUser: true,
              timestamp: new Date(),
            }}
          />
        )}

        {/* Streaming Message */}
        {/* Streaming Message */}
        {isProcessing && streamingContent && (
          <Fade in={true}>
            <Box>
              <ChatMessage
                message={{
                  id: "streaming",
                  content: streamingContent,
                  isUser: false,
                  timestamp: new Date(),
                }}
                isStreaming={true}
              />
            </Box>
          </Fade>
        )}

        {/* Processing Indicator */}
        {isProcessing && !streamingContent && (
          <Fade in={true}>
            <Box
              sx={{ display: "flex", alignItems: "center", gap: 2, p: 2, flexDirection: "column" }}
            >
              <CircularProgress size={32} thickness={5} sx={{ mb: 1 }} />
              <Typography variant="body1" color="text.secondary" sx={{ fontWeight: 500 }}>
                🦄 The AI is conjuring up a magical answer...
              </Typography>
              <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mt: 1, fontStyle: "italic" }}
              >
                (Sometimes even unicorns need a moment to think!)
              </Typography>
            </Box>
          </Fade>
        )}

        <div ref={messagesEndRef} />
      </Box>

      <Divider />

      {/* Input Area */}
      <Paper
        elevation={0}
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: "divider",
          backgroundColor: "background.paper",
        }}
      >
        <Box sx={{ display: "flex", gap: 1, alignItems: "flex-end" }}>
          <TextField
            ref={textFieldRef}
            fullWidth
            multiline
            rows={6}
            maxRows={6}
            value={currentMessage}
            onChange={(e) => setCurrentMessage(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Ask a question about your documents..."
            disabled={isProcessing}
            variant="outlined"
            size="small"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 2,
              },
              "& .MuiInputBase-root": {
                overflow: "auto",
              },
            }}
          />

          {isProcessing ? (
            <IconButton color="error" onClick={handleStopGeneration} sx={{ borderRadius: 2 }}>
              <Stop />
            </IconButton>
          ) : (
            <IconButton
              color="primary"
              onClick={handleSendMessage}
              disabled={!currentMessage.trim()}
              sx={{ borderRadius: 2 }}
            >
              <Send />
            </IconButton>
          )}
        </Box>

        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ display: "block", mt: 1, textAlign: "center" }}
        >
          Press Ctrl+Enter to send • Enter for new line • Ctrl+C to clear chat
        </Typography>
      </Paper>
    </Box>
  );
};

export default ChatInterface;
