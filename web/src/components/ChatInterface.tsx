import React, { useState, useEffect, useRef } from 'react';
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
} from '@mui/material';
import { Send, Stop, Clear } from '@mui/icons-material';
import ChatMessage from './ChatMessage';
import { webSocketService, Message } from '../services/websocket';

interface ChatInterfaceProps {
    onClearChat?: () => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ onClearChat }) => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [currentMessage, setCurrentMessage] = useState('');
    const [isProcessing, setIsProcessing] = useState(false);
    const [streamingContent, setStreamingContent] = useState('');
    const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);

    const messagesEndRef = useRef<HTMLDivElement>(null);
    const textFieldRef = useRef<HTMLInputElement>(null);

    // Auto-scroll to bottom
    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, streamingContent]);

    // Focus input on mount
    useEffect(() => {
        textFieldRef.current?.focus();
    }, []);

    // Global keyboard listener for Ctrl+C
    useEffect(() => {
        const handleGlobalKeyDown = (event: KeyboardEvent) => {
            if (event.key === 'c' && event.ctrlKey) {
                event.preventDefault();
                handleClearChat();
            }
        };

        document.addEventListener('keydown', handleGlobalKeyDown);
        return () => {
            document.removeEventListener('keydown', handleGlobalKeyDown);
        };
    }, []);

    // WebSocket event handlers
    useEffect(() => {
        webSocketService.onMessage((chunk: string) => {
            setStreamingContent(prev => prev + chunk);
        });

        webSocketService.onStatus((status) => {
            if (status === 'processing') {
                setIsProcessing(true);
                setStreamingContent('');

                // Create a new streaming message
                const messageId = Date.now().toString();
                setStreamingMessageId(messageId);

                const userMessage: Message = {
                    id: Date.now().toString(),
                    content: currentMessage,
                    isUser: true,
                    timestamp: new Date(),
                };

                setMessages(prev => [...prev, userMessage]);
                setCurrentMessage('');
            } else if (status === 'complete') {
                // Finalize the streaming message
                if (streamingContent && streamingMessageId) {
                    const assistantMessage: Message = {
                        id: streamingMessageId,
                        content: streamingContent,
                        isUser: false,
                        timestamp: new Date(),
                    };

                    setMessages(prev => [...prev, assistantMessage]);
                    setStreamingContent('');
                    setStreamingMessageId(null);
                }
                setIsProcessing(false);
            } else if (status === 'error') {
                setIsProcessing(false);
                setStreamingContent('');
                setStreamingMessageId(null);
            }
        });

        webSocketService.onError((error) => {
            console.error('WebSocket error:', error);
            setIsProcessing(false);
            setStreamingContent('');
            setStreamingMessageId(null);
        });
    }, [currentMessage, streamingContent, streamingMessageId]);

    const handleSendMessage = () => {
        const message = currentMessage.trim();
        if (!message || isProcessing) return;

        webSocketService.sendMessage(message);
    };

    const handleKeyPress = (event: React.KeyboardEvent) => {
        if (event.key === 'Enter' && event.ctrlKey) {
            event.preventDefault();
            handleSendMessage();
        } else if (event.key === 'c' && event.ctrlKey) {
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
        setMessages([]);
        setStreamingContent('');
        setStreamingMessageId(null);
        setIsProcessing(false);
    };

    const handleStopGeneration = () => {
        // In a real implementation, you might want to send a stop signal
        setIsProcessing(false);
        setStreamingContent('');
        setStreamingMessageId(null);
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            {/* Messages Area */}
            <Box
                sx={{
                    flex: 1,
                    overflow: 'auto',
                    p: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 2,
                }}
            >
                {messages.length === 0 && !isProcessing && (
                    <Box
                        sx={{
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            justifyContent: 'center',
                            height: '100%',
                            textAlign: 'center',
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

                {/* Streaming Message */}
                {isProcessing && streamingContent && (
                    <Fade in={true}>
                        <Box>
                            <ChatMessage
                                message={{
                                    id: 'streaming',
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
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, p: 2 }}>
                            <CircularProgress size={20} />
                            <Typography variant="body2" color="text.secondary">
                                🤔 Thinking...
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
                    borderColor: 'divider',
                    backgroundColor: 'background.paper',
                }}
            >
                <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
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
                            '& .MuiOutlinedInput-root': {
                                borderRadius: 2,
                            },
                            '& .MuiInputBase-root': {
                                overflow: 'auto',
                            },
                        }}
                    />

                    {isProcessing ? (
                        <IconButton
                            color="error"
                            onClick={handleStopGeneration}
                            sx={{ borderRadius: 2 }}
                        >
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

                {/* Clear Chat Button */}
                {messages.length > 0 && (
                    <Box sx={{ display: 'flex', justifyContent: 'center', mt: 1 }}>
                        <Button
                            variant="outlined"
                            size="small"
                            startIcon={<Clear />}
                            onClick={handleClearChat}
                            sx={{
                                borderRadius: 2,
                                textTransform: 'none',
                                color: 'text.secondary',
                                borderColor: 'divider',
                                '&:hover': {
                                    borderColor: 'error.main',
                                    color: 'error.main',
                                },
                            }}
                        >
                            Clear Chat (Ctrl+C)
                        </Button>
                    </Box>
                )}

                <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: 'block', mt: 1, textAlign: 'center' }}
                >
                    Press Ctrl+Enter to send • Enter for new line • Ctrl+C to clear chat
                </Typography>
            </Paper>
        </Box>
    );
};

export default ChatInterface;
