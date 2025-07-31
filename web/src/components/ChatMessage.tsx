import React from 'react';
import {
    Box,
    Paper,
    Typography,
    Chip,
    Avatar,
    useTheme,
} from '@mui/material';
import { Person, SmartToy } from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Message } from '../services/websocket';

interface ChatMessageProps {
    message: Message;
    isStreaming?: boolean;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message, isStreaming = false }) => {
    const theme = useTheme();
    const { content, isUser, timestamp, modelName } = message;

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <Box
            sx={{
                display: 'flex',
                flexDirection: isUser ? 'row-reverse' : 'row',
                gap: 2,
                mb: 2,
            }}
        >
            {/* Avatar */}
            <Avatar
                sx={{
                    bgcolor: isUser ? theme.palette.primary.main : theme.palette.secondary.main,
                    width: 40,
                    height: 40,
                }}
            >
                {isUser ? <Person /> : <SmartToy />}
            </Avatar>

            {/* Message Container */}
            <Box sx={{ flex: 1, maxWidth: '80%' }}>
                {/* Header */}
                <Box
                    sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        mb: 1,
                        justifyContent: isUser ? 'flex-end' : 'flex-start',
                    }}
                >
                    <Typography variant="body2" color="text.secondary">
                        {isUser ? 'You' : (modelName || 'Bot')}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        {formatTime(timestamp)}
                    </Typography>
                    {isStreaming && (
                        <Chip label="Typing..." size="small" color="secondary" variant="outlined" />
                    )}
                </Box>

                {/* Message Content */}
                <Paper
                    elevation={0}
                    sx={{
                        p: 2,
                        backgroundColor: isUser
                            ? theme.palette.primary.main + '20'
                            : theme.palette.background.paper,
                        border: 1,
                        borderColor: isUser
                            ? theme.palette.primary.main + '40'
                            : theme.palette.divider,
                        borderRadius: 2,
                        borderTopLeftRadius: isUser ? 2 : 0,
                        borderTopRightRadius: isUser ? 0 : 2,
                        position: 'relative',
                        '&::after': isStreaming ? {
                            content: '""',
                            position: 'absolute',
                            right: 8,
                            bottom: 8,
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            backgroundColor: theme.palette.secondary.main,
                            animation: 'pulse 1.5s ease-in-out infinite',
                        } : {},
                        '@keyframes pulse': {
                            '0%': {
                                opacity: 1,
                            },
                            '50%': {
                                opacity: 0.5,
                            },
                            '100%': {
                                opacity: 1,
                            },
                        },
                    }}
                >
                    {isUser ? (
                        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
                            {content}
                        </Typography>
                    ) : (
                        <Box
                            sx={{
                                '& > *:first-of-type': { mt: 0 },
                                '& > *:last-child': { mb: 0 },
                                '& p': {
                                    mb: 1,
                                    '&:last-child': { mb: 0 }
                                },
                                '& h1, & h2, & h3, & h4, & h5, & h6': {
                                    mt: 2,
                                    mb: 1,
                                    fontWeight: 600,
                                    '&:first-of-type': { mt: 0 }
                                },
                                '& ul, & ol': {
                                    pl: 3,
                                    mb: 1,
                                },
                                '& li': {
                                    mb: 0.5,
                                },
                                '& blockquote': {
                                    borderLeft: 4,
                                    borderColor: theme.palette.divider,
                                    pl: 2,
                                    py: 0.5,
                                    my: 1,
                                    fontStyle: 'italic',
                                    backgroundColor: theme.palette.action.hover,
                                },
                                '& code': {
                                    backgroundColor: theme.palette.action.hover,
                                    padding: '2px 4px',
                                    borderRadius: 1,
                                    fontSize: '0.875em',
                                    fontFamily: 'monospace',
                                    color: theme.palette.text.primary,
                                },
                                '& pre': {
                                    backgroundColor: theme.palette.action.hover,
                                    padding: 2,
                                    borderRadius: 1,
                                    overflow: 'auto',
                                    mb: 1,
                                    '& code': {
                                        backgroundColor: 'transparent',
                                        padding: 0,
                                    },
                                },
                                '& table': {
                                    borderCollapse: 'collapse',
                                    width: '100%',
                                    mb: 1,
                                },
                                '& th, & td': {
                                    border: 1,
                                    borderColor: theme.palette.divider,
                                    padding: 1,
                                    textAlign: 'left',
                                },
                                '& th': {
                                    backgroundColor: theme.palette.action.hover,
                                    fontWeight: 600,
                                },
                                '& a': {
                                    color: theme.palette.primary.main,
                                    textDecoration: 'none',
                                    '&:hover': {
                                        textDecoration: 'underline',
                                    },
                                },
                                '& strong': {
                                    fontWeight: 600,
                                },
                                '& em': {
                                    fontStyle: 'italic',
                                },
                                // GFM Strikethrough
                                '& del': {
                                    textDecoration: 'line-through',
                                    color: theme.palette.text.disabled,
                                },
                                // GFM Task lists
                                '& input[type="checkbox"]': {
                                    marginRight: 1,
                                },
                                // Footnotes styling
                                '& sup': {
                                    fontSize: '0.75em',
                                    color: theme.palette.primary.main,
                                },
                            }}
                        >
                            <ReactMarkdown
                                remarkPlugins={[remarkGfm]}
                                components={{
                                    // Custom component overrides if needed
                                    h1: ({ children }) => (
                                        <Typography variant="h4" component="h1" gutterBottom>
                                            {children}
                                        </Typography>
                                    ),
                                    h2: ({ children }) => (
                                        <Typography variant="h5" component="h2" gutterBottom>
                                            {children}
                                        </Typography>
                                    ),
                                    h3: ({ children }) => (
                                        <Typography variant="h6" component="h3" gutterBottom>
                                            {children}
                                        </Typography>
                                    ),
                                    p: ({ children }) => (
                                        <Typography variant="body1" component="p" paragraph>
                                            {children}
                                        </Typography>
                                    ),
                                    // Enhanced table styling for better GFM support
                                    table: ({ children }) => (
                                        <Box sx={{
                                            overflow: 'auto',
                                            mb: 2,
                                            border: `1px solid ${theme.palette.divider}`,
                                            borderRadius: 1,
                                        }}>
                                            <table style={{
                                                borderCollapse: 'collapse',
                                                width: '100%',
                                                minWidth: '400px',
                                            }}>
                                                {children}
                                            </table>
                                        </Box>
                                    ),
                                    th: ({ children }) => (
                                        <th style={{
                                            border: `1px solid ${theme.palette.divider}`,
                                            padding: '8px 12px',
                                            textAlign: 'left',
                                            backgroundColor: theme.palette.action.hover,
                                            fontWeight: 600,
                                        }}>
                                            {children}
                                        </th>
                                    ),
                                    td: ({ children }) => (
                                        <td style={{
                                            border: `1px solid ${theme.palette.divider}`,
                                            padding: '8px 12px',
                                            textAlign: 'left',
                                        }}>
                                            {children}
                                        </td>
                                    ),
                                }}
                            >
                                {content}
                            </ReactMarkdown>
                        </Box>
                    )}
                </Paper>
            </Box>
        </Box>
    );
};

export default ChatMessage;
