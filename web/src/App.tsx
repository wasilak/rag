import React, { useState, useEffect } from 'react';
import {
    ThemeProvider,
    CssBaseline,
    Box,
    AppBar,
    Toolbar,
    Typography,
    IconButton,
    Tooltip,
    Chip,
    Alert,
    Snackbar,
} from '@mui/material';
import {
    Brightness4,
    Brightness7,
    SettingsBrightness,
    Info,
    Clear,
} from '@mui/icons-material';

import { createTokyoNightTheme } from './theme/tokyoNight';
import ChatInterface from './components/ChatInterface';
import { webSocketService, apiService, ChatConfig, TokenStats } from './services/websocket';

type ThemeMode = 'light' | 'dark' | 'system';

const App: React.FC = () => {
    const [themeMode, setThemeMode] = useState<ThemeMode>('system');
    const [systemPrefersDark, setSystemPrefersDark] = useState(false);
    const [config, setConfig] = useState<ChatConfig | null>(null);
    const [tokenStats, setTokenStats] = useState<TokenStats | null>(null);
    const [connected, setConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [showInfo, setShowInfo] = useState(false);

    // Determine actual theme mode
    const actualThemeMode = themeMode === 'system'
        ? (systemPrefersDark ? 'dark' : 'light')
        : themeMode;

    const theme = createTokyoNightTheme(actualThemeMode);

    // System theme detection
    useEffect(() => {
        const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
        setSystemPrefersDark(mediaQuery.matches);

        const handler = (e: MediaQueryListEvent) => setSystemPrefersDark(e.matches);
        mediaQuery.addEventListener('change', handler);
        return () => mediaQuery.removeEventListener('change', handler);
    }, []);

    // Load initial data
    useEffect(() => {
        const loadInitialData = async () => {
            try {
                const [configData, tokensData] = await Promise.all([
                    apiService.getConfig(),
                    apiService.getTokens(),
                ]);
                setConfig(configData);
                setTokenStats(tokensData);
            } catch (err) {
                console.error('Failed to load initial data:', err);
                setError('Failed to load configuration');
            }
        };

        loadInitialData();
    }, []);

    // WebSocket connection management
    useEffect(() => {
        webSocketService.onConnectionChange(setConnected);
        webSocketService.onError(setError);
        webSocketService.onTokenUpdate(setTokenStats);

        return () => {
            webSocketService.disconnect();
        };
    }, []);

    const handleThemeChange = () => {
        const modes: ThemeMode[] = ['dark', 'light', 'system'];
        const currentIndex = modes.indexOf(themeMode);
        const nextIndex = (currentIndex + 1) % modes.length;
        setThemeMode(modes[nextIndex]);
    };

    const handleClearChat = async () => {
        try {
            await apiService.clearChat();
            const tokensData = await apiService.getTokens();
            setTokenStats(tokensData);
        } catch (err) {
            console.error('Failed to clear chat:', err);
            setError('Failed to clear chat');
        }
    };

    const handleShowInfo = () => {
        setShowInfo(true);
    };

    const getThemeIcon = () => {
        switch (themeMode) {
            case 'light': return <Brightness7 />;
            case 'dark': return <Brightness4 />;
            case 'system': return <SettingsBrightness />;
        }
    };

    const getConnectionStatus = () => {
        if (connected) {
            return { color: 'success' as const, text: 'Connected' };
        }
        return { color: 'error' as const, text: 'Disconnected' };
    };

    const connectionStatus = getConnectionStatus();

    const infoText = config ? [
        `Model: ${config.model}`,
        `LLM: ${config.llm}`,
        `Collection: ${config.collection}`,
        `Embedding: ${config.embedding_llm}/${config.embedding_model}`,
        tokenStats ? `Messages: ${tokenStats.messages}` : '',
        tokenStats ? `Total Tokens: ${tokenStats.total}` : '',
        tokenStats ? `User: ${tokenStats.user} | Assistant: ${tokenStats.assistant}` : '',
    ].filter(Boolean).join('\n') : '';

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
                {/* App Bar */}
                <AppBar
                    position="static"
                    elevation={0}
                    sx={{
                        backgroundColor: 'background.paper',
                        borderBottom: 1,
                        borderColor: 'divider',
                    }}
                >
                    <Toolbar sx={{ minHeight: { xs: 56, sm: 64 } }}>
                        <Typography
                            variant="h6"
                            component="div"
                            sx={{
                                flexGrow: 1,
                                color: 'text.primary',
                            }}
                        >
                            🤖 RAG Chat
                            {config && (
                                <Typography
                                    variant="body2"
                                    component="span"
                                    sx={{ ml: 2, opacity: 0.8, color: 'text.secondary' }}
                                >
                                    {config.model} | {config.collection}
                                </Typography>
                            )}
                        </Typography>

                        {/* Connection Status */}
                        <Chip
                            label={connectionStatus.text}
                            color={connectionStatus.color}
                            size="small"
                            sx={{
                                mr: 1,
                                backgroundColor: connectionStatus.color === 'success'
                                    ? 'success.main'
                                    : 'error.main',
                                color: 'white',
                            }}
                        />

                        {/* Token Stats */}
                        {tokenStats && (
                            <Chip
                                label={`${tokenStats.total} tokens`}
                                variant="outlined"
                                size="small"
                                sx={{
                                    mr: 1,
                                    borderColor: 'text.secondary',
                                    color: 'text.secondary',
                                    '& .MuiChip-label': {
                                        color: 'text.secondary',
                                    },
                                }}
                            />
                        )}

                        {/* Action Buttons */}
                        <Tooltip title="Clear chat">
                            <IconButton
                                onClick={handleClearChat}
                                sx={{
                                    color: 'text.secondary',
                                    '&:hover': {
                                        backgroundColor: 'action.hover',
                                    },
                                }}
                            >
                                <Clear />
                            </IconButton>
                        </Tooltip>

                        <Tooltip title="Show info">
                            <IconButton
                                onClick={handleShowInfo}
                                sx={{
                                    color: 'text.secondary',
                                    '&:hover': {
                                        backgroundColor: 'action.hover',
                                    },
                                }}
                            >
                                <Info />
                            </IconButton>
                        </Tooltip>

                        <Tooltip title={`Theme: ${themeMode}`}>
                            <IconButton
                                onClick={handleThemeChange}
                                sx={{
                                    color: 'text.secondary',
                                    '&:hover': {
                                        backgroundColor: 'action.hover',
                                    },
                                }}
                            >
                                {getThemeIcon()}
                            </IconButton>
                        </Tooltip>
                    </Toolbar>
                </AppBar>

                {/* Main Chat Interface */}
                <Box sx={{ flex: 1, overflow: 'hidden' }}>
                    <ChatInterface onClearChat={handleClearChat} />
                </Box>

                {/* Error Snackbar */}
                <Snackbar
                    open={!!error}
                    autoHideDuration={6000}
                    onClose={() => setError(null)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
                >
                    <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
                        {error}
                    </Alert>
                </Snackbar>

                {/* Info Snackbar */}
                <Snackbar
                    open={showInfo}
                    autoHideDuration={8000}
                    onClose={() => setShowInfo(false)}
                    anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
                >
                    <Alert onClose={() => setShowInfo(false)} severity="info" sx={{ width: '100%' }}>
                        <pre style={{ margin: 0, fontSize: '0.875rem', lineHeight: 1.4 }}>
                            {infoText}
                        </pre>
                    </Alert>
                </Snackbar>
            </Box>
        </ThemeProvider>
    );
};

export default App;
