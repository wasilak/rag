import React, { useState, useEffect, useCallback } from "react";
import { Routes, Route, useParams, Navigate } from "react-router-dom";
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
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import { Brightness4, Brightness7, SettingsBrightness, Info, Clear } from "@mui/icons-material";

import { createTokyoNightTheme } from "./theme/tokyoNight";
import ChatInterface from "./components/ChatInterface";
import { webSocketService, apiService, ChatConfig, TokenStats } from "./services/websocket";
import {
  Drawer,
  List,
  ListItemText,
  ListItemSecondaryAction,
  IconButton as MuiIconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  Button,
  ListItemButton,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import ChatIcon from "@mui/icons-material/Chat";

type ThemeMode = "light" | "dark" | "system";

/**
 * ChatRouteWrapper is a wrapper to sync chatId in URL with App state.
 */
const ChatRouteWrapper: React.FC<{
  themeMode: ThemeMode;
  systemPrefersDark: boolean;
  setThemeMode: React.Dispatch<React.SetStateAction<ThemeMode>>;
  setSystemPrefersDark: React.Dispatch<React.SetStateAction<boolean>>;
  config: ChatConfig | null;
  tokenStats: TokenStats | null;
  connected: boolean;
  error: string | null;
  showInfo: boolean;
  chatHistory: {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    modelName?: string;
  }[];
  chatList: { id: string; title: string; created_at: string; last_updated: string }[];
  currentChatId: string | null;
  deleteDialogOpen: boolean;
  chatToDelete: string | null;
  summarizing: boolean;
  creatingChat: boolean;
  streamingContent: string;
  pendingUserMessage: string | null;
  pendingChatId: string | null;
  isProcessing: boolean;
  setConfig: React.Dispatch<React.SetStateAction<ChatConfig | null>>;
  setTokenStats: React.Dispatch<React.SetStateAction<TokenStats | null>>;
  setConnected: React.Dispatch<React.SetStateAction<boolean>>;
  setError: React.Dispatch<React.SetStateAction<string | null>>;
  setShowInfo: React.Dispatch<React.SetStateAction<boolean>>;
  setChatHistory: React.Dispatch<
    React.SetStateAction<
      { id: string; content: string; isUser: boolean; timestamp: Date; modelName?: string }[]
    >
  >;
  setChatList: React.Dispatch<
    React.SetStateAction<{ id: string; title: string; created_at: string; last_updated: string }[]>
  >;
  setCurrentChatId: React.Dispatch<React.SetStateAction<string | null>>;
  setDeleteDialogOpen: React.Dispatch<React.SetStateAction<boolean>>;
  setChatToDelete: React.Dispatch<React.SetStateAction<string | null>>;
  setSummarizing: React.Dispatch<React.SetStateAction<boolean>>;
  setCreatingChat: React.Dispatch<React.SetStateAction<boolean>>;
  setStreamingContent: React.Dispatch<React.SetStateAction<string>>;
  setPendingUserMessage: React.Dispatch<React.SetStateAction<string | null>>;
  setPendingChatId: React.Dispatch<React.SetStateAction<string | null>>;
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>;
  handleThemeChange: () => void;
  handleClearChat: () => Promise<void>;
  handleSummarizeChat: () => Promise<void>;
  handleNewChat: () => Promise<void>;
  refreshChatList: () => Promise<void>;
  handleSwitchChat: (chatId: string) => Promise<void>;
  handleDeleteChat: (chatId: string) => Promise<void>;
  infoText: string;
}> = (props) => {
  const navigate = useNavigate();
  const params = useParams();
  const chatIdFromUrl = params.id || null;

  // If no chatId in URL but there are chats, navigate to first chat
  useEffect(() => {
    if (!chatIdFromUrl && props.chatList.length > 0) {
      navigate(`/chat/${props.chatList[0].id}`, { replace: true });
    }
    // If no chats at all, stay on /chat
  }, [chatIdFromUrl, props.chatList]);

  // When a new chat is created, navigate to its URL
  useEffect(() => {
    if (props.currentChatId && chatIdFromUrl !== props.currentChatId) {
      navigate(`/chat/${props.currentChatId}`, { replace: true });
    }
  }, [props.currentChatId]);

  // If no chatId in URL and no chats, show empty state
  if (!chatIdFromUrl && props.chatList.length === 0) {
    return (
      <Box p={3}>
        <Typography variant="h5">No chats yet. Start a new chat!</Typography>
      </Box>
    );
  }

  // If chatId in URL but not in chatList, show not found
  if (chatIdFromUrl && !props.chatList.some((c) => c.id === chatIdFromUrl)) {
    return (
      <Box p={3}>
        <Typography variant="h5">Chat not found.</Typography>
      </Box>
    );
  }

  // Render main UI (sidebar, chat, etc) as before
  // (Insert the main App UI here, replacing all references to currentChatId with chatIdFromUrl)
  // We'll move the main App JSX into a function for clarity below.
  return <MainAppUI {...props} currentChatId={chatIdFromUrl} />;
};

/* (removed old App function, see below for new RoutedApp) */

/**
 * MainAppUI is a React component that renders the main App UI.
 * All props are passed through, and currentChatId is from URL.
 */
function MainAppUI(props: any) {
  const {
    themeMode,
    systemPrefersDark,
    config,
    tokenStats,
    connected,
    error,
    showInfo,
    chatHistory,
    chatList,
    currentChatId,
    deleteDialogOpen,
    chatToDelete,
    summarizing,
    creatingChat,
    streamingContent,
    pendingUserMessage,
    isProcessing,
    setThemeMode,
    setSystemPrefersDark,
    setConfig,
    setTokenStats,
    setConnected,
    setShowInfo,
    setChatHistory,
    setChatList,
    setCurrentChatId,
    setDeleteDialogOpen,
    setChatToDelete,
    setSummarizing,
    setCreatingChat,
    setStreamingContent,
    setPendingUserMessage,
    handleThemeChange,
    handleShowInfo,
    handleNewChat,
    handleSummarizeChat,
    handleClearChat,
    handleSwitchChat,
    handleDeleteChat,
    setError,
    infoText,
  } = props;

  const actualThemeMode =
    themeMode === "system" ? (systemPrefersDark ? "dark" : "light") : themeMode;

  const theme = createTokyoNightTheme(actualThemeMode);

  const navigate = useNavigate();
  const params = useParams();
  const chatIdFromUrl = params.id || null;

  // Load chat content when URL param changes
  useEffect(() => {
    if (chatIdFromUrl && chatIdFromUrl !== currentChatId) {
      handleSwitchChat(chatIdFromUrl);
    }
  }, [chatIdFromUrl]);

  const getThemeIcon = () => {
    switch (themeMode) {
      case "light":
        return <Brightness7 />;
      case "dark":
        return <Brightness4 />;
      case "system":
        return <SettingsBrightness />;
    }
  };

  const getConnectionStatus = () => {
    if (connected) {
      return { color: "success" as const, text: "Connected" };
    }
    return { color: "error" as const, text: "Disconnected" };
  };

  const connectionStatus = getConnectionStatus();

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ height: "100vh", display: "flex", flexDirection: "column" }}>
        {/* App Bar */}
        <AppBar
          position="fixed"
          elevation={0}
          sx={{
            backgroundColor: "background.paper",
            borderBottom: 1,
            borderColor: "divider",
          }}
        >
          <Toolbar sx={{ minHeight: { xs: 56, sm: 64 } }}>
            <Typography
              variant="h6"
              component="div"
              sx={{
                flexGrow: 1,
                color: "text.primary",
              }}
            >
              🤖 RAG Chat
              {config && (
                <Typography
                  variant="body2"
                  component="span"
                  sx={{ ml: 2, opacity: 0.8, color: "text.secondary" }}
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
                backgroundColor:
                  connectionStatus.color === "success" ? "success.main" : "error.main",
                color: "white",
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
                  borderColor: "text.secondary",
                  color: "text.secondary",
                  "& .MuiChip-label": {
                    color: "text.secondary",
                  },
                }}
              />
            )}

            {/* Action Buttons */}

            <Button
              startIcon={<ChatIcon />}
              onClick={handleNewChat}
              disabled={creatingChat}
              sx={{ ml: 2 }}
              variant="outlined"
            >
              New Chat
            </Button>

            <Button
              startIcon={
                <Typography variant="body2" sx={{ fontWeight: 700, fontSize: 18 }}>
                  Σ
                </Typography>
              }
              onClick={handleSummarizeChat}
              disabled={!currentChatId || summarizing}
              sx={{ ml: 2 }}
              variant="outlined"
            >
              Summarize & Compact
            </Button>

            <Tooltip title="Show info">
              <IconButton
                onClick={handleShowInfo}
                sx={{
                  color: "text.secondary",
                  "&:hover": {
                    backgroundColor: "action.hover",
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
                  color: "text.secondary",
                  "&:hover": {
                    backgroundColor: "action.hover",
                  },
                }}
              >
                {getThemeIcon()}
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>

        {/* Layout with permanent sidebar, AppBar always visible */}
        <Box
          sx={{ display: "flex", flexDirection: "column", height: "100vh", pt: { xs: 7, sm: 8 } }}
        >
          {/* Main content row: sidebar + chat */}
          <Box sx={{ display: "flex", flex: 1, minHeight: 0 }}>
            {/* Permanent Sidebar */}
            <Box
              sx={{
                width: 320,
                flexShrink: 0,
                bgcolor: "background.paper",
                borderRight: 1,
                borderColor: "divider",
                p: 2,
                display: "flex",
                flexDirection: "column",
                height: "100%",
                boxSizing: "border-box",
              }}
            >
              <Typography variant="h6" sx={{ mb: 2 }}>
                Chat History
              </Typography>
              <List sx={{ flex: 1, overflowY: "auto" }}>
                {chatList.length === 0 && (
                  <ListItemButton disabled>
                    <ListItemText primary="No chats yet." />
                  </ListItemButton>
                )}
                {chatList.map((chat: any) => (
                  <ListItemButton
                    key={chat.id}
                    selected={chat.id === currentChatId}
                    onClick={() => {
                      // Use navigate to update browser history and URL
                      navigate(`/chat/${chat.id}`);
                    }}
                    sx={{ display: "flex", alignItems: "flex-start" }}
                  >
                    <ChatIcon sx={{ mr: 1, mt: 0.5 }} />
                    <ListItemText
                      primary={chat.title}
                      secondary={
                        <>
                          <span>Created: {new Date(chat.created_at).toLocaleString()}</span>
                          <br />
                          <span>Updated: {new Date(chat.last_updated).toLocaleString()}</span>
                        </>
                      }
                    />
                    <ListItemSecondaryAction>
                      <MuiIconButton
                        edge="end"
                        aria-label="delete"
                        onClick={(e) => {
                          e.stopPropagation();
                          setChatToDelete(chat.id);
                          setDeleteDialogOpen(true);
                        }}
                        size="small"
                      >
                        <DeleteIcon />
                      </MuiIconButton>
                    </ListItemSecondaryAction>
                  </ListItemButton>
                ))}
              </List>
            </Box>

            {/* Main Chat Interface */}
            <Box sx={{ flex: 1, overflow: "hidden", minWidth: 0 }}>
              <ChatInterface
                onClearChat={handleClearChat}
                messages={chatHistory}
                isProcessing={isProcessing}
                streamingContent={streamingContent}
                pendingUserMessage={pendingUserMessage}
                onSendMessage={(message: string) => {
                  setStreamingContent(""); // Clear before sending new message
                  props.setIsProcessing(true);
                  setPendingUserMessage(null);

                  // Only optimistically add user message if this is an existing chat
                  if (currentChatId) {
                    setChatHistory((prev: any) => [
                      ...prev,
                      {
                        id: String(Date.now()),
                        content: message,
                        isUser: true,
                        timestamp: new Date(),
                      },
                    ]);
                  } else {
                    // For new chats, show pending user message until backend responds
                    setPendingUserMessage(message);
                  }
                  webSocketService.sendMessage(message);
                }}
              />
            </Box>
          </Box>
        </Box>

        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
          <DialogTitle>Delete Chat</DialogTitle>
          <DialogContent>
            <DialogContentText>
              Are you sure you want to delete this chat? This action cannot be undone.
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialogOpen(false)} color="primary">
              Cancel
            </Button>
            <Button
              onClick={() => chatToDelete && handleDeleteChat(chatToDelete)}
              color="error"
              variant="contained"
            >
              Delete
            </Button>
          </DialogActions>
        </Dialog>

        {/* Error Snackbar */}
        <Snackbar
          open={!!error}
          autoHideDuration={6000}
          onClose={() => setError(null)}
          anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
        >
          <Alert onClose={() => setError(null)} severity="error" sx={{ width: "100%" }}>
            {error}
          </Alert>
        </Snackbar>

        {/* Info Snackbar */}
        <Snackbar
          open={showInfo}
          autoHideDuration={8000}
          onClose={() => setShowInfo(false)}
          anchorOrigin={{ vertical: "bottom", horizontal: "left" }}
        >
          <Alert onClose={() => setShowInfo(false)} severity="info" sx={{ width: "100%" }}>
            <pre style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.4 }}>{infoText}</pre>
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
}

// All state, effects, and handlers are now here in RoutedApp:
const RoutedApp: React.FC = () => {
  const [themeMode, setThemeMode] = useState<ThemeMode>("system");
  const [systemPrefersDark, setSystemPrefersDark] = useState(false);
  const [config, setConfig] = useState<ChatConfig | null>(null);
  const [tokenStats, setTokenStats] = useState<TokenStats | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showInfo, setShowInfo] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    { id: string; content: string; isUser: boolean; timestamp: Date; modelName?: string }[]
  >([]);
  const [chatList, setChatList] = useState<
    { id: string; title: string; created_at: string; last_updated: string }[]
  >([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [creatingChat, setCreatingChat] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [pendingUserMessage, setPendingUserMessage] = useState<string | null>(null);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  // Use React Router's useNavigate for navigation
  // (must be inside the component)
  const navigate = require("react-router-dom").useNavigate();

  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemPrefersDark(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setSystemPrefersDark(e.matches);
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  // Ensure WebSocket connects only once per app load
  useEffect(() => {
    webSocketService.init();
  }, []);

  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const [configData, tokensData, historyData, chatsData] = await Promise.all([
          apiService.getConfig(),
          apiService.getTokens(),
          apiService.getHistory(),
          fetch("/api/chats").then((r) => r.json()),
        ]);
        setConfig(configData);
        setTokenStats(tokensData);
        if (historyData && Array.isArray(historyData.history)) {
          const mapped = historyData.history.map((item, idx) => ({
            id: String(idx) + "-" + String(Date.now()),
            content: item.content,
            isUser: item.role === "user",
            timestamp: new Date(),
          }));
          setChatHistory(mapped);
        }
        if (Array.isArray(chatsData)) {
          setChatList(chatsData);
          // Do not set currentChatId here; let the URL control it!
        }
      } catch (err) {
        console.error("Failed to load initial data:", err);
        setError("Failed to load configuration");
      }
    };

    loadInitialData();
  }, []);

  useEffect(() => {
    webSocketService.onMessage((chunk: string) => {
      setStreamingContent((prev) => prev + chunk);
    });
  }, []);

  useEffect(() => {
    webSocketService.onStatus(async (status, data) => {
      if (status === "complete") {
        setStreamingContent("");
        setPendingUserMessage(null);
        setIsProcessing(false);

        if (data && data.newChatId) {
          setCurrentChatId(data.newChatId);
          await refreshChatList();
          await handleSwitchChat(data.newChatId);
        } else if (currentChatId) {
          await refreshChatList();
          await handleSwitchChat(currentChatId);
        }
      }
    });
  }, [currentChatId, pendingChatId]);

  useEffect(() => {
    webSocketService.onConnectionChange(setConnected);
    webSocketService.onError(setError);
    webSocketService.onTokenUpdate(setTokenStats);

    return () => {
      webSocketService.disconnect();
    };
  }, []);

  const handleThemeChange = () => {
    const modes: ThemeMode[] = ["dark", "light", "system"];
    const currentIndex = modes.indexOf(themeMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    setThemeMode(modes[nextIndex]);
  };

  const handleClearChat = async () => {
    try {
      await apiService.clearChat();
      const tokensData = await apiService.getTokens();
      setTokenStats(tokensData);
      setChatHistory([]);
    } catch (err) {
      console.error("Failed to clear chat:", err);
      setError("Failed to clear chat");
    }
  };

  const handleSummarizeChat = async () => {
    if (!currentChatId) return;
    setSummarizing(true);
    try {
      const resp = await apiService.summarizeChat(currentChatId);
      if (resp.success && Array.isArray(resp.history)) {
        const mapped = resp.history.map((item, idx) => ({
          id: String(idx) + "-" + String(Date.now()),
          content: item.content,
          isUser: item.role === "user",
          timestamp: new Date(),
        }));
        setChatHistory(mapped);
        setError(null);
      } else {
        setError("Failed to summarize chat");
      }
    } catch (err) {
      console.error("Summarize chat error:", err);
      setError("Failed to summarize chat");
    } finally {
      setSummarizing(false);
    }
  };

  const handleNewChat = async () => {
    setCreatingChat(true);
    try {
      const newChat = await apiService.createChat();
      setCurrentChatId(newChat.id);
      setChatHistory([]);
      setStreamingContent("");
      setPendingUserMessage(null);
      setError(null);
      setTokenStats({ total: 0, user: 0, assistant: 0, messages: 0 });
      await refreshChatList();
      if (webSocketService.switchChat) {
        webSocketService.switchChat(newChat.id);
      }
      if (webSocketService.resetChat) {
        webSocketService.resetChat();
      }
      // Navigate to new chat to update URL and browser history
      navigate(`/chat/${newChat.id}`);
    } catch (err) {
      setError("Failed to create new chat");
    } finally {
      setCreatingChat(false);
    }
  };

  const refreshChatList = async () => {
    try {
      const chatsData = await fetch("/api/chats").then((r) => r.json());
      if (Array.isArray(chatsData)) {
        setChatList(chatsData);
      }
    } catch (err) {
      setError("Failed to load chat list");
    }
  };

  const handleSwitchChat = async (chatId: string) => {
    try {
      setChatHistory([]);
      const resp = await fetch(`/api/chats/${chatId}`);
      const data = await resp.json();
      if (data.success) {
        // Do not setCurrentChatId here; let URL be the source of truth!
        webSocketService.switchChat(chatId);
        const historyData = await apiService.getHistory(chatId);
        if (historyData && Array.isArray(historyData.history)) {
          const mapped = historyData.history.map((item, idx) => ({
            id: String(idx) + "-" + String(Date.now()),
            content: item.content,
            isUser: item.role === "user",
            timestamp: new Date(),
          }));
          setChatHistory(mapped);
        } else {
          setChatHistory([]);
        }
        const tokensData = await apiService.getTokens(chatId);
        setTokenStats(tokensData);
      } else {
        setError("Failed to load chat");
      }
    } catch (err) {
      setError("Failed to switch chat");
    }
  };

  const handleDeleteChat = async (chatId: string) => {
    try {
      const resp = await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
      const data = await resp.json();
      if (data.success) {
        setDeleteDialogOpen(false);
        setChatToDelete(null);
        await refreshChatList();
        if (currentChatId === chatId) {
          if (chatList.length > 1) {
            const next = chatList.find((c) => c.id !== chatId);
            if (next) navigate(`/chat/${next.id}`);
          } else {
            setChatHistory([]);
            setCurrentChatId(null);
          }
        }
      } else {
        setError("Failed to delete chat");
      }
    } catch (err) {
      setError("Failed to delete chat");
    }
  };

  const infoText = config
    ? [
        `Model: ${config.model}`,
        `LLM: ${config.llm}`,
        `Collection: ${config.collection}`,
        `Embedding: ${config.embedding_llm}/${config.embedding_model}`,
        tokenStats ? `Messages: ${tokenStats.messages}` : "",
        tokenStats ? `Total Tokens: ${tokenStats.total}` : "",
        tokenStats ? `User: ${tokenStats.user} | Assistant: ${tokenStats.assistant}` : "",
      ]
        .filter(Boolean)
        .join("\n")
    : "";

  return (
    <Routes>
      <Route
        path="/chat/:id"
        element={
          <MainAppUI
            themeMode={themeMode}
            systemPrefersDark={systemPrefersDark}
            config={config}
            tokenStats={tokenStats}
            connected={connected}
            error={error}
            showInfo={showInfo}
            chatHistory={chatHistory}
            chatList={chatList}
            currentChatId={currentChatId}
            deleteDialogOpen={deleteDialogOpen}
            chatToDelete={chatToDelete}
            summarizing={summarizing}
            creatingChat={creatingChat}
            streamingContent={streamingContent}
            pendingUserMessage={pendingUserMessage}
            pendingChatId={pendingChatId}
            isProcessing={isProcessing}
            setThemeMode={setThemeMode}
            setSystemPrefersDark={setSystemPrefersDark}
            setConfig={setConfig}
            setTokenStats={setTokenStats}
            setConnected={setConnected}
            setError={setError}
            setShowInfo={setShowInfo}
            setChatHistory={setChatHistory}
            setChatList={setChatList}
            setCurrentChatId={setCurrentChatId}
            setDeleteDialogOpen={setDeleteDialogOpen}
            setChatToDelete={setChatToDelete}
            setSummarizing={setSummarizing}
            setCreatingChat={setCreatingChat}
            setStreamingContent={setStreamingContent}
            setPendingUserMessage={setPendingUserMessage}
            setPendingChatId={setPendingChatId}
            setIsProcessing={setIsProcessing}
            handleThemeChange={handleThemeChange}
            handleClearChat={handleClearChat}
            handleSummarizeChat={handleSummarizeChat}
            handleNewChat={handleNewChat}
            refreshChatList={refreshChatList}
            handleSwitchChat={handleSwitchChat}
            handleDeleteChat={handleDeleteChat}
            infoText={infoText}
          />
        }
      />
      <Route
        path="/chat"
        element={
          <MainAppUI
            themeMode={themeMode}
            systemPrefersDark={systemPrefersDark}
            config={config}
            tokenStats={tokenStats}
            connected={connected}
            error={error}
            showInfo={showInfo}
            chatHistory={chatHistory}
            chatList={chatList}
            currentChatId={currentChatId}
            deleteDialogOpen={deleteDialogOpen}
            chatToDelete={chatToDelete}
            summarizing={summarizing}
            creatingChat={creatingChat}
            streamingContent={streamingContent}
            pendingUserMessage={pendingUserMessage}
            pendingChatId={pendingChatId}
            isProcessing={isProcessing}
            setThemeMode={setThemeMode}
            setSystemPrefersDark={setSystemPrefersDark}
            setConfig={setConfig}
            setTokenStats={setTokenStats}
            setConnected={setConnected}
            setError={setError}
            setShowInfo={setShowInfo}
            setChatHistory={setChatHistory}
            setChatList={setChatList}
            setCurrentChatId={setCurrentChatId}
            setDeleteDialogOpen={setDeleteDialogOpen}
            setChatToDelete={setChatToDelete}
            setSummarizing={setSummarizing}
            setCreatingChat={setCreatingChat}
            setStreamingContent={setStreamingContent}
            setPendingUserMessage={setPendingUserMessage}
            setPendingChatId={setPendingChatId}
            setIsProcessing={setIsProcessing}
            handleThemeChange={handleThemeChange}
            handleClearChat={handleClearChat}
            handleSummarizeChat={handleSummarizeChat}
            handleNewChat={handleNewChat}
            refreshChatList={refreshChatList}
            handleSwitchChat={handleSwitchChat}
            handleDeleteChat={handleDeleteChat}
            infoText={infoText}
          />
        }
      />

      <Route path="*" element={<Navigate to="/chat" replace />} />
    </Routes>
  );
};

export default RoutedApp;
