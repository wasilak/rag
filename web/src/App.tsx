import React, { useState, useEffect, useCallback } from "react";
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

const App: React.FC = () => {
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
  // Removed sidebarOpen state, no longer needed with permanent sidebar
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chatToDelete, setChatToDelete] = useState<string | null>(null);

  // Loading states for new features
  const [summarizing, setSummarizing] = useState(false);
  const [creatingChat, setCreatingChat] = useState(false);

  // Determine actual theme mode
  const actualThemeMode =
    themeMode === "system" ? (systemPrefersDark ? "dark" : "light") : themeMode;

  const theme = createTokyoNightTheme(actualThemeMode);

  // System theme detection
  useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    setSystemPrefersDark(mediaQuery.matches);

    const handler = (e: MediaQueryListEvent) => setSystemPrefersDark(e.matches);
    mediaQuery.addEventListener("change", handler);
    return () => mediaQuery.removeEventListener("change", handler);
  }, []);

  // Load initial data
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
        // Convert backend history to Message[] format
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
          if (chatsData.length > 0) setCurrentChatId(chatsData[0].id);
        }
      } catch (err) {
        console.error("Failed to load initial data:", err);
        setError("Failed to load configuration");
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

  // Summarize & Compact current chat
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

  // New Chat: Create chat in backend immediately
  const handleNewChat = async () => {
    try {
      setCreatingChat(true);
      const newChat = await apiService.createChat();
      console.log("Created new chat:", newChat);

      // Update chat list to include the new chat
      await refreshChatList();

      // Set as current chat
      setCurrentChatId(newChat.id);
      setChatHistory([]);
      setError(null);

      // Clear token stats for new chat
      setTokenStats({ total: 0, user: 0, assistant: 0, messages: 0 });
    } catch (err) {
      console.error("Failed to create new chat:", err);
      setError("Failed to create new chat");
    } finally {
      setCreatingChat(false);
    }
  };

  // Load chat list
  const refreshChatList = async () => {
    try {
      console.log("Refreshing chat list...");
      const chatsData = await fetch("/api/chats").then((r) => r.json());
      console.log("Chat list received:", chatsData);
      if (Array.isArray(chatsData)) {
        setChatList(chatsData);
        console.log("Chat list updated with", chatsData.length, "chats");
      }
    } catch (err) {
      console.error("Failed to load chat list:", err);
      setError("Failed to load chat list");
    }
  };

  // Switch chat
  const handleSwitchChat = async (chatId: string) => {
    try {
      console.log("Switching to chat:", chatId);

      // Clear current chat history first
      setChatHistory([]);

      const resp = await fetch(`/api/chats/${chatId}`);
      const data = await resp.json();
      if (data.success) {
        // Set current chat ID first
        setCurrentChatId(chatId);

        // Notify websocket to switch chat context
        webSocketService.switchChat(chatId);

        // Load chat history
        const historyData = await apiService.getHistory(chatId);
        console.log("Loaded history for chat:", chatId, historyData);

        if (historyData && Array.isArray(historyData.history)) {
          const mapped = historyData.history.map((item, idx) => ({
            id: String(idx) + "-" + String(Date.now()),
            content: item.content,
            isUser: item.role === "user",
            timestamp: new Date(),
          }));
          console.log("Setting chat history:", mapped);
          setChatHistory(mapped);
        } else {
          console.log("No history found for chat:", chatId);
          setChatHistory([]);
        }

        // Update token stats for the selected chat
        const tokensData = await apiService.getTokens(chatId);
        setTokenStats(tokensData);
      } else {
        setError("Failed to load chat");
      }
    } catch (err) {
      console.error("Failed to switch chat:", err);
      setError("Failed to switch chat");
    }
  };

  // Delete chat
  const handleDeleteChat = async (chatId: string) => {
    try {
      const resp = await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
      const data = await resp.json();
      if (data.success) {
        setDeleteDialogOpen(false);
        setChatToDelete(null);
        await refreshChatList();
        // If deleted chat was current, switch to another
        if (currentChatId === chatId) {
          if (chatList.length > 1) {
            const next = chatList.find((c) => c.id !== chatId);
            if (next) handleSwitchChat(next.id);
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

  const handleShowInfo = () => {
    setShowInfo(true);
  };

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
              startIcon={<ChatIcon />} // You may want to use <AddIcon /> if available
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
                {chatList.map((chat) => (
                  <ListItemButton
                    key={chat.id}
                    selected={chat.id === currentChatId}
                    onClick={() => handleSwitchChat(chat.id)}
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
                initialMessages={chatHistory}
                onRefreshChatList={refreshChatList}
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
};

export default App;
