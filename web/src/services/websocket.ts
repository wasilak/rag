import { io, Socket } from "socket.io-client";

export interface Message {
  id: string;
  content: string;
  isUser: boolean;
  timestamp: Date;
  modelName?: string;
}

export interface TokenStats {
  total: number;
  user: number;
  assistant: number;
  messages: number;
}

export interface ChatConfig {
  model: string;
  llm: string;
  collection: string;
  embedding_model: string;
  embedding_llm: string;
}

export type MessageHandler = (chunk: string) => void;
export type StatusHandler = (status: "processing" | "complete" | "error", data?: any) => void;
export type TokenHandler = (tokens: TokenStats) => void;
export type ErrorHandler = (error: string) => void;

class WebSocketService {
  private socket: Socket | null = null;
  private isConnected = false;
  private connecting = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  // Event handlers
  private messageHandler: MessageHandler | null = null;
  private statusHandler: StatusHandler | null = null;
  private tokenHandler: TokenHandler | null = null;
  private errorHandler: ErrorHandler | null = null;
  private connectionHandler: ((connected: boolean) => void) | null = null;

  // Do not auto-connect in constructor; use explicit init()
  constructor() {}

  public init() {
    if (this.socket && this.isConnected) return; // Already connected
    if (this.connecting) return; // Already trying to connect
    this.connecting = true;
    this.connect();
  }

  private connect() {
    const socketUrl =
      process.env.NODE_ENV === "development"
        ? "http://localhost:8080"
        : `${window.location.protocol}//${window.location.host}`;

    this.socket = io(socketUrl, {
      transports: ["websocket", "polling"],
      timeout: 20000,
      // forceNew: true, // Removed for robust singleton
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers() {
    if (!this.socket) return;

    this.socket.on("connect", () => {
      this.isConnected = true;
      this.connecting = false;
      this.reconnectAttempts = 0;
      this.connectionHandler?.(true);
      console.log("Connected to WebSocket");
    });

    this.socket.on("disconnect", (reason) => {
      this.isConnected = false;
      this.connectionHandler?.(false);
      this.connecting = false;
      console.log("Disconnected from WebSocket:", reason);
    });

    this.socket.on("connect_error", (error) => {
      this.isConnected = false;
      this.connectionHandler?.(false);
      this.connecting = false;
      console.error("Connection error:", error);
    });

    this.socket.on("message_chunk", (data: { chunk: string }) => {
      this.messageHandler?.(data.chunk);
    });

    this.socket.on("message_status", (data: { status: string; [key: string]: any }) => {
      console.log("WebSocket message_status received:", data);
      this.statusHandler?.(data.status as "processing" | "complete" | "error", data);
    });

    this.socket.on(
      "message_complete",
      (data: { status: string; tokens: TokenStats; newChatId?: string }) => {
        console.log("WebSocket message_complete received:", data);
        this.statusHandler?.("complete", data);
        this.tokenHandler?.(data.tokens);
      },
    );

    this.socket.on("message_error", (data: { error: string }) => {
      this.statusHandler?.("error");
      this.errorHandler?.(data.error);
    });

    this.socket.on("chat_title_updated", (data: { chatId: string; title: string }) => {
      console.log("Chat title updated:", data);
      // This could be used to update the chat list without a full refresh
    });
  }

  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts reached");
      this.errorHandler?.("Connection lost. Please refresh the page.");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(
      `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`,
    );

    setTimeout(() => {
      this.connect();
    }, delay);
  }

  public sendMessage(message: string) {
    if (!this.socket || !this.isConnected) {
      this.errorHandler?.("Not connected to server");
      return;
    }

    this.socket.emit("send_message", { message });
  }

  public resetChat() {
    if (this.socket && this.isConnected) {
      this.socket.emit("reset_chat");
    }
  }

  public switchChat(chatId: string) {
    if (this.socket && this.isConnected) {
      console.log("Switching websocket to chat:", chatId);
      this.socket.emit("switch_chat", { chat_id: chatId });
    }
  }

  public onMessage(handler: MessageHandler) {
    this.messageHandler = handler;
  }

  public onStatus(handler: StatusHandler) {
    this.statusHandler = handler;
  }

  public onTokenUpdate(handler: TokenHandler) {
    this.tokenHandler = handler;
  }

  public onError(handler: ErrorHandler) {
    this.errorHandler = handler;
  }

  public onConnectionChange(handler: (connected: boolean) => void) {
    this.connectionHandler = handler;
  }

  public disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.isConnected = false;
  }

  public isSocketConnected(): boolean {
    return this.isConnected && this.socket?.connected === true;
  }
}

// API service for REST endpoints
class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = process.env.NODE_ENV === "development" ? "http://localhost:8080/api" : "/api";
  }

  async getConfig(): Promise<ChatConfig> {
    const response = await fetch(`${this.baseUrl}/config`);
    if (!response.ok) throw new Error("Failed to fetch config");
    return response.json();
  }

  async getTokens(chatId?: string): Promise<TokenStats> {
    const url = chatId
      ? `${this.baseUrl}/tokens?chat_id=${encodeURIComponent(chatId)}`
      : `${this.baseUrl}/tokens`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Failed to fetch tokens");
    return response.json();
  }

  async getHistory(
    chatId?: string,
  ): Promise<{ history: Array<{ role: string; content: string }> }> {
    const url = chatId
      ? `${this.baseUrl}/history?chat_id=${encodeURIComponent(chatId)}`
      : `${this.baseUrl}/history`;
    const response = await fetch(url);
    if (!response.ok) throw new Error("Failed to fetch history");
    return response.json();
  }

  async clearChat(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/clear`, { method: "POST" });
    if (!response.ok) throw new Error("Failed to clear chat");
    return response.json();
  }

  async listChats(): Promise<
    Array<{ id: string; title: string; created_at: string; last_updated: string }>
  > {
    const response = await fetch(`${this.baseUrl}/chats`);
    if (!response.ok) throw new Error("Failed to fetch chat list");
    return response.json();
  }

  async loadChat(chatId: string): Promise<{ success: boolean }> {
    const response = await fetch(`${this.baseUrl}/chats/${chatId}`);
    if (!response.ok) throw new Error("Failed to load chat");
    return response.json();
  }

  async deleteChat(chatId: string): Promise<{ success: boolean }> {
    const response = await fetch(`${this.baseUrl}/chats/${chatId}`, { method: "DELETE" });
    if (!response.ok) throw new Error("Failed to delete chat");
    return response.json();
  }

  // Summarize and compact the current chat
  async summarizeChat(
    chatId: string,
  ): Promise<{ success: boolean; history: Array<{ role: string; content: string }> }> {
    const response = await fetch(`${this.baseUrl}/chats/${chatId}/summarize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error("Failed to summarize chat");
    return response.json();
  }

  // Create a new chat
  async createChat(): Promise<{
    id: string;
    title: string;
    created_at: string;
    last_updated: string;
  }> {
    const response = await fetch(`${this.baseUrl}/chats`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!response.ok) throw new Error("Failed to create new chat");
    return response.json();
  }
}

export const webSocketService = new WebSocketService();
export const apiService = new ApiService();
