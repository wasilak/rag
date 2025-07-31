import { io, Socket } from 'socket.io-client';

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
export type StatusHandler = (status: 'processing' | 'complete' | 'error') => void;
export type TokenHandler = (tokens: TokenStats) => void;
export type ErrorHandler = (error: string) => void;

class WebSocketService {
    private socket: Socket | null = null;
    private isConnected = false;
    private reconnectAttempts = 0;
    private maxReconnectAttempts = 5;
    private reconnectDelay = 1000;

    // Event handlers
    private messageHandler: MessageHandler | null = null;
    private statusHandler: StatusHandler | null = null;
    private tokenHandler: TokenHandler | null = null;
    private errorHandler: ErrorHandler | null = null;
    private connectionHandler: ((connected: boolean) => void) | null = null;

    constructor() {
        this.connect();
    }

    private connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = process.env.NODE_ENV === 'development'
            ? 'localhost:8080'
            : window.location.host;

        const socketUrl = process.env.NODE_ENV === 'development'
            ? 'http://localhost:8080'
            : `${window.location.protocol}//${window.location.host}`;

        this.socket = io(socketUrl, {
            transports: ['websocket', 'polling'],
            timeout: 20000,
            forceNew: true,
        });

        this.setupEventHandlers();
    }

    private setupEventHandlers() {
        if (!this.socket) return;

        this.socket.on('connect', () => {
            console.log('Connected to WebSocket');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.connectionHandler?.(true);
        });

        this.socket.on('disconnect', (reason) => {
            console.log('Disconnected from WebSocket:', reason);
            this.isConnected = false;
            this.connectionHandler?.(false);

            // Auto-reconnect on unexpected disconnection
            if (reason === 'io server disconnect') {
                // Server initiated disconnect, don't reconnect
                return;
            }

            this.handleReconnect();
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.isConnected = false;
            this.connectionHandler?.(false);
            this.handleReconnect();
        });

        this.socket.on('message_chunk', (data: { chunk: string }) => {
            this.messageHandler?.(data.chunk);
        });

        this.socket.on('message_status', (data: { status: string }) => {
            this.statusHandler?.(data.status as 'processing' | 'complete' | 'error');
        });

        this.socket.on('message_complete', (data: { status: string; tokens: TokenStats }) => {
            this.statusHandler?.('complete');
            this.tokenHandler?.(data.tokens);
        });

        this.socket.on('message_error', (data: { error: string }) => {
            this.statusHandler?.('error');
            this.errorHandler?.(data.error);
        });
    }

    private handleReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.errorHandler?.('Connection lost. Please refresh the page.');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`);

        setTimeout(() => {
            this.connect();
        }, delay);
    }

    public sendMessage(message: string) {
        if (!this.socket || !this.isConnected) {
            this.errorHandler?.('Not connected to server');
            return;
        }

        this.socket.emit('send_message', { message });
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
        this.baseUrl = process.env.NODE_ENV === 'development'
            ? 'http://localhost:8080/api'
            : '/api';
    }

    async getConfig(): Promise<ChatConfig> {
        const response = await fetch(`${this.baseUrl}/config`);
        if (!response.ok) throw new Error('Failed to fetch config');
        return response.json();
    }

    async getTokens(): Promise<TokenStats> {
        const response = await fetch(`${this.baseUrl}/tokens`);
        if (!response.ok) throw new Error('Failed to fetch tokens');
        return response.json();
    }

    async getHistory(): Promise<{ history: Array<{ role: string; content: string }> }> {
        const response = await fetch(`${this.baseUrl}/history`);
        if (!response.ok) throw new Error('Failed to fetch history');
        return response.json();
    }

    async clearChat(): Promise<{ status: string }> {
        const response = await fetch(`${this.baseUrl}/clear`, { method: 'POST' });
        if (!response.ok) throw new Error('Failed to clear chat');
        return response.json();
    }
}

export const webSocketService = new WebSocketService();
export const apiService = new ApiService();
