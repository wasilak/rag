import React, { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useParams } from "react-router-dom";
import ChatInterface from "./ChatInterface";

// Props for ChatWindow
interface ChatWindowProps {
  currentChatId: string | null;
  chatHistory: {
    id: string;
    content: string;
    isUser: boolean;
    timestamp: Date;
    modelName?: string;
  }[];
  isProcessing: boolean;
  streamingContent: string;
  pendingUserMessage: string | null;
  setChatHistory: React.Dispatch<React.SetStateAction<any[]>>;
  setStreamingContent: React.Dispatch<React.SetStateAction<string>>;
  setPendingUserMessage: React.Dispatch<React.SetStateAction<string | null>>;
  handleClearChat: () => Promise<void>;
  onSendMessage: (message: string) => void;
  setIsProcessing: React.Dispatch<React.SetStateAction<boolean>>;
  handleSwitchChat: (chatId: string) => Promise<void>;
}

const ChatWindow: React.FC<ChatWindowProps> = ({
  currentChatId,
  chatHistory,
  isProcessing,
  streamingContent,
  pendingUserMessage,
  setChatHistory,
  setStreamingContent,
  setPendingUserMessage,
  handleClearChat,
  onSendMessage,
  setIsProcessing,
  handleSwitchChat,
}) => {
  const params = useParams();
  const chatIdFromUrl = params.id || null;

  // Load chat content when URL param changes
  useEffect(() => {
    if (chatIdFromUrl && chatIdFromUrl !== currentChatId) {
      handleSwitchChat(chatIdFromUrl);
    }
  }, [chatIdFromUrl, currentChatId, handleSwitchChat]);

  return (
    <AnimatePresence mode="wait">
      <motion.div
        className="chat-fade"
        key={currentChatId}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
        style={{ height: "100%" }}
      >
        <ChatInterface
          onClearChat={handleClearChat}
          messages={chatHistory}
          isProcessing={isProcessing}
          streamingContent={streamingContent}
          pendingUserMessage={pendingUserMessage}
          onSendMessage={(message: string) => {
            setStreamingContent(""); // Clear before sending new message
            setIsProcessing(true);
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
            onSendMessage(message);
          }}
        />
      </motion.div>
    </AnimatePresence>
  );
};

export default ChatWindow;
