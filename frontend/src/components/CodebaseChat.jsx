import React, { useState, useRef, useEffect } from 'react';
import { chatWithCodebase, indexCodebase, getRagStatus } from '../api';

/**
 * CodebaseChat Component
 * A floating chat interface for querying the backend codebase using RAG.
 * Features glassmorphism styling consistent with the app's design.
 */
const CodebaseChat = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [indexStatus, setIndexStatus] = useState(null);
    const [isIndexing, setIsIndexing] = useState(false);
    const messagesEndRef = useRef(null);

    // Scroll to bottom when messages change
    useEffect(() => {
        if (messagesEndRef.current) {
            messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [messages]);

    // Check index status when chat opens
    useEffect(() => {
        if (isOpen && !indexStatus) {
            checkIndexStatus();
        }
    }, [isOpen]);

    const checkIndexStatus = async () => {
        try {
            const status = await getRagStatus();
            setIndexStatus(status);
        } catch (error) {
            console.error('Failed to check index status:', error);
        }
    };

    const handleRebuildIndex = async () => {
        setIsIndexing(true);
        try {
            const result = await indexCodebase();
            setIndexStatus({ indexed: true, total_chunks: result.total_chunks });
            setMessages(prev => [...prev, {
                type: 'system',
                content: `✅ Index rebuilt! ${result.total_chunks} code chunks from ${result.files_indexed?.length || 0} files.`
            }]);
        } catch (error) {
            setMessages(prev => [...prev, {
                type: 'system',
                content: `❌ Failed to rebuild index: ${error.message}`
            }]);
        } finally {
            setIsIndexing(false);
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputValue.trim() || isLoading) return;

        const userMessage = inputValue.trim();
        setInputValue('');
        setMessages(prev => [...prev, { type: 'user', content: userMessage }]);
        setIsLoading(true);

        try {
            const response = await chatWithCodebase(userMessage);

            if (response.success) {
                setMessages(prev => [...prev, {
                    type: 'assistant',
                    content: response.explanation,
                    results: response.results
                }]);
            } else {
                setMessages(prev => [...prev, {
                    type: 'error',
                    content: response.error || 'Failed to get response'
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                type: 'error',
                content: error.response?.data?.detail || error.message || 'Failed to query codebase'
            }]);
        } finally {
            setIsLoading(false);
        }
    };

    const renderCodeBlock = (code, filename) => (
        <div className="code-block-container">
            <div className="code-block-header">
                <span className="code-filename">{filename}</span>
                <button
                    className="copy-btn"
                    onClick={() => navigator.clipboard.writeText(code)}
                    title="Copy code"
                >
                    📋
                </button>
            </div>
            <pre className="code-block">
                <code>{code}</code>
            </pre>
        </div>
    );

    const renderMessage = (message, index) => {
        const { type, content, results } = message;

        return (
            <div key={index} className={`chat-message ${type}`}>
                {type === 'user' && (
                    <div className="message-bubble user-bubble">
                        <span className="message-icon">👤</span>
                        <p>{content}</p>
                    </div>
                )}

                {type === 'assistant' && (
                    <div className="message-bubble assistant-bubble">
                        <span className="message-icon">🤖</span>
                        <div className="assistant-content">
                            <div className="explanation" dangerouslySetInnerHTML={{
                                __html: content
                                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                                    .replace(/`(.*?)`/g, '<code>$1</code>')
                                    .replace(/\n/g, '<br/>')
                            }} />

                            {results && results.length > 0 && (
                                <div className="code-results">
                                    <details>
                                        <summary>📂 View {results.length} code snippets</summary>
                                        {results.slice(0, 3).map((result, i) => (
                                            <div key={i} className="code-result-item">
                                                <div className="result-header">
                                                    <span className="result-rank">#{result.rank}</span>
                                                    <span className="result-name">{result.name}</span>
                                                    <span className="result-file">{result.file} ({result.lines})</span>
                                                </div>
                                                {renderCodeBlock(result.code, result.file)}
                                            </div>
                                        ))}
                                    </details>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {type === 'system' && (
                    <div className="message-bubble system-bubble">
                        <p>{content}</p>
                    </div>
                )}

                {type === 'error' && (
                    <div className="message-bubble error-bubble">
                        <span className="message-icon">⚠️</span>
                        <p>{content}</p>
                    </div>
                )}
            </div>
        );
    };

    return (
        <>
            {/* Floating Chat Button */}
            <button
                className={`chat-fab ${isOpen ? 'open' : ''}`}
                onClick={() => setIsOpen(!isOpen)}
                title="Chat with Codebase"
            >
                {isOpen ? '✕' : '💬'}
            </button>

            {/* Chat Panel */}
            {isOpen && (
                <div className="codebase-chat-panel glass-panel">
                    {/* Header */}
                    <div className="chat-header">
                        <div className="header-title">
                            <span className="header-icon">🔍</span>
                            <h3>Chat with Codebase</h3>
                        </div>
                        <button
                            className="rebuild-btn"
                            onClick={handleRebuildIndex}
                            disabled={isIndexing}
                            title="Rebuild code index"
                        >
                            {isIndexing ? '⏳' : '🔄'}
                        </button>
                    </div>

                    {/* Index Status */}
                    {indexStatus && (
                        <div className={`index-status ${indexStatus.indexed ? 'indexed' : 'not-indexed'}`}>
                            {indexStatus.indexed
                                ? `✅ Index ready (${indexStatus.total_chunks} chunks)`
                                : '⚠️ Index not built - click 🔄 to build'
                            }
                        </div>
                    )}

                    {/* Messages */}
                    <div className="chat-messages">
                        {messages.length === 0 && (
                            <div className="welcome-message">
                                <h4>👋 Welcome!</h4>
                                <p>Ask questions about the codebase:</p>
                                <ul>
                                    <li>"Where is password validation?"</li>
                                    <li>"Explain the dependency graph"</li>
                                    <li>"How are tasks created?"</li>
                                </ul>
                            </div>
                        )}
                        {messages.map(renderMessage)}
                        {isLoading && (
                            <div className="message-bubble assistant-bubble loading">
                                <span className="loading-dots">
                                    <span>.</span><span>.</span><span>.</span>
                                </span>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>

                    {/* Input */}
                    <form className="chat-input-form" onSubmit={handleSubmit}>
                        <input
                            type="text"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder="Ask about the code..."
                            disabled={isLoading}
                            autoFocus
                        />
                        <button type="submit" disabled={isLoading || !inputValue.trim()}>
                            ➤
                        </button>
                    </form>
                </div>
            )}

            <style>{`
        /* Floating Action Button */
        .chat-fab {
          position: fixed;
          bottom: 24px;
          right: 24px;
          width: 56px;
          height: 56px;
          border-radius: 50%;
          border: none;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          font-size: 24px;
          cursor: pointer;
          box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
          transition: all 0.3s ease;
          z-index: 1000;
        }
        
        .chat-fab:hover {
          transform: scale(1.1);
          box-shadow: 0 6px 30px rgba(102, 126, 234, 0.6);
        }
        
        .chat-fab.open {
          background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }

        /* Chat Panel */
        .codebase-chat-panel {
          position: fixed;
          bottom: 100px;
          right: 24px;
          width: 400px;
          max-width: calc(100vw - 48px);
          height: 500px;
          max-height: calc(100vh - 150px);
          border-radius: 16px;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          z-index: 999;
          background: rgba(30, 30, 60, 0.95);
          backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.1);
          box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
          animation: slideUp 0.3s ease-out;
        }
        
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }

        /* Header */
        .chat-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(255, 255, 255, 0.05);
        }
        
        .header-title {
          display: flex;
          align-items: center;
          gap: 8px;
        }
        
        .header-icon {
          font-size: 20px;
        }
        
        .chat-header h3 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
          color: #fff;
        }
        
        .rebuild-btn {
          width: 32px;
          height: 32px;
          border-radius: 8px;
          border: none;
          background: rgba(255, 255, 255, 0.1);
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .rebuild-btn:hover:not(:disabled) {
          background: rgba(255, 255, 255, 0.2);
        }
        
        .rebuild-btn:disabled {
          cursor: not-allowed;
          opacity: 0.5;
        }

        /* Index Status */
        .index-status {
          padding: 8px 16px;
          font-size: 12px;
          text-align: center;
        }
        
        .index-status.indexed {
          background: rgba(76, 175, 80, 0.2);
          color: #81c784;
        }
        
        .index-status.not-indexed {
          background: rgba(255, 152, 0, 0.2);
          color: #ffb74d;
        }

        /* Messages Container */
        .chat-messages {
          flex: 1;
          overflow-y: auto;
          padding: 16px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        /* Welcome Message */
        .welcome-message {
          text-align: center;
          padding: 24px;
          color: rgba(255, 255, 255, 0.7);
        }
        
        .welcome-message h4 {
          margin: 0 0 12px 0;
          color: #fff;
        }
        
        .welcome-message ul {
          text-align: left;
          margin: 12px 0 0 0;
          padding-left: 20px;
        }
        
        .welcome-message li {
          margin: 6px 0;
          font-size: 13px;
          color: rgba(255, 255, 255, 0.6);
        }

        /* Message Bubbles */
        .message-bubble {
          max-width: 90%;
          padding: 12px 16px;
          border-radius: 12px;
          animation: fadeIn 0.2s ease-out;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        .message-bubble p {
          margin: 0;
          line-height: 1.5;
        }

        .user-bubble {
          align-self: flex-end;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          display: flex;
          gap: 8px;
          align-items: flex-start;
        }

        .assistant-bubble {
          align-self: flex-start;
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
          display: flex;
          gap: 8px;
          align-items: flex-start;
        }
        
        .system-bubble {
          align-self: center;
          background: rgba(102, 126, 234, 0.2);
          color: #a5b4fc;
          font-size: 13px;
          text-align: center;
        }
        
        .error-bubble {
          align-self: center;
          background: rgba(244, 67, 54, 0.2);
          color: #ef9a9a;
          display: flex;
          gap: 8px;
        }

        .message-icon {
          font-size: 16px;
          flex-shrink: 0;
        }

        /* Assistant Content */
        .assistant-content {
          flex: 1;
          min-width: 0;
        }
        
        .explanation {
          font-size: 14px;
          line-height: 1.6;
        }
        
        .explanation code {
          background: rgba(0, 0, 0, 0.3);
          padding: 2px 6px;
          border-radius: 4px;
          font-family: 'Fira Code', monospace;
          font-size: 12px;
        }
        
        .explanation strong {
          color: #a5b4fc;
        }

        /* Code Results */
        .code-results {
          margin-top: 12px;
        }
        
        .code-results summary {
          cursor: pointer;
          padding: 8px;
          background: rgba(0, 0, 0, 0.2);
          border-radius: 8px;
          font-size: 13px;
        }
        
        .code-results summary:hover {
          background: rgba(0, 0, 0, 0.3);
        }
        
        .code-result-item {
          margin-top: 12px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 8px;
          overflow: hidden;
        }
        
        .result-header {
          display: flex;
          gap: 8px;
          align-items: center;
          padding: 8px 12px;
          background: rgba(0, 0, 0, 0.2);
          font-size: 12px;
        }
        
        .result-rank {
          color: #667eea;
          font-weight: 600;
        }
        
        .result-name {
          color: #81c784;
          font-weight: 500;
        }
        
        .result-file {
          color: rgba(255, 255, 255, 0.5);
          margin-left: auto;
        }

        /* Code Block */
        .code-block-container {
          background: rgba(0, 0, 0, 0.4);
        }
        
        .code-block-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 6px 12px;
          background: rgba(0, 0, 0, 0.3);
          font-size: 11px;
        }
        
        .code-filename {
          color: rgba(255, 255, 255, 0.6);
        }
        
        .copy-btn {
          background: none;
          border: none;
          cursor: pointer;
          padding: 4px;
          opacity: 0.6;
          transition: opacity 0.2s;
        }
        
        .copy-btn:hover {
          opacity: 1;
        }
        
        .code-block {
          margin: 0;
          padding: 12px;
          overflow-x: auto;
          font-family: 'Fira Code', 'Consolas', monospace;
          font-size: 11px;
          line-height: 1.5;
          color: #e0e0e0;
          max-height: 200px;
          overflow-y: auto;
        }

        /* Loading Animation */
        .loading .loading-dots {
          display: flex;
          gap: 4px;
        }
        
        .loading-dots span {
          animation: bounce 1.4s infinite;
          font-size: 24px;
          color: #667eea;
        }
        
        .loading-dots span:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .loading-dots span:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); }
          40% { transform: translateY(-8px); }
        }

        /* Input Form */
        .chat-input-form {
          display: flex;
          gap: 8px;
          padding: 16px;
          border-top: 1px solid rgba(255, 255, 255, 0.1);
          background: rgba(0, 0, 0, 0.2);
        }
        
        .chat-input-form input {
          flex: 1;
          padding: 12px 16px;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 24px;
          background: rgba(255, 255, 255, 0.1);
          color: #fff;
          font-size: 14px;
          outline: none;
          transition: all 0.2s;
        }
        
        .chat-input-form input::placeholder {
          color: rgba(255, 255, 255, 0.4);
        }
        
        .chat-input-form input:focus {
          border-color: #667eea;
          background: rgba(255, 255, 255, 0.15);
        }
        
        .chat-input-form button {
          width: 44px;
          height: 44px;
          border-radius: 50%;
          border: none;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          font-size: 18px;
          cursor: pointer;
          transition: all 0.2s;
        }
        
        .chat-input-form button:hover:not(:disabled) {
          transform: scale(1.05);
        }
        
        .chat-input-form button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        /* Scrollbar */
        .chat-messages::-webkit-scrollbar {
          width: 6px;
        }
        
        .chat-messages::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .chat-messages::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 3px;
        }
        
        .code-block::-webkit-scrollbar {
          height: 4px;
        }
        
        .code-block::-webkit-scrollbar-thumb {
          background: rgba(255, 255, 255, 0.2);
          border-radius: 2px;
        }

        /* Responsive */
        @media (max-width: 480px) {
          .codebase-chat-panel {
            width: calc(100vw - 32px);
            right: 16px;
            bottom: 88px;
            height: calc(100vh - 120px);
          }
          
          .chat-fab {
            right: 16px;
            bottom: 16px;
            width: 48px;
            height: 48px;
          }
        }
      `}</style>
        </>
    );
};

export default CodebaseChat;
