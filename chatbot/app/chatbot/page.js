"use client"
import { useState, useRef, useEffect } from "react"
import { Send, Bot, User, Loader2 } from "lucide-react"

export default function ChatbotPage() {
  const [message, setMessage] = useState("")
  const [chat, setChat] = useState([
    {
      sender: "user",
      text: "Hi, I'm looking for Italian restaurants in London"
    },
    {
      sender: "bot",
      text: "I'd be happy to help you find Italian restaurants in London! I found several great options. Would you like to see restaurants in a specific area of London, or do you have any preference for price range?",
      intent: "search_restaurant",
      slots: {
        cuisine: "Italian",
        location: "London"
      }
    },
    {
      sender: "user",
      text: "Something in Central London, mid-range price would be perfect"
    },
    {
      sender: "bot",
      text: "Great! Here are some excellent mid-range Italian restaurants in Central London:\n\n1. Bella Italia Covent Garden\n2. Pasta Romana Leicester Square\n3. Trattoria Mondello Soho\n\nWould you like to make a reservation at any of these?",
      intent: "search_restaurant",
      slots: {
        cuisine: "Italian",
        location: "Central London",
        price_range: "mid-range"
      }
    },
    {
      sender: "user",
      text: "Yes, I'd like to book a table at Trattoria Mondello for 4 people tomorrow at 7 PM"
    },
    {
      sender: "bot",
      text: "Perfect! Let me confirm your booking details:\n\n• Restaurant: Trattoria Mondello, Soho\n• Date: Tomorrow (November 3, 2025)\n• Time: 7:00 PM\n• Party size: 4 people\n\nShall I proceed with this reservation?",
      intent: "book_restaurant",
      slots: {
        restaurant_name: "Trattoria Mondello",
        date: "2025-11-03",
        time: "19:00",
        party_size: "4"
      }
    },
    {
      sender: "user",
      text: "Yes, please confirm the booking"
    },
    {
      sender: "bot",
      text: "Excellent! Your reservation has been confirmed! \n\nBooking Reference: #TM-458921\n\nYou'll receive a confirmation email shortly. The restaurant is located at 45 Old Compton Street, Soho. Is there anything else I can help you with?",
      intent: "confirm_booking",
      slots: {
        booking_reference: "TM-458921",
        status: "confirmed"
      }
    }
  ])
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId] = useState(() => Math.floor(Math.random() * 101))
  const chatEndRef = useRef(null)
  const inputRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [chat])

const sendMessage = async () => {
  if (!message.trim() || isLoading) return;
  
  const userMessage = message.trim();
  setMessage("");
  setChat(prev => [...prev, { sender: "user", text: userMessage }]);
  setIsLoading(true);

  try {
    const response = await fetch("https://smart-integrated-system.onrender.com/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        message: userMessage,
        session_id: String(sessionId),
        return_probabilities: false
      })
    });

    const data = await response.json();

    const formattedSlots = {};
    if (Array.isArray(data.slots)) {
      data.slots.forEach(slot => {
        formattedSlots[slot.token] = slot.label;
      });
    }

    const botReply = {
      sender: "bot",
      text: data.response || "✅ Backend running but did not return a response!",
      intent: data.intent,
      slots: formattedSlots
    };

    setChat(prev => [...prev, botReply]);
  } catch (error) {
    console.error("Chat error:", error);
    setChat(prev => [...prev, {
      sender: "bot",
      text: "⚠️ Server error. Please try again!",
      intent: "unknown_intent",
      slots: {}
    }]);
  } finally {
    setIsLoading(false);
    inputRef.current?.focus();
  }
};

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <>
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes slideUp {
          from {
            opacity: 0;
            transform: translateY(10px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        
        .page-container {
          min-height: 100vh;
          background: linear-gradient(135deg, #eff6ff 0%, #ffffff 50%, #faf5ff 100%);
          padding: 1rem;
        }
        
        @media (min-width: 640px) {
          .page-container { padding: 1.5rem; }
        }
        
        @media (min-width: 768px) {
          .page-container { padding: 2rem; }
        }
        
        .main-container {
          max-width: 56rem;
          margin: 0 auto;
        }
        
        .header {
          text-align: center;
          margin-bottom: 2rem;
          animation: fadeIn 0.5s ease-out;
        }
        
        .header-icon {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 4rem;
          height: 4rem;
          background: linear-gradient(135deg, #3b82f6 0%, #9333ea 100%);
          border-radius: 1rem;
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
          margin-bottom: 1rem;
        }
        
        .header-title {
          font-size: 2.25rem;
          font-weight: bold;
          background: linear-gradient(to right, #2563eb, #9333ea);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
          margin-bottom: 0.5rem;
        }
        
        .session-id {
          color: #4b5563;
          font-size: 0.875rem;
        }
        
        .chat-container {
          background: white;
          border-radius: 1.5rem;
          box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
          overflow: hidden;
          border: 1px solid #f3f4f6;
        }
        
        .messages-area {
          height: 500px;
          overflow-y: auto;
          padding: 1.5rem;
          background: linear-gradient(to bottom, rgba(249, 250, 251, 0.5), white);
        }
        
        .empty-state {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
          color: #9ca3af;
        }
        
        .empty-icon {
          width: 4rem;
          height: 4rem;
          margin-bottom: 1rem;
          opacity: 0.5;
        }
        
        .message-row {
          display: flex;
          margin: 1rem 0;
          animation: slideUp 0.3s ease-out forwards;
        }
        
        .message-row-user {
          justify-content: flex-end;
        }
        
        .message-row-bot {
          justify-content: flex-start;
        }
        
        .message-container {
          display: flex;
          align-items: flex-end;
          gap: 0.5rem;
          max-width: 80%;
        }
        
        .message-container-user {
          flex-direction: row-reverse;
        }
        
        .avatar {
          flex-shrink: 0;
          width: 2rem;
          height: 2rem;
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
        }
        
        .avatar-user {
          background: linear-gradient(135deg, #3b82f6, #2563eb);
        }
        
        .avatar-bot {
          background: linear-gradient(135deg, #a855f7, #9333ea);
        }
        
        .message-content {
          flex: 1;
        }
        
        .message-bubble {
          border-radius: 1rem;
          padding: 0.75rem 1rem;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        .message-bubble-user {
          background: linear-gradient(135deg, #3b82f6, #2563eb);
          color: white;
          border-bottom-right-radius: 0.25rem;
        }
        
        .message-bubble-bot {
          background: white;
          border: 1px solid #e5e7eb;
          color: #1f2937;
          border-bottom-left-radius: 0.25rem;
        }
        
        .message-text {
          font-size: 0.875rem;
          line-height: 1.5;
          margin: 0;
          white-space: pre-line;
        }
        
        .message-meta {
          margin-top: 0.25rem;
          margin-left: 0.5rem;
          font-size: 0.75rem;
          color: #6b7280;
        }
        
        .input-area {
          padding: 1rem;
          background: white;
          border-top: 1px solid #f3f4f6;
        }
        
        .input-container {
          display: flex;
          gap: 0.75rem;
          align-items: flex-end;
        }
        
        .input-wrapper {
          flex: 1;
          position: relative;
        }
        
        .message-input {
          width: 100%;
          padding: 0.75rem 1.25rem;
          border-radius: 1rem;
          border: 1px solid #d1d5db;
          outline: none;
          transition: all 0.2s;
          font-size: 1rem;
          color: #1f2937;
        }
        
        .message-input:focus {
          border-color: #3b82f6;
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .message-input:disabled {
          background: #f9fafb;
          cursor: not-allowed;
        }
        
        .send-button {
          flex-shrink: 0;
          width: 3rem;
          height: 3rem;
          background: linear-gradient(135deg, #3b82f6, #9333ea);
          color: white;
          border-radius: 1rem;
          border: none;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .send-button:hover:not(:disabled) {
          background: linear-gradient(135deg, #2563eb, #7c3aed);
          transform: scale(1.05);
          box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        
        .send-button:active:not(:disabled) {
          transform: scale(0.95);
        }
        
        .send-button:disabled {
          background: linear-gradient(135deg, #d1d5db, #9ca3af);
          cursor: not-allowed;
          transform: none;
        }
        
        .footer {
          text-align: center;
          font-size: 0.75rem;
          color: #6b7280;
          margin-top: 1rem;
        }
        
        .spinner {
          animation: spin 1s linear infinite;
        }
      `}} />
      
      <div className="page-container">
        <div className="main-container">
          {/* Header */}
          <div className="header">
            <div className="header-icon">
              <Bot style={{ width: '2rem', height: '2rem', color: 'white' }} />
            </div>
            <h1 className="header-title">Conversational Chatbot With Intent and Slot Labeling</h1>
            <p className="session-id">Session ID: {sessionId}</p>
          </div>

          {/* Chat Container */}
          <div className="chat-container">
            {/* Chat Messages */}
            <div className="messages-area">
              {chat.length === 0 && (
                <div className="empty-state">
                  <Bot className="empty-icon" />
                  <p style={{ fontSize: '1.125rem' }}>Start a conversation...</p>
                </div>
              )}
              
              {chat.map((m, i) => (
                <div
                  key={i}
                  className={`message-row ${m.sender === "user" ? "message-row-user" : "message-row-bot"}`}
                  style={{ animationDelay: `${i * 0.05}s` }}
                >
                  <div className={`message-container ${m.sender === "user" ? "message-container-user" : ""}`}>
                    {/* Avatar */}
                    <div className={`avatar ${m.sender === "user" ? "avatar-user" : "avatar-bot"}`}>
                      {m.sender === "user" ? (
                        <User style={{ width: '1rem', height: '1rem', color: 'white' }} />
                      ) : (
                        <Bot style={{ width: '1rem', height: '1rem', color: 'white' }} />
                      )}
                    </div>
                    
                    {/* Message Bubble */}
                    <div className="message-content">
                      <div className={`message-bubble ${m.sender === "user" ? "message-bubble-user" : "message-bubble-bot"}`}>
                        <p className="message-text">{m.text}</p>
                      </div>
                      
                      {/* Intent and Slots Info */}
                      {m.sender === "bot" && (m.intent || m.slots) && (
                        <div className="message-meta">
                          {m.intent && <p style={{ margin: '0.125rem 0' }}>Intent: {m.intent}</p>}
                          {m.slots && Object.keys(m.slots).length > 0 && (
                            <p style={{ margin: '0.125rem 0' }}>Slots: {JSON.stringify(m.slots)}</p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
              
              {isLoading && (
                <div className="message-row message-row-bot">
                  <div className="message-container">
                    <div className="avatar avatar-bot">
                      <Bot style={{ width: '1rem', height: '1rem', color: 'white' }} />
                    </div>
                    <div className="message-bubble message-bubble-bot">
                      <Loader2 style={{ width: '1.25rem', height: '1.25rem', color: '#a855f7' }} className="spinner" />
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={chatEndRef} />
            </div>

            {/* Input Area */}
            <div className="input-area">
              <div className="input-container">
                <div className="input-wrapper">
                  <input
                    ref={inputRef}
                    type="text"
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type your message..."
                    disabled={isLoading}
                    className="message-input"
                  />
                </div>
                
                <button
                  onClick={sendMessage}
                  disabled={!message.trim() || isLoading}
                  className="send-button"
                >
                  {isLoading ? (
                    <Loader2 style={{ width: '1.25rem', height: '1.25rem' }} className="spinner" />
                  ) : (
                    <Send style={{ width: '1.25rem', height: '1.25rem' }} />
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Footer */}
          <p className="footer">Powered by CASA-NLU AI Model</p>
        </div>
      </div>
    </>
  )
}
