"""
Web-based Chatbot Interface for Laptop Recommender System
Provides a web interface for the conversational AI
"""

from flask import Flask, render_template, request, jsonify, session
import sys
import os
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatbot_interface import LaptopRecommenderChatbot

app = Flask(__name__)
app.secret_key = 'laptop_recommender_secret_key'

# Initialize chatbot
chatbot = LaptopRecommenderChatbot()

@app.route('/')
def index():
    """Main page"""
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'})
        
        # Get bot response
        bot_response = chatbot.get_response(user_message)
        
        # Get conversation summary
        summary = chatbot.get_conversation_summary()
        
        return jsonify({
            'response': bot_response,
            'user_preferences': summary['user_preferences'],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Error processing message: {str(e)}'})

@app.route('/reset', methods=['POST'])
def reset():
    """Reset conversation"""
    try:
        chatbot.reset_conversation()
        return jsonify({'message': 'Conversation reset successfully'})
    except Exception as e:
        return jsonify({'error': f'Error resetting conversation: {str(e)}'})

@app.route('/preferences', methods=['GET'])
def get_preferences():
    """Get current user preferences"""
    try:
        summary = chatbot.get_conversation_summary()
        return jsonify(summary['user_preferences'])
    except Exception as e:
        return jsonify({'error': f'Error getting preferences: {str(e)}'})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create the HTML template
    create_html_template()
    
    print("Starting web chatbot server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)

def create_html_template():
    """Create the HTML template for the chatbot interface"""
    html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Laptop Recommender Chatbot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .message-time {
            font-size: 11px;
            color: #999;
            margin-top: 5px;
        }
        
        .chat-input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .chat-input-wrapper {
            display: flex;
            gap: 10px;
        }
        
        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 12px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            transition: transform 0.2s;
        }
        
        .send-button:hover {
            transform: translateY(-2px);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .reset-button {
            padding: 8px 16px;
            background: #dc3545;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            font-size: 12px;
            margin-left: 10px;
        }
        
        .reset-button:hover {
            background: #c82333;
        }
        
        .preferences-panel {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #667eea;
        }
        
        .preferences-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        
        .preference-item {
            display: inline-block;
            background: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            margin: 2px;
            border: 1px solid #e0e0e0;
        }
        
        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 18px;
            border-bottom-left-radius: 4px;
            color: #666;
            font-style: italic;
        }
        
        .recommendation-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .recommendation-title {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 8px;
        }
        
        .recommendation-details {
            font-size: 13px;
            color: #666;
            line-height: 1.4;
        }
        
        .recommendation-price {
            font-weight: bold;
            color: #28a745;
            font-size: 16px;
            margin-top: 8px;
        }
        
        @media (max-width: 768px) {
            .chat-container {
                width: 95%;
                height: 90vh;
            }
            
            .message-content {
                max-width: 85%;
            }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>💻 Laptop Recommender</h1>
            <p>Your AI assistant for finding the perfect laptop</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="message bot">
                <div class="message-content">
                    Hello! I'm your laptop recommendation assistant. I can help you find the perfect laptop based on your needs. What are you looking for?
                    <div class="message-time">Just now</div>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            Bot is typing...
        </div>
        
        <div class="chat-input-container">
            <div class="chat-input-wrapper">
                <input type="text" id="messageInput" class="chat-input" placeholder="Tell me about your laptop needs..." autocomplete="off">
                <button id="sendButton" class="send-button">Send</button>
                <button id="resetButton" class="reset-button">Reset</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const resetButton = document.getElementById('resetButton');
        const typingIndicator = document.getElementById('typingIndicator');
        
        let isTyping = false;
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            
            // Check if content contains recommendations (bold text)
            if (content.includes('**')) {
                // Format recommendations
                content = content.replace(/\\*\\*(.*?)\\*\\*/g, '<strong>$1</strong>');
                content = content.replace(/\\n/g, '<br>');
            }
            
            messageContent.innerHTML = content;
            
            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            timeDiv.textContent = new Date().toLocaleTimeString();
            
            messageContent.appendChild(timeDiv);
            messageDiv.appendChild(messageContent);
            chatMessages.appendChild(messageDiv);
            
            // Scroll to bottom
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function showTyping() {
            isTyping = true;
            typingIndicator.style.display = 'block';
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function hideTyping() {
            isTyping = false;
            typingIndicator.style.display = 'none';
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isTyping) return;
            
            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            
            // Show typing indicator
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Sorry, I encountered an error: ' + data.error);
                } else {
                    // Hide typing indicator
                    hideTyping();
                    
                    // Add bot response
                    addMessage(data.response);
                    
                    // Show preferences if available
                    if (data.user_preferences && Object.keys(data.user_preferences).length > 0) {
                        showPreferences(data.user_preferences);
                    }
                }
            } catch (error) {
                hideTyping();
                addMessage('Sorry, I encountered an error. Please try again.');
            }
        }
        
        function showPreferences(preferences) {
            // Remove existing preferences panel
            const existingPanel = document.querySelector('.preferences-panel');
            if (existingPanel) {
                existingPanel.remove();
            }
            
            const preferencesPanel = document.createElement('div');
            preferencesPanel.className = 'preferences-panel';
            
            const title = document.createElement('div');
            title.className = 'preferences-title';
            title.textContent = 'Your Preferences:';
            preferencesPanel.appendChild(title);
            
            for (const [key, value] of Object.entries(preferences)) {
                const item = document.createElement('span');
                item.className = 'preference-item';
                item.textContent = `${key}: ${value}`;
                preferencesPanel.appendChild(item);
            }
            
            chatMessages.appendChild(preferencesPanel);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        async function resetConversation() {
            try {
                const response = await fetch('/reset', {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Error resetting conversation: ' + data.error);
                } else {
                    // Clear chat messages except the first bot message
                    const messages = chatMessages.querySelectorAll('.message');
                    for (let i = 1; i < messages.length; i++) {
                        messages[i].remove();
                    }
                    
                    // Remove preferences panel
                    const preferencesPanel = document.querySelector('.preferences-panel');
                    if (preferencesPanel) {
                        preferencesPanel.remove();
                    }
                    
                    addMessage('Conversation reset. How can I help you today?');
                }
            } catch (error) {
                addMessage('Error resetting conversation. Please try again.');
            }
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendMessage);
        resetButton.addEventListener('click', resetConversation);
        
        messageInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Focus on input when page loads
        messageInput.focus();
    </script>
</body>
</html>
'''
    
    with open('templates/chatbot.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("HTML template created successfully!") 