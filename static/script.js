document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const resetButton = document.getElementById('reset-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const suggestionButtons = document.querySelectorAll('.suggestion-btn');

    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        // Reset height if empty
        if (this.value === '') {
            this.style.height = 'auto';
        }
    });

    // Start a conversation and store the conversation ID
    let conversationId = null;

    // Initialize the conversation
    function initializeConversation() {
        fetch('/start-conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            conversationId = data.conversation_id;
            console.log("Conversation started with ID:", conversationId);
        })
        .catch(error => {
            console.error('Error starting conversation:', error);
            addMessage('assistant', 'Sorry, there was an error connecting to the server. Please try refreshing the page.');
        });
    }

    // Reset conversation
    function resetConversation() {
        if (!conversationId) {
            initializeConversation();
            return;
        }

        showLoading();
        
        fetch(`/reset-conversation/${conversationId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            // Clear chat messages except the first welcome message
            while (chatMessages.children.length > 1) {
                chatMessages.removeChild(chatMessages.lastChild);
            }
            console.log("Conversation reset");
        })
        .catch(error => {
            hideLoading();
            console.error('Error resetting conversation:', error);
            addMessage('assistant', 'Sorry, there was an error resetting the conversation. Please try refreshing the page.');
        });
    }

    // Add message to chat
    function addMessage(sender, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        const avatarIcon = document.createElement('i');
        avatarIcon.className = sender === 'user' ? 'fas fa-user' : 'fas fa-robot';
        avatarDiv.appendChild(avatarIcon);
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Process markdown in text
        contentDiv.innerHTML = processMarkdown(text);
        
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);
        
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add syntax highlighting to code blocks
        addSyntaxHighlighting();
        
        // Make links open in new tab
        makeLinksExternal();
    }

    // Process markdown
    function processMarkdown(text) {
        return marked.parse(text)
    }

    // Escape HTML to prevent XSS
    function escapeHTML(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    // Add syntax highlighting to code blocks (placeholder - would use a library in production)
    function addSyntaxHighlighting() {
        // This is a placeholder. In a production app, you'd use a library like highlight.js
        // For now, we'll just apply some basic styling
        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.style.color = '#333';
        });
    }

    // Make links open in new tab
    function makeLinksExternal() {
        const links = document.querySelectorAll('.message-content a');
        links.forEach(link => {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        });
    }

    // Show loading overlay
    function showLoading() {
        loadingOverlay.classList.remove('hidden');
    }

    // Hide loading overlay
    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    // Send message
    function sendMessage() {
        const message = userInput.value.trim();
        
        if (!message) return;
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // Add user message to chat
        addMessage('user', message);
        
        // Disable input while waiting for response
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Show loading
        showLoading();
        
        // If conversation not initialized yet, initialize it first
        if (!conversationId) {
            initializeConversation();
            // Wait a bit for initialization to complete
            setTimeout(() => {
                getChatResponse(message);
            }, 1000);
        } else {
            getChatResponse(message);
        }
    }

    // Get response from server
    function getChatResponse(message) {
        fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                question: message,
                use_realtime: true
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading
            hideLoading();
            
            // Add assistant message to chat
            addMessage('assistant', data.response);
            
            // Re-enable input
            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        })
        .catch(error => {
            // Hide loading
            hideLoading();
            
            console.error('Error:', error);
            addMessage('assistant', 'Sorry, there was an error processing your request. Please try again later.');
            
            // Re-enable input
            userInput.disabled = false;
            sendButton.disabled = false;
        });
    }

    // Event listeners
    sendButton.addEventListener('click', sendMessage);
    
    userInput.addEventListener('keydown', function(event) {
        // Send on Enter without Shift
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });
    
    resetButton.addEventListener('click', resetConversation);
    
    // Suggestion buttons
    suggestionButtons.forEach(button => {
        button.addEventListener('click', function() {
            userInput.value = this.textContent;
            userInput.dispatchEvent(new Event('input')); // Trigger resize
            sendMessage();
        });
    });

    // Initialize the conversation on page load
    initializeConversation();
    
    // Focus input on page load
    userInput.focus();
});
