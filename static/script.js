document.addEventListener('DOMContentLoaded', function() {

    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-btn');
    const resetButton = document.getElementById('reset-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const suggestionButtons = document.querySelectorAll('.suggestion-btn');

    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';

        if (this.value === '') {
            this.style.height = 'auto';
        }
    });

    let conversationId = null;

    function initializeConversation() {
        showLoading();

        fetch('/start-conversation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            if (data.status === "success") {
                conversationId = data.conversation_id;
                console.log("Conversation started with ID:", conversationId);

                addMessage('assistant', 'Hello! I\'m your Open Source Contribution Assistant. How can I help you today?');
            } else {
                throw new Error(data.message || 'Failed to start conversation');
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error starting conversation:', error);
            addMessage('assistant', 'Sorry, there was an error connecting to the server. Please try refreshing the page.');
        });
    }

    function resetConversation() {
        showLoading();

        fetch('/api/reset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            if (data.status === "success") {

                chatMessages.innerHTML = '';

                initializeConversation();
            } else {
                throw new Error(data.message || 'Failed to reset conversation');
            }
        })
        .catch(error => {
            hideLoading();
            console.error('Error resetting conversation:', error);
            addMessage('assistant', 'Error resetting conversation: ' + error.message);
        });
    }

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

        contentDiv.innerHTML = processMarkdown(text);

        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(contentDiv);

        chatMessages.appendChild(messageDiv);

        chatMessages.scrollTop = chatMessages.scrollHeight;

        addSyntaxHighlighting();

        makeLinksExternal();
    }

    function processMarkdown(text) {
        return marked.parse(text);
    }

    function escapeHTML(text) {
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    function addSyntaxHighlighting() {

        const codeBlocks = document.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.style.color = '#333';
        });
    }

    function makeLinksExternal() {
        const links = document.querySelectorAll('.message-content a');
        links.forEach(link => {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        });
    }

    function showLoading() {
        loadingOverlay.classList.remove('hidden');
    }

    function hideLoading() {
        loadingOverlay.classList.add('hidden');
    }

    function sendMessage() {
        const message = userInput.value.trim();

        if (!message) return;

        userInput.value = '';
        userInput.style.height = 'auto';

        addMessage('user', message);

        userInput.disabled = true;
        sendButton.disabled = true;

        showLoading();

        if (!conversationId) {
            initializeConversation();

            setTimeout(() => {
                getChatResponse(message);
            }, 1000);
        } else {
            getChatResponse(message);
        }
    }

    function getChatResponse(message) {
        fetch('/api/chat', {  
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
            hideLoading();

            if (data.error) {
                throw new Error(data.error);
            }

            addMessage('assistant', data.answer);

            userInput.disabled = false;
            sendButton.disabled = false;
            userInput.focus();
        })
        .catch(error => {
            hideLoading();
            console.error('Error:', error);
            addMessage('assistant', 'Sorry, there was an error: ' + error.message);
            userInput.disabled = false;
            sendButton.disabled = false;
        });
    }

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keydown', function(event) {

        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    resetButton.addEventListener('click', resetConversation);

    suggestionButtons.forEach(button => {
        button.addEventListener('click', function() {
            userInput.value = this.textContent;
            userInput.dispatchEvent(new Event('input')); 
            sendMessage();
        });
    });

    initializeConversation();

    userInput.focus();
});