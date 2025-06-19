class DocumentTranslator {
    constructor() {
        this.clientId = this.generateClientId();
        this.websocket = null;
        this.initializeEventListeners();
        this.checkApiKey();
    }
    
    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9);
    }
    
    initializeEventListeners() {
        const form = document.getElementById('translationForm');
        form.addEventListener('submit', this.handleSubmit.bind(this));
    }
    
    async checkApiKey() {
        try {
            const response = await fetch('/api/check-api-key');
            const data = await response.json();
            
            const apiKeySection = document.getElementById('apiKeySection');
            const translationSection = document.getElementById('translationSection');
            
            if (data.has_api_key) {
                apiKeySection.style.display = 'none';
                translationSection.style.display = 'block';
            } else {
                apiKeySection.style.display = 'block';
                translationSection.style.display = 'none';
            }
        } catch (error) {
            console.error('API Key Check Error:', error);
            this.showErrorMessage('Failed to check API key');
        }
    }
    
    async saveApiKey(apiKey) {
        try {
            const formData = new FormData();
            formData.append('api_key', apiKey);
            
            const response = await fetch('/api/save-api-key', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                this.showSuccessMessage('API key saved successfully');
                this.checkApiKey();
            } else {
                throw new Error(result.detail || 'Failed to save API key');
            }
        } catch (error) {
            console.error('API Key Save Error:', error);
            this.showErrorMessage(error.message);
        }
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/${this.clientId}`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateProgress(data.progress, data.message);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showErrorMessage('An error occurred during communication');
        };
    }
    
    async handleSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(event.target);
        formData.append('client_id', this.clientId);
        
        // Update UI
        this.showProgress();
        this.connectWebSocket();
        
        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showResults(result);
            } else {
                throw new Error(result.detail || 'Translation failed');
            }
        } catch (error) {
            this.showError(error.message);
        } finally {
            if (this.websocket) {
                this.websocket.close();
            }
        }
    }
    
    showProgress() {
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('resultSection').style.display = 'none';
        document.getElementById('translateBtn').disabled = true;
    }
    
    updateProgress(progress, message) {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const statusMessage = document.getElementById('statusMessage');
        
        const percentage = Math.round(progress * 100);
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = `${percentage}%`;
        statusMessage.textContent = message || 'Translating...';
    }
    
    showResults(result) {
        const resultSection = document.getElementById('resultSection');
        const downloadLinks = document.getElementById('downloadLinks');
        
        let linksHtml = `<a href="${result.download_url}" class="download-btn">Download Translated File</a>`;
        
        if (result.extracted_text_url) {
            linksHtml += `<a href="${result.extracted_text_url}" class="download-btn secondary">Download Extracted Text</a>`;
        }
        
        if (result.translated_text_url) {
            linksHtml += `<a href="${result.translated_text_url}" class="download-btn secondary">Download Translated Text</a>`;
        }
        
        downloadLinks.innerHTML = linksHtml;
        resultSection.style.display = 'block';
        document.getElementById('translateBtn').disabled = false;
    }
    
    showError(message) {
        this.showErrorMessage(`Translation Error: ${message}`);
        document.getElementById('progressSection').style.display = 'none';
        document.getElementById('translateBtn').disabled = false;
    }
    
    showSuccessMessage(message) {
        const messageContainer = document.getElementById('messageContainer');
        messageContainer.innerHTML = `
            <div class="alert alert-success">
                <i class="icon-success">✓</i> ${message}
            </div>
        `;
        messageContainer.style.display = 'block';
        
        // Hide message after 3 seconds
        setTimeout(() => {
            messageContainer.style.display = 'none';
        }, 3000);
    }
    
    showErrorMessage(message) {
        const messageContainer = document.getElementById('messageContainer');
        messageContainer.innerHTML = `
            <div class="alert alert-danger">
                <i class="icon-error">!</i> ${message}
            </div>
        `;
        messageContainer.style.display = 'block';
        
        // Hide message after 5 seconds
        setTimeout(() => {
            messageContainer.style.display = 'none';
        }, 5000);
    }
}

// Application initialization
document.addEventListener('DOMContentLoaded', () => {
    const translator = new DocumentTranslator();
    
    // API key save form event listener
    const apiKeyForm = document.getElementById('apiKeyForm');
    apiKeyForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const apiKeyInput = document.getElementById('apiKey');
        translator.saveApiKey(apiKeyInput.value);
        apiKeyInput.value = ''; // Clear input field
    });
});
