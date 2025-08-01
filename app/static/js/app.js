// WebSocket connection
let ws = null;
let uploadInProgress = false;
let translationAborted = false;
let currentTranslationController = null;

// Generate a unique client ID
const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
console.log('Generated client ID:', clientId);

// Initialize WebSocket connection
function initializeWebSocket() {
    console.log('Initializing WebSocket connection...');
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/${clientId}`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected successfully');
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data);
            updateProgress(data.progress, data.message);
        } catch (error) {
            console.error('Error processing WebSocket message:', error);
        }
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error occurred:', error);
        showError('WebSocket connection error');
    };
    
    ws.onclose = () => {
        console.log('WebSocket connection closed');
        if (uploadInProgress && !translationAborted) {
            console.log('Attempting to reconnect...');
            setTimeout(initializeWebSocket, 1000);
        }
    };
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    
    // デバッグ: 要素の存在確認
    const apiSettingsSection = document.getElementById('apiSettingsSection');
    const uploadForm = document.getElementById('uploadForm');
    
    console.log('Elements found:');
    console.log('- apiSettingsSection:', !!apiSettingsSection);
    console.log('- uploadForm:', !!uploadForm);
    
    if (!apiSettingsSection) {
        console.error('apiSettingsSection element not found!');
    }
    if (!uploadForm) {
        console.error('uploadForm element not found!');
    }
    
    initializeWebSocket();
    
    // API設定チェックを少し遅延させる
    setTimeout(() => {
        checkApiSettings();
    }, 100);
    
    // Settings button and modal elements
    const settingsButton = document.getElementById('settingsButton');
    const settingsModal = document.getElementById('settingsModal');
    const closeButton = settingsModal?.querySelector('.close-button');
    const updateApiSettingsButton = document.getElementById('updateApiSettingsButton');
    
    // Settings button click event
    if (settingsButton) {
        settingsButton.addEventListener('click', function() {
            if (settingsModal) settingsModal.style.display = 'block';
        });
    }
    
    // Close button click event
    if (closeButton) {
        closeButton.addEventListener('click', function() {
            if (settingsModal) settingsModal.style.display = 'none';
        });
    }
    
    // Close modal when clicking outside
    window.addEventListener('click', function(event) {
        if (event.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });
    
    // API settings update button click event
    if (updateApiSettingsButton) {
        updateApiSettingsButton.addEventListener('click', function() {
            updateApiSettings();
        });
    }
    
    // API settings form event listener
    const apiSettingsForm = document.getElementById('apiSettingsForm');
    if (apiSettingsForm) {
        apiSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveApiSettings();
        });
    }
});

// API Settings management
async function checkApiSettings() {
    try {
        console.log('Checking API settings...');
        const response = await fetch('/api/check-api-settings');
        
        if (!response.ok) {
            throw new Error(`API settings check failed with status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API settings check result:', data);

        const apiSettingsSection = document.getElementById('apiSettingsSection');
        const uploadForm = document.getElementById('uploadForm');

        if (data.has_api_settings) {
            console.log('API settings found, showing upload form');
            if (apiSettingsSection) apiSettingsSection.style.display = 'none';
            if (uploadForm) uploadForm.style.display = 'block';
            
            // Set API URL field in modal
            const modalApiUrl = document.getElementById('modalApiUrl');
            if (modalApiUrl && data.api_url) {
                modalApiUrl.value = data.api_url;
            }
        } else {
            console.log('API settings not found, showing API settings form');
            if (apiSettingsSection) apiSettingsSection.style.display = 'block';
            if (uploadForm) uploadForm.style.display = 'none';
        }
    } catch (error) {
        console.error('Error checking API settings:', error);
        // エラーが発生した場合はAPI設定画面を表示
        const apiSettingsSection = document.getElementById('apiSettingsSection');
        const uploadForm = document.getElementById('uploadForm');
        if (apiSettingsSection) apiSettingsSection.style.display = 'block';
        if (uploadForm) uploadForm.style.display = 'none';
        showError('Failed to check API settings status. Please configure your API settings.');
    }
}

// Save API settings function
async function saveApiSettings() {
    const apiKeyInput = document.getElementById('apiKey');
    const apiUrlInput = document.getElementById('apiUrl');
    
    if (!apiKeyInput || !apiUrlInput) {
        showError('API settings form elements not found');
        return;
    }
    
    const apiKey = apiKeyInput.value;
    const apiUrl = apiUrlInput.value;
    
    if (!apiKey) {
        showError('Please enter an API key');
        return;
    }
    
    if (!apiUrl) {
        showError('Please enter an API URL');
        return;
    }
    
    if (!apiUrl.startsWith('http')) {
        showError('Please enter a valid API URL (must start with http or https)');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('api_key', apiKey);
        formData.append('api_url', apiUrl);

        const response = await fetch('/api/save-api-settings', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showSuccess('API settings saved successfully');
            // Clear form fields
            apiKeyInput.value = '';
            apiUrlInput.value = '';
            // Re-check API settings status
            await checkApiSettings();
        } else {
            showError(result.detail || 'Failed to save API settings');
        }
    } catch (error) {
        console.error('Error saving API settings:', error);
        showError('Failed to save API settings');
    }
}

// Update API settings function
async function updateApiSettings() {
    const modalApiKey = document.getElementById('modalApiKey');
    const modalApiUrl = document.getElementById('modalApiUrl');
    
    if (!modalApiKey || !modalApiUrl) {
        showError('Modal form elements not found');
        return;
    }
    
    const apiKey = modalApiKey.value;
    const apiUrl = modalApiUrl.value;
    
    if (!apiKey) {
        showError('Please enter an API key');
        return;
    }
    
    if (!apiUrl) {
        showError('Please enter an API URL');
        return;
    }
    
    if (!apiUrl.startsWith('http')) {
        showError('Please enter a valid API URL (must start with http or https)');
        return;
    }

    try {
        const formData = new FormData();
        formData.append('api_key', apiKey);
        formData.append('api_url', apiUrl);

        const response = await fetch('/api/save-api-settings', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok) {
            showSuccess('API settings updated successfully');
            const settingsModal = document.getElementById('settingsModal');
            if (settingsModal) settingsModal.style.display = 'none';
            modalApiKey.value = '';
        } else {
            showError(result.detail || 'Failed to update API settings');
        }
    } catch (error) {
        console.error('Error updating API settings:', error);
        showError('Failed to update API settings');
    }
}

// Translation cancellation function
function cancelTranslation() {
    console.log('Translation cancellation requested');
    translationAborted = true;
    
    // Send abort signal if AbortController exists
    if (currentTranslationController) {
        currentTranslationController.abort();
    }
    
    // Update UI
    const cancelButton = document.getElementById('cancelButton');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (cancelButton) {
        cancelButton.disabled = true;
        cancelButton.textContent = 'Cancelling...';
    }
    
    // Close WebSocket connection
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    
    // Update progress display
    if (progressBar) progressBar.classList.remove('progress-bar-animated');
    if (progressText) progressText.textContent = 'Translation cancelled';
    
    // Reset UI after a short delay
    setTimeout(() => {
        resetTranslationUI();
        showWarning('Translation was cancelled by user');
    }, 1000);
}

// UI reset function
function resetTranslationUI() {
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressText = document.getElementById('progressText');
    const cancelButton = document.getElementById('cancelButton');
    const progressSection = document.getElementById('progressSection');
    const downloadSection = document.getElementById('downloadSection');
    
    if (progressBar) {
        progressBar.style.width = '0%';
        progressBar.setAttribute('aria-valuenow', 0);
    }
    if (progressPercentage) progressPercentage.textContent = '0%';
    if (progressText) progressText.textContent = 'Ready to translate';
    if (cancelButton) {
        cancelButton.style.display = 'none';
        cancelButton.disabled = false;
        cancelButton.textContent = 'Cancel Translation';
    }
    if (progressSection) progressSection.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    
    uploadInProgress = false;
    translationAborted = false;
    currentTranslationController = null;
    
    // Reconnect WebSocket
    initializeWebSocket();
}

// File upload and translation
async function uploadFile() {
    console.log('Starting file upload...');
    const fileInput = document.getElementById('file');
    const modelSelect = document.getElementById('model');
    const sourceLangSelect = document.getElementById('sourceLang');
    const targetLangSelect = document.getElementById('targetLang');
    const aiInstruction = document.getElementById('aiInstruction');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressSection = document.getElementById('progressSection');
    const downloadSection = document.getElementById('downloadSection');
    const cancelButton = document.getElementById('cancelButton');

    if (!fileInput || !fileInput.files.length) {
        showError('Please select a file');
        return;
    }

    // Reset
    translationAborted = false;
    
    // Create AbortController
    currentTranslationController = new AbortController();
    const signal = currentTranslationController.signal;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model', modelSelect?.value || 'claude-3-5-haiku');
    formData.append('source_lang', sourceLangSelect?.value || 'en');
    formData.append('target_lang', targetLangSelect?.value || 'ja');
    formData.append('client_id', clientId);
    formData.append('ai_instruction', aiInstruction?.value || '');

    console.log('Form data prepared:', {
        filename: fileInput.files[0].name,
        model: modelSelect?.value,
        source_lang: sourceLangSelect?.value,
        target_lang: targetLangSelect?.value,
        client_id: clientId
    });

    try {
        uploadInProgress = true;
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.setAttribute('aria-valuenow', 0);
        }
        const progressPercentage = document.getElementById('progressPercentage');
        if (progressPercentage) progressPercentage.textContent = '0%';
        if (progressText) progressText.textContent = 'Starting translation...';
        if (progressSection) progressSection.style.display = 'block';
        if (downloadSection) downloadSection.style.display = 'none';
        
        // Show cancel button
        if (cancelButton) cancelButton.style.display = 'inline-block';

        // Ensure WebSocket connection is active
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.log('Reconnecting WebSocket before upload...');
            initializeWebSocket();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log('Sending translation request...');
        const response = await fetch('/api/translate', {
            method: 'POST',
            body: formData,
            signal: signal
        });

        console.log('Translation response received:', response.status);

        // Return early if cancelled
        if (translationAborted) {
            console.log('Translation was aborted');
            return;
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Translation failed');
        }

        const result = await response.json();
        console.log('Translation result:', result);
        
        if (result.warning) {
            showWarning(result.warning);
        }

        // Create download links
        let downloadHtml = `<p>Translation completed successfully!</p>`;
        
        if (result.download_url) {
            downloadHtml += `<p><a href="${result.download_url}" class="btn btn-primary" download>Download Translated File</a></p>`;
        }
        
        if (result.extracted_text_url) {
            downloadHtml += `<p><a href="${result.extracted_text_url}" class="btn btn-secondary" download>Download Extracted Text</a></p>`;
        }
        
        if (result.translated_text_url) {
            downloadHtml += `<p><a href="${result.translated_text_url}" class="btn btn-secondary" download>Download Translated Text</a></p>`;
        }

        if (downloadSection) {
            downloadSection.innerHTML = downloadHtml;
            downloadSection.style.display = 'block';
        }
        showSuccess('Translation completed successfully');
        
        // Hide cancel button
        if (cancelButton) cancelButton.style.display = 'none';

    } catch (error) {
        // Handle AbortError specially
        if (error.name === 'AbortError') {
            console.log('Fetch aborted');
            return;
        }
        
        console.error('Translation error:', error);
        showError(error.message || 'Translation failed');
        if (progressBar) progressBar.classList.remove('progress-bar-animated');
        
        // Hide cancel button
        if (cancelButton) cancelButton.style.display = 'none';
    } finally {
        uploadInProgress = false;
        if (!translationAborted) {
            currentTranslationController = null;
        }
    }
}

// Progress bar updates
function updateProgress(progress, message) {
    // Skip progress updates if cancelled
    if (translationAborted) {
        return;
    }
    
    console.log('Updating progress:', progress, message);
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressText = document.getElementById('progressText');
    
    // Convert progress to percentage
    const percentage = Math.round(progress * 100);
    
    // Update progress bar width
    if (progressBar) {
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
    }
    
    // Update percentage text
    if (progressPercentage) progressPercentage.textContent = `${percentage}%`;
    
    // Update message
    if (progressText) progressText.textContent = message || '';
    
    // Add/remove animation based on completion
    if (progressBar) {
        if (percentage < 100) {
            progressBar.classList.add('progress-bar-animated');
        } else {
            progressBar.classList.remove('progress-bar-animated');
            
            // Hide cancel button when completed
            const cancelButton = document.getElementById('cancelButton');
            if (cancelButton) cancelButton.style.display = 'none';
        }
    }

    console.log(`Progress updated to: ${percentage}% - ${message}`);
}

// Notification functions
function showError(message) {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        console.error('Alerts container not found');
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertsContainer.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

function showSuccess(message) {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        console.error('Alerts container not found');
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertsContainer.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

function showWarning(message) {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        console.error('Alerts container not found');
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-warning alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertsContainer.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

// File input validation
function validateFileInput() {
    const fileInput = document.getElementById('file');
    const file = fileInput?.files[0];
    const submitButton = document.getElementById('submitButton');
    
    if (file) {
        const fileName = file.name.toLowerCase();
        const validExtensions = ['.docx', '.pptx', '.pdf', '.xlsx'];
        const isValid = validExtensions.some(ext => fileName.endsWith(ext));
        
        if (!isValid) {
            showError('Please select a valid file (DOCX, PPTX, PDF or XLSX)');
            fileInput.value = '';
            if (submitButton) submitButton.disabled = true;
        } else {
            if (submitButton) submitButton.disabled = false;
        }
    } else {
        if (submitButton) submitButton.disabled = true;
    }
}

// Language selection validation
function validateLanguageSelection() {
    const sourceLang = document.getElementById('sourceLang')?.value;
    const targetLang = document.getElementById('targetLang')?.value;
    const submitButton = document.getElementById('submitButton');
    
    if (sourceLang === targetLang) {
        showError('Source and target languages must be different');
        if (submitButton) submitButton.disabled = true;
    } else {
        // Re-enable if file is also selected
        const fileInput = document.getElementById('file');
        if (fileInput && fileInput.files.length > 0 && submitButton) {
            submitButton.disabled = false;
        }
    }
}

// Close alert button functionality
document.addEventListener('click', function(e) {
    if (e.target && e.target.classList.contains('btn-close')) {
        const alert = e.target.closest('.alert');
        if (alert) {
            alert.remove();
        }
    }
});
