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


// Load models function
async function loadModels() {
    try {
        console.log('Loading models from API...');
        
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${data.detail || 'Unknown error'}`);
        }
        
        const modelSelect = document.getElementById('model');
        if (!modelSelect) {
            console.error('Model select element not found');
            return;
        }
        
        modelSelect.innerHTML = '';
        
        const models = data.models || {};
        
        // Handle empty models
        if (Object.keys(models).length === 0) {
            if (data.error) {
                showError(`Failed to load models: ${data.error}`);
            }
            
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'API configuration required';
            placeholderOption.disabled = true;
            placeholderOption.selected = true;
            modelSelect.appendChild(placeholderOption);
            
            const submitButton = document.getElementById('submitButton');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.title = 'Please complete API configuration';
            }
            
            console.log('No models available. API configuration required.');
            return;
        }
        
        // Add model options in alphabetical order
        Object.entries(models)
            .sort(([a], [b]) => a.localeCompare(b))
            .forEach(([modelId, displayName]) => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = displayName;
                modelSelect.appendChild(option);
            });
        
        // Select default model (first in alphabetical order)
        const sortedModelIds = Object.keys(models).sort();
        if (sortedModelIds.length > 0) {
            modelSelect.value = sortedModelIds[0];
            console.log(`Default model selected: ${sortedModelIds[0]}`);
        }
        
        // Enable translation button
        const submitButton = document.getElementById('submitButton');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.title = '';
        }
        
        console.log('Models loaded successfully (alphabetical order):', sortedModelIds);
        showSuccess('Models loaded successfully');
        
    } catch (error) {
        console.error('Model loading error:', error);
        showError(`Failed to load models: ${error.message}`);
        
        const modelSelect = document.getElementById('model');
        if (modelSelect) {
            modelSelect.innerHTML = '';
            const errorOption = document.createElement('option');
            errorOption.value = '';
            errorOption.textContent = 'Model loading error';
            errorOption.disabled = true;
            errorOption.selected = true;
            modelSelect.appendChild(errorOption);
        }
        
        const submitButton = document.getElementById('submitButton');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.title = 'Please check API settings';
        }
    }
}


// Add refresh models button
function addRefreshModelsButton() {
    const modelSelect = document.getElementById('model');
    if (!modelSelect) {
        console.error('Model select element not found');
        return;
    }
    
    const modelContainer = modelSelect.parentElement;
    
    // Remove existing button if present
    const existingButton = modelContainer.querySelector('.refresh-models-btn');
    if (existingButton) {
        existingButton.remove();
    }
    
    const refreshButton = document.createElement('button');
    refreshButton.type = 'button';
    refreshButton.className = 'refresh-models-btn btn btn-outline-secondary btn-sm';
    refreshButton.innerHTML = '🔄 Refresh';
    refreshButton.title = 'Refresh models list';
    refreshButton.style.marginLeft = '10px';
    
    refreshButton.addEventListener('click', async () => {
        try {
            refreshButton.disabled = true;
            refreshButton.innerHTML = '🔄 Refreshing...';
            
            const response = await fetch('/api/models/refresh');
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to refresh models');
            }
            
            showSuccess('Models list refreshed successfully');
            await loadModels();
            
        } catch (error) {
            console.error('Model refresh error:', error);
            showError(`Model refresh error: ${error.message}`);
        } finally {
            refreshButton.disabled = false;
            refreshButton.innerHTML = '🔄 Refresh';
        }
    });
    
    modelContainer.appendChild(refreshButton);
}

// Load languages function
async function loadLanguages() {
    try {
        console.log('Loading languages from API...');
        
        const response = await fetch('/api/languages');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${data.detail || 'Unknown error'}`);
        }
        
        const sourceLangSelect = document.getElementById('sourceLang');
        const targetLangSelect = document.getElementById('targetLang');
        
        if (!sourceLangSelect || !targetLangSelect) {
            console.error('Language select elements not found');
            return;
        }
        
        const languages = data.languages || {};
        
        // Clear existing options
        sourceLangSelect.innerHTML = '';
        targetLangSelect.innerHTML = '';
        
        // Add language options
        Object.entries(languages).forEach(([langCode, langName]) => {
            const sourceOption = document.createElement('option');
            sourceOption.value = langCode;
            sourceOption.textContent = langName;
            sourceLangSelect.appendChild(sourceOption);
            
            const targetOption = document.createElement('option');
            targetOption.value = langCode;
            targetOption.textContent = langName;
            targetLangSelect.appendChild(targetOption);
        });
        
        // Set default languages
        sourceLangSelect.value = 'en'; // English
        targetLangSelect.value = 'ja'; // Japanese
        
        console.log('Languages loaded successfully:', languages);
        
    } catch (error) {
        console.error('Language loading error:', error);
        showError(`Failed to load languages: ${error.message}`);
    }
}

// New notification function
function showStatus(message, type = 'info') {
    switch(type) {
        case 'error':
            showError(message);
            break;
        case 'success':
            showSuccess(message);
            break;
        case 'warning':
            showWarning(message);
            break;
        case 'info':
        default:
            showInfo(message);
            break;
    }
}

function showInfo(message) {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        console.error('Alerts container not found');
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-info alert-dismissible fade show';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    alertsContainer.appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Page loaded, initializing...');
    
    // Debug: Check element existence
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
    
    // Delay API settings check slightly
    setTimeout(async () => {
        await checkApiSettings();
        
        // Load models and languages only if API settings are complete
        const uploadFormVisible = uploadForm && uploadForm.style.display !== 'none';
        if (uploadFormVisible) {
            await loadModels();
            await loadLanguages();
            addRefreshModelsButton();
        }
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
    
    // File input validation event listener
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', validateFileInput);
    }
    
    // Language selection validation event listener
    const sourceLangSelect = document.getElementById('sourceLang');
    const targetLangSelect = document.getElementById('targetLang');
    if (sourceLangSelect) {
        sourceLangSelect.addEventListener('change', validateLanguageSelection);
    }
    if (targetLangSelect) {
        targetLangSelect.addEventListener('change', validateLanguageSelection);
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
        // Show API settings form on error
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
            
            // Load models and languages after API settings are saved
            await loadModels();
            await loadLanguages();
            addRefreshModelsButton();
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
            
            // Load models and languages after API settings are updated
            await loadModels();
            await loadLanguages();
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
    
    // Send model only if selected (server will auto-select if not provided)
    if (modelSelect?.value) {
        formData.append('model', modelSelect.value);
    }
    
    formData.append('source_lang', sourceLangSelect?.value || 'en');
    formData.append('target_lang', targetLangSelect?.value || 'ja');
    formData.append('client_id', clientId);
    formData.append('ai_instruction', aiInstruction?.value || '');

    console.log('Form data prepared:', {
        filename: fileInput.files[0].name,
        model: modelSelect?.value || 'auto-select',
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
