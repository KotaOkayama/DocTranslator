// ===== グローバル変数 =====
let ws = null;
let uploadInProgress = false;
let translationAborted = false;
let currentTranslationController = null;

const clientId = 'client_' + Math.random().toString(36).substr(2, 9);
console.log('Generated client ID:', clientId);

// 音声合成関連（テキスト翻訳専用）
let speechSynthesis = window.speechSynthesis;
let currentUtterance = null;
let sourceUtterance = null;
let availableVoices = [];

// 言語コードから音声言語コードへのマッピング
const speechLangMap = {
    'en': ['en-US', 'en-GB', 'en-AU', 'en'],
    'ja': ['ja-JP', 'ja'],
    'ko': ['ko-KR', 'ko'],
    'zh': ['zh-CN', 'zh-TW', 'zh-HK', 'zh'],
    'fr': ['fr-FR', 'fr-CA', 'fr-BE', 'fr-CH', 'fr'],
    'de': ['de-DE', 'de-AT', 'de-CH', 'de'],
    'es': ['es-ES', 'es-MX', 'es-AR', 'es-US', 'es'],
    'hi': ['hi-IN', 'hi'],
    'vi': ['vi-VN', 'vi'],
    'th': ['th-TH', 'th']
};

// ===== WebSocket接続管理 =====
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

// ===== ページ初期化 =====
document.addEventListener('DOMContentLoaded', async function() {
    console.log('=== Page Initialization Started ===');
    console.log('Current URL:', window.location.href);
    
    // 1. 要素の存在確認
    const elements = {
        globalModel: document.getElementById('globalModel'),
        sourceLang: document.getElementById('sourceLang'),
        targetLang: document.getElementById('targetLang'),
        docTranslatorTab: document.getElementById('docTranslatorTab'),
        langTranslatorTab: document.getElementById('langTranslatorTab'),
        docTranslatorPanel: document.getElementById('docTranslatorPanel'),
        langTranslatorPanel: document.getElementById('langTranslatorPanel')
    };
    
    console.log('Elements check:', elements);
    
    // 要素が見つからない場合は警告
    for (const [name, element] of Object.entries(elements)) {
        if (!element) {
            console.error(`Element not found: ${name}`);
        }
    }
    
    // 2. WebSocket初期化
    console.log('Initializing WebSocket...');
    initializeWebSocket();
    
    // 3. API設定チェック
    console.log('Checking API settings...');
    try {
        const hasSettings = await checkApiSettings();
        console.log('API settings check result:', hasSettings);
        
        if (hasSettings) {
            // 4. モデルと言語を読み込む
            console.log('Loading models...');
            await loadGlobalModels();
            
            console.log('Loading languages...');
            await loadLanguages();
            await loadTextTranslationLanguages();
        }
    } catch (error) {
        console.error('Initialization error:', error);
    }
    
    // 5. イベントリスナーの設定
    console.log('Setting up event listeners...');
    setupEventListeners();
    
    console.log('=== Page Initialization Completed ===');
});

// イベントリスナーを別関数に分離
function setupEventListeners() {
    // 設定モーダル
    setupSettingsModal();
    
    // フォーム
    const apiSettingsForm = document.getElementById('apiSettingsForm');
    if (apiSettingsForm) {
        apiSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            saveApiSettings();
        });
        console.log('API settings form listener added');
    }
    
    const updateApiSettingsForm = document.getElementById('updateApiSettingsForm');
    if (updateApiSettingsForm) {
        updateApiSettingsForm.addEventListener('submit', function(e) {
            e.preventDefault();
            updateApiSettings();
        });
        console.log('Update API settings form listener added');
    }
    
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            uploadFile();
        });
        console.log('Upload form listener added');
    }
    
    // グローバルモデル更新ボタン
    const refreshGlobalModelsBtn = document.getElementById('refreshGlobalModelsBtn');
    if (refreshGlobalModelsBtn) {
        refreshGlobalModelsBtn.addEventListener('click', async function() {
            await refreshGlobalModels(this);
        });
        console.log('Refresh models button listener added');
    }
    
    // ファイル入力
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', validateFileInput);
        console.log('File input listener added');
    }
    
    // 言語選択
    const sourceLangSelect = document.getElementById('sourceLang');
    const targetLangSelect = document.getElementById('targetLang');
    if (sourceLangSelect) {
        sourceLangSelect.addEventListener('change', validateLanguageSelection);
        console.log('Source language listener added');
    }
    if (targetLangSelect) {
        targetLangSelect.addEventListener('change', validateLanguageSelection);
        console.log('Target language listener added');
    }
    
    // タブボタン
    const docTranslatorTab = document.getElementById('docTranslatorTab');
    const langTranslatorTab = document.getElementById('langTranslatorTab');
    
    if (docTranslatorTab) {
        docTranslatorTab.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('DocTranslator tab clicked');
            switchTab('docTranslator');
        });
        console.log('DocTranslator tab listener added');
    } else {
        console.error('DocTranslator tab element not found!');
    }
    
    if (langTranslatorTab) {
        langTranslatorTab.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('LangTranslator tab clicked');
            switchTab('langTranslator');
        });
        console.log('LangTranslator tab listener added');
    } else {
        console.error('LangTranslator tab element not found!');
    }
}

// ===== 設定モーダル管理 =====
function setupSettingsModal() {
    const settingsButton = document.getElementById('settingsButton');
    const settingsModal = document.getElementById('settingsModal');
    const closeButton = document.getElementById('closeModalButton');
    const closeModalBtn = document.getElementById('closeModalBtn');
    
    if (settingsButton) {
        settingsButton.addEventListener('click', function() {
            if (settingsModal) settingsModal.style.display = 'block';
        });
    }
    
    if (closeButton) {
        closeButton.addEventListener('click', function() {
            if (settingsModal) settingsModal.style.display = 'none';
        });
    }
    
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', function() {
            if (settingsModal) settingsModal.style.display = 'none';
        });
    }
    
    window.addEventListener('click', function(event) {
        if (event.target === settingsModal) {
            settingsModal.style.display = 'none';
        }
    });
}

// ===== API設定管理 =====
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
        const docPanel = document.getElementById('docTranslatorPanel');
        const langPanel = document.getElementById('langTranslatorPanel');
        const tabNavigation = document.querySelector('.tab-navigation');
        const headerToolbar = document.querySelector('.header-toolbar');

        if (data.has_api_settings) {
            console.log('API settings found, showing translation interface');
            if (apiSettingsSection) apiSettingsSection.style.display = 'none';
            if (tabNavigation) tabNavigation.style.display = 'flex';
            if (headerToolbar) headerToolbar.style.display = 'flex';
            
            // デフォルトでDocTranslatorタブを表示
            switchTab('docTranslator');
            
            const modalApiUrl = document.getElementById('modalApiUrl');
            if (modalApiUrl && data.api_url) {
                modalApiUrl.value = data.api_url;
            }
            
            return true;
        } else {
            console.log('API settings not found, showing API settings form');
            if (apiSettingsSection) apiSettingsSection.style.display = 'block';
            if (tabNavigation) tabNavigation.style.display = 'none';
            if (headerToolbar) headerToolbar.style.display = 'none';
            if (docPanel) docPanel.style.display = 'none';
            if (langPanel) langPanel.style.display = 'none';
            
            return false;
        }
    } catch (error) {
        console.error('Error checking API settings:', error);
        const apiSettingsSection = document.getElementById('apiSettingsSection');
        const tabNavigation = document.querySelector('.tab-navigation');
        const headerToolbar = document.querySelector('.header-toolbar');
        if (apiSettingsSection) apiSettingsSection.style.display = 'block';
        if (tabNavigation) tabNavigation.style.display = 'none';
        if (headerToolbar) headerToolbar.style.display = 'none';
        showError('Failed to check API settings status. Please configure your API settings.');
        
        return false;
    }
}

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
            apiKeyInput.value = '';
            apiUrlInput.value = '';
            
            // API設定確認
            await checkApiSettings();
            
            // モデルと言語を読み込む
            await loadGlobalModels();
            await loadLanguages();
            await loadTextTranslationLanguages();
        } else {
            showError(result.detail || 'Failed to save API settings');
        }
    } catch (error) {
        console.error('Error saving API settings:', error);
        showError('Failed to save API settings');
    }
}

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
            
            // モデルと言語を再読み込み
            await loadGlobalModels();
            await loadLanguages();
            await loadTextTranslationLanguages();
        } else {
            showError(result.detail || 'Failed to update API settings');
        }
    } catch (error) {
        console.error('Error updating API settings:', error);
        showError('Failed to update API settings');
    }
}

// ===== グローバルモデル管理 =====
async function loadGlobalModels() {
    try {
        console.log('Loading global models from API...');
        
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${data.detail || 'Unknown error'}`);
        }
        
        const globalModelSelect = document.getElementById('globalModel');
        if (!globalModelSelect) {
            console.error('Global model select element not found');
            return;
        }
        
        // 既存のオプションをクリア
        globalModelSelect.innerHTML = '';
        
        const models = data.models || {};
        
        if (Object.keys(models).length === 0) {
            console.warn('No models available');
            if (data.error) {
                showError(`Failed to load models: ${data.error}`);
            }
            
            const placeholderOption = document.createElement('option');
            placeholderOption.value = '';
            placeholderOption.textContent = 'API configuration required';
            placeholderOption.disabled = true;
            placeholderOption.selected = true;
            globalModelSelect.appendChild(placeholderOption);
            
            const submitButton = document.getElementById('submitButton');
            if (submitButton) {
                submitButton.disabled = true;
                submitButton.title = 'Please complete API configuration';
            }
            
            return;
        }
        
        // モデルをアルファベット順に追加
        Object.entries(models)
            .sort(([a], [b]) => a.localeCompare(b))
            .forEach(([modelId, displayName]) => {
                const option = document.createElement('option');
                option.value = modelId;
                option.textContent = displayName;
                globalModelSelect.appendChild(option);
            });
        
        // 最初のモデルを選択
        const sortedModelIds = Object.keys(models).sort();
        if (sortedModelIds.length > 0) {
            globalModelSelect.value = sortedModelIds[0];
            console.log(`Default global model selected: ${sortedModelIds[0]}`);
        }
        
        // 送信ボタンを有効化
        const submitButton = document.getElementById('submitButton');
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.title = '';
        }
        
        console.log('Global models loaded successfully:', sortedModelIds);
        showSuccess('Models loaded successfully');
        
    } catch (error) {
        console.error('Global model loading error:', error);
        showError(`Failed to load models: ${error.message}`);
        
        const globalModelSelect = document.getElementById('globalModel');
        if (globalModelSelect) {
            globalModelSelect.innerHTML = '';
            const errorOption = document.createElement('option');
            errorOption.value = '';
            errorOption.textContent = 'Model loading error';
            errorOption.disabled = true;
            errorOption.selected = true;
            globalModelSelect.appendChild(errorOption);
        }
        
        const submitButton = document.getElementById('submitButton');
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.title = 'Please check API settings';
        }
    }
}

async function refreshGlobalModels(button) {
    try {
        button.disabled = true;
        const icon = button.querySelector('i');
        if (icon) icon.classList.add('fa-spin');
        
        const response = await fetch('/api/models/refresh');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to refresh models');
        }
        
        showSuccess('Models list refreshed successfully');
        await loadGlobalModels();
        
    } catch (error) {
        console.error('Model refresh error:', error);
        showError(`Model refresh error: ${error.message}`);
    } finally {
        button.disabled = false;
        const icon = button.querySelector('i');
        if (icon) icon.classList.remove('fa-spin');
    }
}

// ===== 言語管理 =====
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
        
        // 既存のオプションをクリア
        sourceLangSelect.innerHTML = '';
        targetLangSelect.innerHTML = '';
        
        // 言語オプションを追加
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
        
        // デフォルト値を設定
        sourceLangSelect.value = 'en';
        targetLangSelect.value = 'ja';
        
        console.log('Languages loaded successfully:', languages);
        
    } catch (error) {
        console.error('Language loading error:', error);
        showError(`Failed to load languages: ${error.message}`);
    }
}

// ===== ファイルアップロードと翻訳（ドキュメント翻訳専用） =====
async function uploadFile() {
    console.log('[Document Translation] Starting file upload...');
    const fileInput = document.getElementById('file');
    const globalModelSelect = document.getElementById('globalModel');
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

    translationAborted = false;
    currentTranslationController = new AbortController();
    const signal = currentTranslationController.signal;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    if (globalModelSelect?.value) {
        formData.append('model', globalModelSelect.value);
    }
    
    formData.append('source_lang', sourceLangSelect?.value || 'en');
    formData.append('target_lang', targetLangSelect?.value || 'ja');
    formData.append('client_id', clientId);
    formData.append('ai_instruction', aiInstruction?.value || '');

    console.log('[Document Translation] Form data prepared:', {
        filename: fileInput.files[0].name,
        model: globalModelSelect?.value || 'auto-select',
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
        
        if (cancelButton) cancelButton.style.display = 'inline-flex';

        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.log('Reconnecting WebSocket before upload...');
            initializeWebSocket();
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        console.log('[Document Translation] Sending translation request...');
        const response = await fetch('/api/translate', {
            method: 'POST',
            body: formData,
            signal: signal
        });

        console.log('[Document Translation] Translation response received:', response.status);

        if (translationAborted) {
            console.log('[Document Translation] Translation was aborted');
            return;
        }

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Translation failed');
        }

        const result = await response.json();
        console.log('[Document Translation] Translation result:', result);
        
        if (result.warning) {
            showWarning(result.warning);
        }

        let downloadHtml = '';
        
        if (result.download_url) {
            downloadHtml += `
                <a href="${result.download_url}" class="download-btn" download>
                    <i class="fas fa-download"></i>
                    <span>Download Translated File</span>
                </a>
            `;
        }
        
        if (result.extracted_text_url) {
            downloadHtml += `
                <a href="${result.extracted_text_url}" class="download-btn secondary" download>
                    <i class="fas fa-file-alt"></i>
                    <span>Download Extracted Text</span>
                </a>
            `;
        }
        
        if (result.translated_text_url) {
            downloadHtml += `
                <a href="${result.translated_text_url}" class="download-btn secondary" download>
                    <i class="fas fa-language"></i>
                    <span>Download Translated Text</span>
                </a>
            `;
        }

        const downloadLinks = document.getElementById('downloadLinks');
        if (downloadLinks) {
            downloadLinks.innerHTML = downloadHtml;
        }
        
        if (downloadSection) downloadSection.style.display = 'block';
        if (progressSection) progressSection.style.display = 'none';
        
        showSuccess('Translation completed successfully');
        
        if (cancelButton) cancelButton.style.display = 'none';

    } catch (error) {
        if (error.name === 'AbortError') {
            console.log('[Document Translation] Fetch aborted');
            return;
        }
        
        console.error('[Document Translation] Translation error:', error);
        showError(error.message || 'Translation failed');
        if (progressBar) progressBar.classList.remove('progress-bar-animated');
        
        if (cancelButton) cancelButton.style.display = 'none';
    } finally {
        uploadInProgress = false;
        if (!translationAborted) {
            currentTranslationController = null;
        }
    }
}

// ===== 進捗管理 =====
function updateProgress(progress, message) {
    if (translationAborted) {
        return;
    }
    
    console.log('Updating progress:', progress, message);
    const progressBar = document.getElementById('progressBar');
    const progressPercentage = document.getElementById('progressPercentage');
    const progressText = document.getElementById('progressText');
    
    const percentage = Math.round(progress * 100);
    
    if (progressBar) {
        progressBar.style.width = `${percentage}%`;
        progressBar.setAttribute('aria-valuenow', percentage);
    }
    
    if (progressPercentage) progressPercentage.textContent = `${percentage}%`;
    
    if (progressText) progressText.textContent = message || '';
    
    if (progressBar) {
        if (percentage < 100) {
            progressBar.classList.add('progress-bar-animated');
        } else {
            progressBar.classList.remove('progress-bar-animated');
            
            const cancelButton = document.getElementById('cancelButton');
            if (cancelButton) cancelButton.style.display = 'none';
        }
    }

    console.log(`Progress updated to: ${percentage}% - ${message}`);
}

// ===== 翻訳キャンセル =====
function cancelTranslation() {
    console.log('[Document Translation] Translation cancellation requested');
    translationAborted = true;
    
    if (currentTranslationController) {
        currentTranslationController.abort();
    }
    
    const cancelButton = document.getElementById('cancelButton');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    
    if (cancelButton) {
        cancelButton.disabled = true;
        cancelButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Cancelling...';
    }
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    
    if (progressBar) progressBar.classList.remove('progress-bar-animated');
    if (progressText) progressText.textContent = 'Translation cancelled';
    
    setTimeout(() => {
        resetTranslationUI();
        showWarning('Translation was cancelled by user');
    }, 1000);
}

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
        cancelButton.innerHTML = '<i class="fas fa-times"></i> Cancel Translation';
    }
    if (progressSection) progressSection.style.display = 'none';
    if (downloadSection) downloadSection.style.display = 'none';
    
    uploadInProgress = false;
    translationAborted = false;
    currentTranslationController = null;
    
    initializeWebSocket();
}

// ===== バリデーション =====
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

function validateLanguageSelection() {
    const sourceLang = document.getElementById('sourceLang')?.value;
    const targetLang = document.getElementById('targetLang')?.value;
    const submitButton = document.getElementById('submitButton');
    
    if (sourceLang === targetLang) {
        showError('Source and target languages must be different');
        if (submitButton) submitButton.disabled = true;
    } else {
        const fileInput = document.getElementById('file');
        if (fileInput && fileInput.files.length > 0 && submitButton) {
            submitButton.disabled = false;
        }
    }
}

// ===== 通知機能 =====
function showError(message) {
    showAlert(message, 'danger', 'fas fa-exclamation-circle');
}

function showSuccess(message) {
    showAlert(message, 'success', 'fas fa-check-circle');
}

function showWarning(message) {
    showAlert(message, 'warning', 'fas fa-exclamation-triangle');
}

function showInfo(message) {
    showAlert(message, 'info', 'fas fa-info-circle');
}

function showAlert(message, type, icon) {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) {
        console.error('Alerts container not found');
        return;
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `
        <i class="${icon}"></i>
        <span>${message}</span>
        <button type="button" class="btn-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    alertsContainer.appendChild(alertDiv);
    
    setTimeout(() => {
        alertDiv.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
}

// ===== タブ切り替え機能 =====
function switchTab(tabName) {
    console.log('Switching to tab:', tabName);
    
    const docTab = document.getElementById('docTranslatorTab');
    const langTab = document.getElementById('langTranslatorTab');
    const docPanel = document.getElementById('docTranslatorPanel');
    const langPanel = document.getElementById('langTranslatorPanel');
    
    if (!docTab || !langTab || !docPanel || !langPanel) {
        console.error('Tab elements not found');
        return;
    }
    
    if (tabName === 'docTranslator') {
        // DocTranslatorタブをアクティブに
        docTab.classList.add('active');
        docTab.setAttribute('aria-selected', 'true');
        langTab.classList.remove('active');
        langTab.setAttribute('aria-selected', 'false');
        
        // DocTranslatorパネルを表示
        docPanel.classList.add('active');
        docPanel.style.display = 'block';
        langPanel.classList.remove('active');
        langPanel.style.display = 'none';
        
        console.log('DocTranslator tab activated');
    } else if (tabName === 'langTranslator') {
        // LangTranslatorタブをアクティブに
        langTab.classList.add('active');
        langTab.setAttribute('aria-selected', 'true');
        docTab.classList.remove('active');
        docTab.setAttribute('aria-selected', 'false');
        
        // LangTranslatorパネルを表示
        langPanel.classList.add('active');
        langPanel.style.display = 'block';
        docPanel.classList.remove('active');
        docPanel.style.display = 'none';
        
        // LangTranslatorの初期化（初回のみ）
        if (!langPanel.dataset.initialized) {
            console.log('Initializing LangTranslator for the first time');
            initializeLangTranslator();
            langPanel.dataset.initialized = 'true';
        }
        
        console.log('LangTranslator tab activated');
    }
}

// ===== LangTranslator初期化（音声・履歴機能付き） =====
function initializeLangTranslator() {
    console.log('[Text Translation] Initializing LangTranslator...');
    
    // 言語リストを読み込む
    loadTextTranslationLanguages();
    
    // 音声合成の初期化（テキスト翻訳専用）
    initializeSpeechSynthesis();
    
    // 翻訳履歴の読み込み（テキスト翻訳専用）
    loadTranslationHistory();
    
    // 入力テキストの変更監視（言語自動検出用）
    const inputText = document.getElementById('inputText');
    if (inputText) {
        inputText.addEventListener('input', onInputTextChange);
    }
    
    // 出力テキストの変更監視
    const outputText = document.getElementById('outputText');
    if (outputText) {
        outputText.addEventListener('input', updateOutputCharCount);
    }
    
    // 言語選択変更時の音声ボタン状態更新
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    if (textSourceLang) {
        textSourceLang.addEventListener('change', updateSpeechButtonStates);
    }
    if (textTargetLang) {
        textTargetLang.addEventListener('change', updateSpeechButtonStates);
    }
    
    // テキスト翻訳フォームのサブミットイベント
    const textTranslationForm = document.getElementById('textTranslationForm');
    if (textTranslationForm) {
        textTranslationForm.addEventListener('submit', function(e) {
            e.preventDefault();
            translateText();
        });
    }
    
    // 初期文字数カウント
    updateInputCharCount();
    updateOutputCharCount();
    
    console.log('[Text Translation] LangTranslator initialized successfully');
}

// ===== テキスト翻訳用の言語読み込み =====
async function loadTextTranslationLanguages() {
    try {
        console.log('[Text Translation] Loading languages...');
        const response = await fetch('/api/languages');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${data.detail || 'Unknown error'}`);
        }
        
        const textSourceLang = document.getElementById('textSourceLang');
        const textTargetLang = document.getElementById('textTargetLang');
        
        if (!textSourceLang || !textTargetLang) {
            console.error('Text language select elements not found');
            return;
        }
        
        const languages = data.languages || {};
        
        // 既存のオプションをクリア
        textSourceLang.innerHTML = '';
        textTargetLang.innerHTML = '';
        
        // 言語オプションを追加
        Object.entries(languages).forEach(([langCode, langName]) => {
            const sourceOption = document.createElement('option');
            sourceOption.value = langCode;
            sourceOption.textContent = langName;
            textSourceLang.appendChild(sourceOption);
            
            const targetOption = document.createElement('option');
            targetOption.value = langCode;
            targetOption.textContent = langName;
            textTargetLang.appendChild(targetOption);
        });
        
        // デフォルト値を設定
        textSourceLang.value = 'en';
        textTargetLang.value = 'ja';
        
        console.log('[Text Translation] Languages loaded successfully');
        
    } catch (error) {
        console.error('[Text Translation] Language loading error:', error);
        showError(`Failed to load languages: ${error.message}`);
    }
}

// ===== 音声合成機能（テキスト翻訳専用）- 修正版 =====

/**
 * 音声合成がサポートされているかチェック
 */
function checkSpeechSynthesisSupport() {
    return 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
}

/**
 * 利用可能な音声を取得
 */
function loadVoices() {
    if (!checkSpeechSynthesisSupport()) {
        console.log('[Text Translation] Speech synthesis not supported');
        return;
    }

    availableVoices = speechSynthesis.getVoices();
    
    console.log('[Text Translation] Available voices:', availableVoices.map(v => ({
        name: v.name,
        lang: v.lang,
        default: v.default,
        localService: v.localService
    })));

    // 音声ボタンの状態を更新
    updateSpeechButtonStates();
}

/**
 * 指定された言語に最適な音声を見つける
 */
function findBestVoice(langCode) {
    if (!availableVoices.length) return null;

    const candidates = speechLangMap[langCode] || [langCode];
    
    // 各候補言語コードで音声を検索
    for (const candidate of candidates) {
        // 完全一致を優先
        let voice = availableVoices.find(v => v.lang === candidate);
        if (voice) {
            console.log(`[Text Translation] Found exact match voice: ${voice.name} (${voice.lang})`);
            return voice;
        }
        
        // 部分一致
        voice = availableVoices.find(v => v.lang.startsWith(candidate.split('-')[0]));
        if (voice) {
            console.log(`[Text Translation] Found partial match voice: ${voice.name} (${voice.lang})`);
            return voice;
        }
    }

    // フォールバック：デフォルト音声
    const fallbackVoice = availableVoices.find(v => v.default) || availableVoices[0];
    console.log(`[Text Translation] Using fallback voice: ${fallbackVoice?.name} (${fallbackVoice?.lang})`);
    return fallbackVoice;
}

/**
 * 音声ボタンの状態を更新（テキスト翻訳専用）
 */
function updateSpeechButtonStates() {
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    
    if (!textSourceLang || !textTargetLang) return;
    
    const sourceLang = textSourceLang.value;
    const targetLang = textTargetLang.value;

    // ソース言語の音声チェック
    const sourceVoice = findBestVoice(sourceLang);
    const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
    const sourceErrorSpan = document.getElementById('sourceSpeechError');
    
    if (sourceSpeakBtn && sourceErrorSpan) {
        if (sourceVoice && checkSpeechSynthesisSupport()) {
            sourceSpeakBtn.classList.remove('btn-disabled');
            sourceSpeakBtn.disabled = false;
            sourceErrorSpan.style.display = 'none';
        } else {
            sourceSpeakBtn.classList.add('btn-disabled');
            sourceSpeakBtn.disabled = true;
            sourceErrorSpan.style.display = 'inline';
            console.log(`[Text Translation] No voice available for source language: ${sourceLang}`);
        }
    }

    // ターゲット言語の音声チェック
    const targetVoice = findBestVoice(targetLang);
    const targetSpeakBtn = document.getElementById('targetSpeakButton');
    const targetErrorSpan = document.getElementById('targetSpeechError');
    
    if (targetSpeakBtn && targetErrorSpan) {
        if (targetVoice && checkSpeechSynthesisSupport()) {
            targetSpeakBtn.classList.remove('btn-disabled');
            targetSpeakBtn.disabled = false;
            targetErrorSpan.style.display = 'none';
        } else {
            targetSpeakBtn.classList.add('btn-disabled');
            targetSpeakBtn.disabled = true;
            targetErrorSpan.style.display = 'inline';
            console.log(`[Text Translation] No voice available for target language: ${targetLang}`);
        }
    }
}

/**
 * 音声合成の初期化（テキスト翻訳専用）
 */
function initializeSpeechSynthesis() {
    if (!checkSpeechSynthesisSupport()) {
        console.warn('[Text Translation] Speech synthesis is not supported in this browser');
        return;
    }

    console.log('[Text Translation] Initializing speech synthesis...');
    loadVoices();
    
    if (speechSynthesis.onvoiceschanged !== undefined) {
        speechSynthesis.onvoiceschanged = loadVoices;
    }

    // 定期的に音声リストを更新
    setTimeout(() => {
        if (availableVoices.length === 0) {
            console.log('[Text Translation] Retrying voice loading...');
            loadVoices();
        }
    }, 1000);
}

/**
 * ソーステキストの読み上げ（テキスト翻訳専用）- 修正版
 */
function speakSourceText() {
    const inputText = document.getElementById('inputText');
    if (!inputText || !inputText.value) {
        showWarning('No text to speak');
        return;
    }

    const text = inputText.value;
    const textSourceLang = document.getElementById('textSourceLang');
    if (!textSourceLang) return;
    
    const sourceLang = textSourceLang.value;
    const voice = findBestVoice(sourceLang);
    
    if (!voice) {
        showWarning(`Voice not available for ${sourceLang}`);
        return;
    }

    // 既存の読み上げを停止
    if (sourceUtterance) {
        sourceUtterance.manualStop = true;
        speechSynthesis.cancel();
        sourceUtterance = null;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    // 手動停止フラグ
    utterance.manualStop = false;
    
    utterance.onend = function() {
        const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
        const sourceStopBtn = document.getElementById('sourceStopButton');
        if (sourceSpeakBtn) sourceSpeakBtn.style.display = 'inline-block';
        if (sourceStopBtn) sourceStopBtn.style.display = 'none';
        sourceUtterance = null;
        console.log('[Text Translation] Source speech ended');
    };

    utterance.onerror = function(event) {
        // 手動停止の場合はエラーを表示しない
        if (utterance.manualStop || event.error === 'interrupted' || event.error === 'canceled') {
            console.log('[Text Translation] Source speech manually stopped');
            const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
            const sourceStopBtn = document.getElementById('sourceStopButton');
            if (sourceSpeakBtn) sourceSpeakBtn.style.display = 'inline-block';
            if (sourceStopBtn) sourceStopBtn.style.display = 'none';
            sourceUtterance = null;
            return;
        }
        
        console.error('[Text Translation] Speech synthesis error:', event);
        showError('Error during speech synthesis');
        
        const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
        const sourceStopBtn = document.getElementById('sourceStopButton');
        if (sourceSpeakBtn) sourceSpeakBtn.style.display = 'inline-block';
        if (sourceStopBtn) sourceStopBtn.style.display = 'none';
        sourceUtterance = null;
    };

    speechSynthesis.speak(utterance);
    sourceUtterance = utterance;
    
    const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
    const sourceStopBtn = document.getElementById('sourceStopButton');
    if (sourceSpeakBtn) sourceSpeakBtn.style.display = 'none';
    if (sourceStopBtn) sourceStopBtn.style.display = 'inline-block';

    console.log(`[Text Translation] Speaking source text with voice: ${voice.name} (${voice.lang})`);
}

/**
 * ソーステキストの読み上げ停止 - 修正版
 */
function stopSourceSpeaking() {
    if (speechSynthesis && sourceUtterance) {
        // 手動停止フラグを設定
        sourceUtterance.manualStop = true;
        
        // 読み上げを停止
        speechSynthesis.cancel();
        
        // UI更新
        const sourceSpeakBtn = document.getElementById('sourceSpeakButton');
        const sourceStopBtn = document.getElementById('sourceStopButton');
        if (sourceSpeakBtn) sourceSpeakBtn.style.display = 'inline-block';
        if (sourceStopBtn) sourceStopBtn.style.display = 'none';
        
        sourceUtterance = null;
        console.log('[Text Translation] Source speech stopped manually');
    }
}

/**
 * ターゲットテキストの読み上げ（テキスト翻訳専用）- 修正版
 */
function speakTargetText() {
    const outputText = document.getElementById('outputText');
    if (!outputText || !outputText.value) {
        showWarning('No translation to speak');
        return;
    }

    const text = outputText.value;
    const textTargetLang = document.getElementById('textTargetLang');
    if (!textTargetLang) return;
    
    const targetLang = textTargetLang.value;
    const voice = findBestVoice(targetLang);
    
    if (!voice) {
        showWarning(`Voice not available for ${targetLang}`);
        return;
    }

    // 既存の読み上げを停止
    if (currentUtterance) {
        currentUtterance.manualStop = true;
        speechSynthesis.cancel();
        currentUtterance = null;
    }

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = voice;
    utterance.lang = voice.lang;
    utterance.rate = 0.9;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    
    // 手動停止フラグ
    utterance.manualStop = false;
    
    utterance.onend = function() {
        const targetSpeakBtn = document.getElementById('targetSpeakButton');
        const targetStopBtn = document.getElementById('targetStopButton');
        if (targetSpeakBtn) targetSpeakBtn.style.display = 'inline-block';
        if (targetStopBtn) targetStopBtn.style.display = 'none';
        currentUtterance = null;
        console.log('[Text Translation] Target speech ended');
    };

    utterance.onerror = function(event) {
        // 手動停止の場合はエラーを表示しない
        if (utterance.manualStop || event.error === 'interrupted' || event.error === 'canceled') {
            console.log('[Text Translation] Target speech manually stopped');
            const targetSpeakBtn = document.getElementById('targetSpeakButton');
            const targetStopBtn = document.getElementById('targetStopButton');
            if (targetSpeakBtn) targetSpeakBtn.style.display = 'inline-block';
            if (targetStopBtn) targetStopBtn.style.display = 'none';
            currentUtterance = null;
            return;
        }
        
        console.error('[Text Translation] Speech synthesis error:', event);
        showError('Error during speech synthesis');
        
        const targetSpeakBtn = document.getElementById('targetSpeakButton');
        const targetStopBtn = document.getElementById('targetStopButton');
        if (targetSpeakBtn) targetSpeakBtn.style.display = 'inline-block';
        if (targetStopBtn) targetStopBtn.style.display = 'none';
        currentUtterance = null;
    };

    speechSynthesis.speak(utterance);
    currentUtterance = utterance;
    
    const targetSpeakBtn = document.getElementById('targetSpeakButton');
    const targetStopBtn = document.getElementById('targetStopButton');
    if (targetSpeakBtn) targetSpeakBtn.style.display = 'none';
    if (targetStopBtn) targetStopBtn.style.display = 'inline-block';

    console.log(`[Text Translation] Speaking target text with voice: ${voice.name} (${voice.lang})`);
}

/**
 * ターゲットテキストの読み上げ停止 - 修正版
 */
function stopTargetSpeaking() {
    if (speechSynthesis && currentUtterance) {
        // 手動停止フラグを設定
        currentUtterance.manualStop = true;
        
        // 読み上げを停止
        speechSynthesis.cancel();
        
        // UI更新
        const targetSpeakBtn = document.getElementById('targetSpeakButton');
        const targetStopBtn = document.getElementById('targetStopButton');
        if (targetSpeakBtn) targetSpeakBtn.style.display = 'inline-block';
        if (targetStopBtn) targetStopBtn.style.display = 'none';
        
        currentUtterance = null;
        console.log('[Text Translation] Target speech stopped manually');
    }
}

// ページ遷移時に読み上げを停止
window.addEventListener('beforeunload', function() {
    if (sourceUtterance) {
        sourceUtterance.manualStop = true;
    }
    if (currentUtterance) {
        currentUtterance.manualStop = true;
    }
    speechSynthesis.cancel();
});

// ===== 言語自動検出機能（テキスト翻訳専用） =====

/**
 * テキストから言語を自動検出（クライアント側）
 */
function detectLanguageClient(text) {
    if (!text) return 'en';
    
    const languagePatterns = {
        'ja': /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]/g,
        'ko': /[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]/g,
        'zh': /[\u4E00-\u9FFF]/g,
        'hi': /[\u0900-\u097F]/g,
        'th': /[\u0E00-\u0E7F]/g,
        'vi': /[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]/gi,
        'fr': /[àâäéèêëïîôöùûüÿç]/gi,
        'de': /[äöüßÄÖÜ]/g,
        'es': /[ñáéíóúü¿¡]/gi
    };
    
    const totalChars = text.replace(/\s/g, '').length;
    if (totalChars === 0) return 'en';
    
    for (const [langCode, pattern] of Object.entries(languagePatterns)) {
        const matches = text.match(pattern);
        const charCount = matches ? matches.length : 0;
        
        if (charCount / totalChars >= 0.3) {
            console.log(`[Text Translation] Detected language: ${langCode} (${(charCount/totalChars*100).toFixed(1)}%)`);
            return langCode;
        }
    }
    
    console.log('[Text Translation] No language detected, defaulting to en');
    return 'en';
}

/**
 * 入力テキスト変更時の言語自動検出（テキスト翻訳専用）
 */
function onInputTextChange() {
    const inputText = document.getElementById('inputText');
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    const autoDetected = document.getElementById('autoDetected');
    
    if (!inputText || !textSourceLang || !textTargetLang) return;
    
    const text = inputText.value.trim();
    
    // 文字数カウンター更新
    updateInputCharCount();
    
    if (text) {
        // 言語検出（クライアント側で高速処理）
        const detectedLang = detectLanguageClient(text);
        
        if (textSourceLang.value !== detectedLang) {
            console.log(`[Text Translation] Auto-detected language change: ${textSourceLang.value} -> ${detectedLang}`);
            textSourceLang.value = detectedLang;
            
            // ターゲット言語を自動調整
            textTargetLang.value = detectedLang === 'en' ? 'ja' : 'en';
            
            // 音声ボタンの状態を更新
            updateSpeechButtonStates();
            
            // 自動検出表示
            if (autoDetected) {
                const langName = textSourceLang.options[textSourceLang.selectedIndex].text;
                autoDetected.textContent = `(Auto-detected: ${langName})`;
                autoDetected.style.display = 'inline';
            }
        }
    } else {
        if (autoDetected) {
            autoDetected.style.display = 'none';
        }
    }
}

// ===== 文字数カウント =====

function updateInputCharCount() {
    const inputText = document.getElementById('inputText');
    const charCount = document.getElementById('inputCharCount');
    if (inputText && charCount) {
        const count = inputText.value.length;
        charCount.textContent = `(${count} characters)`;
    }
}

function updateOutputCharCount() {
    const outputText = document.getElementById('outputText');
    const charCount = document.getElementById('outputCharCount');
    if (outputText && charCount) {
        const count = outputText.value.length;
        charCount.textContent = `(${count} characters)`;
    }
}

// ===== テキスト翻訳実行（言語自動検出付き） =====

async function translateText() {
    const inputText = document.getElementById('inputText');
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    const globalModel = document.getElementById('globalModel');
    const outputText = document.getElementById('outputText');
    const translateButton = document.getElementById('textTranslateButton');
    
    if (!inputText.value.trim()) {
        showError('Please enter text to translate');
        return;
    }
    
    if (textSourceLang.value === textTargetLang.value) {
        showError('Source and target languages must be different');
        return;
    }
    
    try {
        console.log('[Text Translation] Starting translation...');
        translateButton.disabled = true;
        translateButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Translating...';
        
        outputText.value = 'Translating...';
        
        const formData = new FormData();
        formData.append('text', inputText.value);
        formData.append('source_lang', textSourceLang.value);
        formData.append('target_lang', textTargetLang.value);
        formData.append('auto_detect', 'true');  // 自動検出を有効化
        if (globalModel && globalModel.value) {
            formData.append('model', globalModel.value);
        }
        
        const response = await fetch('/api/translate-text', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Translation failed');
        }
        
        outputText.value = result.translated;
        updateOutputCharCount();
        
        // 自動検出された場合の表示更新
        if (result.auto_detected && result.detected_lang) {
            const autoDetected = document.getElementById('autoDetected');
            if (autoDetected) {
                autoDetected.textContent = `(Auto-detected: ${result.source_lang})`;
                autoDetected.style.display = 'inline';
            }
            
            // 言語選択を更新
            textSourceLang.value = result.detected_lang;
            
            // ターゲット言語も更新されている可能性がある
            const targetLangCode = getLanguageCodeFromName(result.target_lang);
            if (targetLangCode) {
                textTargetLang.value = targetLangCode;
            }
            
            // 音声ボタンの状態を更新
            updateSpeechButtonStates();
            
            showInfo(`Language auto-detected: ${result.source_lang}`);
        }
        
        // 履歴を更新
        loadTranslationHistory();
        
        showSuccess('Translation completed successfully');
        console.log('[Text Translation] Translation completed');
        
    } catch (error) {
        console.error('[Text Translation] Translation error:', error);
        showError(error.message || 'Translation failed');
        outputText.value = '';
    } finally {
        translateButton.disabled = false;
        translateButton.innerHTML = '<i class="fas fa-language"></i> Translate Text';
    }
}

// ===== クリップボード操作 =====
async function pasteFromClipboard() {
    try {
        const text = await navigator.clipboard.readText();
        const inputText = document.getElementById('inputText');
        if (inputText) {
            inputText.value = text;
            updateInputCharCount();
            
            // 言語自動検出をトリガー
            onInputTextChange();
            
            showSuccess('Text pasted from clipboard');
        }
    } catch (error) {
        console.error('[Text Translation] Paste error:', error);
        showError('Failed to paste from clipboard. Please use Ctrl+V or Cmd+V.');
    }
}

async function copyToClipboard() {
    const outputText = document.getElementById('outputText');
    if (outputText && outputText.value) {
        try {
            await navigator.clipboard.writeText(outputText.value);
            showSuccess('Translation copied to clipboard');
        } catch (error) {
            console.error('[Text Translation] Copy error:', error);
            showError('Failed to copy to clipboard. Please use Ctrl+C or Cmd+C.');
        }
    } else {
        showError('No translation to copy');
    }
}

// ===== テキストクリア =====
function clearInputText() {
    const inputText = document.getElementById('inputText');
    const autoDetected = document.getElementById('autoDetected');
    if (inputText) {
        inputText.value = '';
        updateInputCharCount();
    }
    if (autoDetected) {
        autoDetected.style.display = 'none';
    }
    stopSourceSpeaking();
}

function clearOutputText() {
    const outputText = document.getElementById('outputText');
    if (outputText) {
        outputText.value = '';
        updateOutputCharCount();
    }
    stopTargetSpeaking();
}

// ===== 言語スワップ =====
function swapLanguages() {
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const autoDetected = document.getElementById('autoDetected');
    
    if (textSourceLang && textTargetLang) {
        const temp = textSourceLang.value;
        textSourceLang.value = textTargetLang.value;
        textTargetLang.value = temp;
        
        if (inputText && outputText) {
            const tempText = inputText.value;
            inputText.value = outputText.value;
            outputText.value = tempText;
            
            updateInputCharCount();
            updateOutputCharCount();
        }
        
        // 音声ボタンの状態を更新
        updateSpeechButtonStates();
        
        // 自動検出表示をクリア
        if (autoDetected) {
            autoDetected.style.display = 'none';
        }
        
        showInfo('Languages swapped');
    }
}

// ===== 翻訳履歴（テキスト翻訳専用） =====

/**
 * 翻訳履歴を読み込む
 */
async function loadTranslationHistory() {
    try {
        console.log('[Text Translation] Loading translation history...');
        const response = await fetch('/api/text-translation-history');
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.detail || 'Failed to load history');
        }
        
        const historyTableBody = document.getElementById('historyTableBody');
        if (!historyTableBody) return;
        
        historyTableBody.innerHTML = '';
        
        if (!data.entries || data.entries.length === 0) {
            historyTableBody.innerHTML = '<tr><td colspan="6" class="no-history">No translation history yet</td></tr>';
            return;
        }
        
        console.log(`[Text Translation] Loaded ${data.entries.length} history entries`);
        data.entries.forEach(entry => {
            addToHistoryTable(entry);
        });
        
    } catch (error) {
        console.error('[Text Translation] Error loading translation history:', error);
    }
}

/**
 * 履歴テーブルにエントリーを追加
 */
function addToHistoryTable(entry) {
    const historyTableBody = document.getElementById('historyTableBody');
    if (!historyTableBody) return;
    
    const row = historyTableBody.insertRow(0);
    row.classList.add('clickable-row');
    
    // クリックイベント
    row.addEventListener('click', function() {
        loadTranslationFromHistory(entry);
    });
    
    const timestamp = new Date(entry.timestamp).toLocaleString();
    
    // 自動検出マークを追加
    const sourceLangDisplay = entry.source_lang + (entry.auto_detected ? ' 🔍' : '');
    
    const cells = [
        timestamp,
        sourceLangDisplay,
        entry.target_lang,
        entry.model,
        entry.source_text,
        entry.translated_text
    ];
    
    cells.forEach(text => {
        const cell = row.insertCell();
        cell.textContent = text;
        cell.title = text;
    });
}

/**
 * 履歴から翻訳を読み込む
 */
function loadTranslationFromHistory(entry) {
    const inputText = document.getElementById('inputText');
    const outputText = document.getElementById('outputText');
    const textSourceLang = document.getElementById('textSourceLang');
    const textTargetLang = document.getElementById('textTargetLang');
    
    if (!inputText || !outputText || !textSourceLang || !textTargetLang) return;
    
    console.log('[Text Translation] Loading translation from history:', entry.id);
    
    // テキストを設定
    inputText.value = entry.source_text;
    outputText.value = entry.translated_text;
    
    // 文字数カウンター更新
    updateInputCharCount();
    updateOutputCharCount();
    
    // 言語設定を更新
    const sourceLangCode = getLanguageCodeFromName(entry.source_lang);
    const targetLangCode = getLanguageCodeFromName(entry.target_lang);
    
    if (sourceLangCode) textSourceLang.value = sourceLangCode;
    if (targetLangCode) textTargetLang.value = targetLangCode;
    
    // 音声ボタンの状態を更新
    updateSpeechButtonStates();
    
    // 選択された行をハイライト
    highlightSelectedRow(entry);
    
    // 自動検出表示
    const autoDetected = document.getElementById('autoDetected');
    if (autoDetected && entry.auto_detected) {
        autoDetected.textContent = `(Auto-detected: ${entry.source_lang})`;
        autoDetected.style.display = 'inline';
    } else if (autoDetected) {
        autoDetected.style.display = 'none';
    }
}

/**
 * 言語名からコードを取得
 */
function getLanguageCodeFromName(langName) {
    // 自動検出マークを削除
    langName = langName.replace(' 🔍', '').trim();
    
    const languages = {
        'English': 'en',
        'Japanese': 'ja',
        'Korean': 'ko',
        'Chinese': 'zh',
        'French': 'fr',
        'German': 'de',
        'Spanish': 'es',
        'Hindi': 'hi',
        'Vietnamese': 'vi',
        'Thai': 'th'
    };
    return languages[langName];
}

/**
 * 選択された行をハイライト
 */
function highlightSelectedRow(selectedEntry) {
    const rows = document.querySelectorAll('#historyTableBody tr');
    rows.forEach(row => {
        row.classList.remove('table-primary');
    });
    
    rows.forEach(row => {
        const sourceText = row.cells[4]?.textContent;
        const translatedText = row.cells[5]?.textContent;
        if (sourceText === selectedEntry.source_text && 
            translatedText === selectedEntry.translated_text) {
            row.classList.add('table-primary');
        }
    });
}

/**
 * 翻訳履歴をクリア
 */
async function clearHistory() {
    if (!confirm('Are you sure you want to clear all translation history?')) {
        return;
    }
    
    try {
        console.log('[Text Translation] Clearing translation history...');
        const response = await fetch('/api/clear-text-translation-history', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || 'Failed to clear history');
        }
        
        loadTranslationHistory();
        showSuccess('Translation history cleared');
        
    } catch (error) {
        console.error('[Text Translation] Clear history error:', error);
        showError('Failed to clear history');
    }
}

/**
 * 翻訳履歴をエクスポート
 */
async function exportHistory() {
    try {
        console.log('[Text Translation] Exporting translation history...');
        const response = await fetch('/api/export-text-translation-history');
        
        if (!response.ok) {
            throw new Error('Failed to export history');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'text_translation_history.csv';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showSuccess('History exported successfully');
        
    } catch (error) {
        console.error('[Text Translation] Export history error:', error);
        showError('Failed to export history');
    }
}

// ===== ユーティリティ関数 =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
