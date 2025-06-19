# API Documentation

## 🔑 Authentication

### API Key
- All API requests require a GenAI Hub API key
- The key should be provided in the environment variables or through the web interface

```http
Authorization: Bearer your_genai_hub_api_key
```

## 📝 API Endpoints

### Translation API

#### POST /api/translate
Translates a document file.

**Request**
```http
POST /api/translate
Content-Type: multipart/form-data

file: <file>
model: "claude-3-5-haiku"
source_lang: "en"
target_lang: "ja"
ai_instruction: ""
client_id: "unique_client_id"
```

**Parameters**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Document file (PPTX, DOCX, or PDF) |
| model | String | No | Translation model (default: "claude-3-5-haiku") |
| source_lang | String | No | Source language code (default: "en") |
| target_lang | String | No | Target language code (default: "ja") |
| ai_instruction | String | No | Additional instructions for AI |
| client_id | String | Yes | Unique client identifier for WebSocket updates |

**Response**
```json
{
    "success": true,
    "file_id": "uuid",
    "original_filename": "example.pdf",
    "translated_filename": "example_translated.pdf",
    "download_url": "/api/download/uuid_example_translated.pdf",
    "extracted_text_url": "/api/download/uuid_extracted.txt",
    "translated_text_url": "/api/download/uuid_translated.txt",
    "model": "claude-3-5-haiku",
    "source_language": "en",
    "target_language": "ja"
}
```

**Error Response**
```json
{
    "detail": "Error message"
}
```

#### GET /api/download/{filename}
Downloads a translated file.

**Request**
```http
GET /api/download/uuid_example_translated.pdf
```

**Response**
- File download response with appropriate Content-Type header
- 404 if file not found

### API Key Management

#### POST /api/save-api-key
Saves a new API key.

**Request**
```http
POST /api/save-api-key
Content-Type: application/x-www-form-urlencoded

api_key=your_api_key
```

**Response**
```json
{
    "message": "API key saved successfully"
}
```

**Error Response**
```json
{
    "detail": "Invalid API key"
}
```

#### GET /api/check-api-key
Checks if API key is configured.

**Request**
```http
GET /api/check-api-key
```

**Response**
```json
{
    "has_api_key": true
}
```

### System Information

#### GET /api/status
Returns application status.

**Request**
```http
GET /api/status
```

**Response**
```json
{
    "status": "ok",
    "version": "0.1.0",
    "timestamp": "2025-06-02T14:59:39.025Z",
    "debug_mode": true
}
```

#### GET /api/models
Returns available translation models.

**Request**
```http
GET /api/models
```

**Response**
```json
{
    "models": {
        "claude-4-sonnet": "Claude 4 Sonnet",
        "claude-3-7-sonnet": "Claude 3.7 Sonnet",
        "claude-3-5-sonnet-v2": "Claude 3.5 Sonnet V2",
        "claude-3-5-haiku": "Claude 3.5 Haiku"
    }
}
```

#### GET /api/languages
Returns supported languages.

**Request**
```http
GET /api/languages
```

**Response**
```json
{
    "languages": {
        "ja": "Japanese",
        "en": "English",
        "zh": "Chinese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish"
    }
}
```

## 🔌 WebSocket API

### WS /ws/{client_id}
Real-time progress updates for translation.

**Connection**
```javascript
const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
```

**Messages**
```json
{
    "progress": 0.5,
    "message": "Translating..."
}
```

**Progress Messages**
| Progress | Message |
|----------|---------|
| < 0.15 | "Preparing file..." |
| < 0.3 | "Extracting text..." |
| < 0.8 | "Translating..." |
| < 1.0 | "Reimporting text..." |
| 1.0 | "Completed" |

## 🛠 Error Handling

### HTTP Status Codes
| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request |
| 401 | Unauthorized |
| 404 | Not Found |
| 500 | Internal Server Error |

### Error Response Format
```json
{
    "detail": "Error message"
}
```

## 📦 Data Models

### Translation Request
```python
class TranslationRequest:
    file: UploadFile
    model: str = "claude-3-5-haiku"
    source_lang: str = "en"
    target_lang: str = "ja"
    ai_instruction: str = ""
    client_id: str
```

### Translation Response
```python
class TranslationResponse:
    success: bool
    file_id: str
    original_filename: str
    translated_filename: str
    download_url: str
    extracted_text_url: Optional[str]
    translated_text_url: Optional[str]
    model: str
    source_language: str
    target_language: str
```

## 📝 Examples

### cURL Examples

```bash
# Translate document
curl -X POST http://localhost:8000/api/translate \
  -H "Authorization: Bearer your_api_key" \
  -F "file=@document.pdf" \
  -F "model=claude-3-5-haiku" \
  -F "source_lang=en" \
  -F "target_lang=ja" \
  -F "client_id=test_client"

# Check API key
curl http://localhost:8000/api/check-api-key

# Get available models
curl http://localhost:8000/api/models
```

### Python Example

```python
import requests
import websockets
import asyncio
import json

async def translate_document(file_path, api_key):
    # WebSocket connection for progress updates
    client_id = "test_client"
    ws = await websockets.connect(f"ws://localhost:8000/ws/{client_id}")
    
    # File upload
    url = "http://localhost:8000/api/translate"
    
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "model": "claude-3-5-haiku",
            "source_lang": "en",
            "target_lang": "ja",
            "client_id": client_id
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Send translation request
        response = requests.post(url, files=files, data=data, headers=headers)
        result = response.json()
        
        # Listen for progress updates
        while True:
            message = await ws.recv()
            progress_data = json.loads(message)
            print(f"Progress: {progress_data['progress']*100}% - {progress_data['message']}")
            
            if progress_data['progress'] >= 1.0:
                break
        
        await ws.close()
        return result

# Usage
asyncio.run(translate_document("document.pdf", "your_api_key"))
```

### JavaScript Example

```javascript
async function translateDocument(file, apiKey) {
    // Create WebSocket connection
    const clientId = 'test_client';
    const ws = new WebSocket(`ws://localhost:8000/ws/${clientId}`);
    
    // Listen for progress updates
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log(`Progress: ${data.progress * 100}% - ${data.message}`);
    };
    
    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model', 'claude-3-5-haiku');
    formData.append('source_lang', 'en');
    formData.append('target_lang', 'ja');
    formData.append('client_id', clientId);

    // Send translation request
    const response = await fetch('http://localhost:8000/api/translate', {
        method: 'POST',
        headers: {
            'Authorization': `Bearer ${apiKey}`
        },
        body: formData
    });

    return await response.json();
}

// Usage
const fileInput = document.querySelector('input[type="file"]');
const file = fileInput.files[0];
const result = await translateDocument(file, 'your_api_key');
console.log(result);
```

## 🔒 Security Considerations

### API Key Security
- Store API keys securely
- Never expose API keys in client-side code
- Use environment variables for API key storage
- Rotate API keys periodically

### File Upload Security
- Validate file types
- Limit file sizes
- Sanitize filenames
- Clean up temporary files

### Error Handling Security
- Don't expose internal errors
- Log errors securely
- Return appropriate status codes
- Provide user-friendly error messages

## 📊 Rate Limiting

- Maximum file size: 100MB
- Request timeout: 300 seconds
- Concurrent translations: 4 per client
- API calls: 100 per minute

## 🔄 WebSocket Events

### Connection Events
```javascript
ws.onopen = () => {
    console.log('Connected to WebSocket');
};

ws.onclose = () => {
    console.log('Disconnected from WebSocket');
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

### Progress Updates
```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`Progress: ${data.progress * 100}% - ${data.message}`);
    
    if (data.progress >= 1.0) {
        ws.close();
    }
};
```

## 📝 API Versioning

Current version: v1

Future versions will be available at:
```
/api/v2/...
/api/v3/...
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [WebSocket Documentation](https://websockets.readthedocs.io/)
- [Python Requests Library](https://requests.readthedocs.io/)
- [JavaScript Fetch API](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
```