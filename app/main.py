# app/main.py
"""
DocTranslator / LangTranslator - 統合版メインアプリケーション

Document Translation (PPTX, DOCX, PDF, XLSX) + Text Translation
テキスト翻訳には言語自動検出・音声機能・履歴管理を追加
"""

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    WebSocket,
    HTTPException,
    Request,
    WebSocketDisconnect,
    BackgroundTasks,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response

import os
import uuid
import asyncio
import subprocess
import sqlite3
from pathlib import Path
import logging
import sys
from typing import Dict, Optional
from datetime import datetime
import csv
from io import StringIO
from contextlib import contextmanager

# Import configuration module
from app.config import (
    save_api_key,
    get_api_key,
    api_key_exists,
    save_api_settings,
    api_settings_exist,
    get_default_model,
    get_api_url,
)

# 言語検出ユーティリティのインポート（テキスト翻訳専用）
from app.utils.language_detector import (
    detect_language,
    get_language_name,
    suggest_target_language,
    validate_language_pair
)

# Logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("doctranslator")

# Load settings from environment variables
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logger.setLevel(LOG_LEVEL)

# Output more detailed logs in debug mode
if DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode is enabled")

# Static file directory paths
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
CSS_DIR = os.path.join(STATIC_DIR, "css")
JS_DIR = os.path.join(STATIC_DIR, "js")

# Text translation database path（テキスト翻訳専用）
TEXT_TRANSLATION_DB = Path("text_translations.db")

# データベース接続管理（テキスト翻訳専用）
@contextmanager
def get_db_connection():
    """テキスト翻訳用データベース接続のコンテキストマネージャー"""
    conn = sqlite3.connect(TEXT_TRANSLATION_DB)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_text_translation_db():
    """テキスト翻訳データベースの初期化"""
    try:
        if not TEXT_TRANSLATION_DB.exists():
            logger.info("Initializing text translation database...")
            
            # schema.sqlを読み込んで実行
            schema_path = Path("schema.sql")
            if schema_path.exists():
                with get_db_connection() as conn:
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        conn.executescript(f.read())
                    conn.commit()
                logger.info("Text translation database initialized successfully")
            else:
                logger.warning("schema.sql not found, creating basic schema")
                with get_db_connection() as conn:
                    conn.execute('''
                        CREATE TABLE IF NOT EXISTS text_translations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            source_lang TEXT NOT NULL,
                            target_lang TEXT NOT NULL,
                            source_text TEXT NOT NULL,
                            translated_text TEXT NOT NULL,
                            model TEXT NOT NULL,
                            auto_detected BOOLEAN DEFAULT 0,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    # インデックス作成
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_text_translations_timestamp ON text_translations(timestamp DESC)')
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_text_translations_source_lang ON text_translations(source_lang)')
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_text_translations_target_lang ON text_translations(target_lang)')
                    conn.execute('CREATE INDEX IF NOT EXISTS idx_text_translations_model ON text_translations(model)')
                    conn.commit()
        else:
            logger.info("Text translation database already exists")
    except Exception as e:
        logger.error(f"Failed to initialize text translation database: {e}")

# Import translation module
try:
    from app.core.translator import translate_document, LANGUAGES
    from app.config import api_settings_exist
    
    # Check API settings existence
    if api_settings_exist():
        # Dynamically fetch model list only if API settings exist
        try:
            from app.core.translator import fetch_available_models
            logger.info("API settings found. Fetching available models...")
            AVAILABLE_MODELS = fetch_available_models()
            logger.info(f"Available models loaded: {list(AVAILABLE_MODELS.keys())}")
        except Exception as e:
            logger.warning(f"Failed to fetch models despite API settings existing: {e}")
            logger.info("Setting models to empty due to fetch failure")
            AVAILABLE_MODELS = {}
    else:
        # Set models list to empty if no API settings
        logger.info("No API settings found. Models list will be empty until API is configured.")
        AVAILABLE_MODELS = {}
    
    logger.info(f"Supported languages: {list(LANGUAGES.keys())}")
    
except ImportError as e:
    logger.error(f"Failed to import translation module: {e}")
    # Set to empty on import error
    AVAILABLE_MODELS = {}
    LANGUAGES = {
        "ja": "Japanese",
        "en": "English",
        "zh": "Chinese",
        "ko": "Korean",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "hi": "Hindi",
        "vi": "Vietnamese",
        "th": "Thai",
    }
    logger.warning("Models list set to empty due to import error")
except Exception as e:
    logger.error(f"Unexpected error during initialization: {e}")
    # Set models list to empty on unexpected error
    AVAILABLE_MODELS = {}
    logger.warning("Models list set to empty due to initialization error")

# Debug information for static file directory
logger.debug(f"Static file directory: {STATIC_DIR}")
logger.debug(f"Static directory exists: {os.path.exists(STATIC_DIR)}")
if os.path.exists(STATIC_DIR):
    logger.debug(f"Static directory contents: {os.listdir(STATIC_DIR)}")

# Create application
app = FastAPI(
    title="DocTranslator / LangTranslator",
    description="Document and Text Translation Service",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Create upload and download directories
UPLOAD_DIR = Path("uploads")
DOWNLOAD_DIR = Path("downloads")
LOGS_DIR = Path("logs")

for directory in [UPLOAD_DIR, DOWNLOAD_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Dictionary for managing translation tasks
active_translations = {}

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        logger.info("WebSocket connection manager initialized")

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.debug(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.debug(f"Client {client_id} disconnected")

            # Cancel translation task when client disconnects
            if client_id in active_translations:
                active_translations[client_id]["cancelled"] = True
                logger.info(
                    f"Translation for client {client_id} marked as cancelled due to disconnect"
                )

    async def send_progress(self, client_id: str, progress: float, message: str):
        if client_id in self.active_connections:
            try:
                # Don't send progress if client's translation is cancelled
                if (
                    client_id in active_translations
                    and active_translations[client_id]["cancelled"]
                ):
                    logger.debug(
                        f"Progress update skipped for cancelled translation: {client_id}"
                    )
                    return

                progress_data = {
                    "progress": max(0.0, min(1.0, progress)),
                    "message": str(message),
                }
                await self.active_connections[client_id].send_json(progress_data)
                logger.debug(f"Progress sent to client {client_id}: {progress_data}")
            except Exception as e:
                logger.error(f"WebSocket send error: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

# LibreOffice check function
def check_libreoffice():
    """Check LibreOffice availability"""
    try:
        result = subprocess.run(
            ["libreoffice", "--version"], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.info(f"LibreOffice version: {result.stdout.strip()}")
            return True
    except Exception as e:
        logger.error(f"LibreOffice check error: {e}")
    return False

# ===== Application Startup Event =====

@app.on_event("startup")
async def startup_event():
    """Application startup process"""
    logger.info("DocTranslator / LangTranslator application has started")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Log level: {LOG_LEVEL}")

    # Initialize text translation database（テキスト翻訳専用）
    init_text_translation_db()

    # Start virtual display
    try:
        # Check for existing Xvfb process
        result = subprocess.run(["pgrep", "Xvfb"], capture_output=True)
        if result.returncode != 0:
            # Start Xvfb if not running
            subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Wait a bit
            await asyncio.sleep(2)
            logger.info("Virtual display (Xvfb) started")
        else:
            logger.info("Virtual display is already running")
    except Exception as e:
        logger.warning(f"Failed to start virtual display: {e}")

    # Test LibreOffice
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"LibreOffice check successful: {result.stdout.strip()}")
        else:
            logger.warning(f"LibreOffice check failed: {result.stderr}")
    except Exception as e:
        logger.error(f"LibreOffice check error: {e}")

    # Check directory existence
    for dir_name in ["uploads", "downloads", "logs"]:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)

    # LibreOffice check
    from app.core.translator import check_libreoffice

    libreoffice_available, libreoffice_info = check_libreoffice()
    if not libreoffice_available:
        logger.warning(f"LibreOffice is not available: {libreoffice_info}")
        logger.warning(
            "PDF conversion functionality will be limited. Please install LibreOffice."
        )
    else:
        logger.info(f"LibreOffice is available: {libreoffice_info}")

    logger.info("Application preparation completed with text translation support")

    # Temporary directory cleanup
    try:
        cleanup_count = 0
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.name != ".gitkeep":
                file_path.unlink()
                cleanup_count += 1

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} temporary files at startup")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {e}")

# ===== Document Translation Endpoints =====

@app.post("/api/cancel-translation/{client_id}")
async def cancel_translation(client_id: str):
    """Cancel an ongoing translation"""
    logger.info(f"Received cancellation request for client: {client_id}")

    if client_id in active_translations:
        active_translations[client_id]["cancelled"] = True
        logger.info(f"Translation for client {client_id} marked as cancelled")
        return {"status": "cancelled", "message": "Translation cancelled successfully"}
    else:
        logger.warning(f"No active translation found for client {client_id}")
        return {"status": "not_found", "message": "No active translation found"}

def get_progress_message(progress: float) -> str:
    """Return a message based on the progress"""
    if progress < 0.1:
        return "Preparing document..."
    elif progress < 0.2:
        return "Analyzing document structure..."
    elif progress < 0.3:
        return "Extracting text content..."
    elif progress < 0.5:
        return "Processing text for translation..."
    elif progress < 0.7:
        return "Translating content..."
    elif progress < 0.8:
        return "Formatting translated document..."
    elif progress < 0.9:
        return "Finalizing document..."
    else:
        return "Translation completed"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Received health check request")
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.now().isoformat()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main page"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        logger.debug(f"Serving index.html from: {index_path}")
        
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Add cache control headers for better performance
        return HTMLResponse(
            content=content,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    except FileNotFoundError:
        logger.error(f"index.html not found: {index_path}")
        return HTMLResponse(
            content="""
            <html>
                <head>
                    <title>DocTranslator - Error</title>
                    <style>
                        body {
                            font-family: Arial, sans-serif;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            height: 100vh;
                            margin: 0;
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        }
                        .error-container {
                            background: white;
                            padding: 40px;
                            border-radius: 20px;
                            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                            text-align: center;
                        }
                        h1 { color: #dc3545; }
                        p { color: #6c757d; }
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h1>⚠️ Error</h1>
                        <p>Static files not found. Please check your installation.</p>
                    </div>
                </body>
            </html>
            """,
            status_code=500
        )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Process messages from client
            if data == "cancel":
                if client_id in active_translations:
                    active_translations[client_id]["cancelled"] = True
                    logger.info(
                        f"Translation for client {client_id} cancelled via WebSocket"
                    )
                    await websocket.send_json(
                        {"status": "cancelled", "message": "Translation cancelled"}
                    )
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

# ===== API Endpoints =====

@app.get("/api/models")
async def get_models():
    """Return list of available translation models"""
    try:
        from app.config import api_settings_exist
        
        # Return empty models list if no API settings
        if not api_settings_exist():
            logger.debug("No API settings found. Returning empty models list.")
            return {
                "models": {}, 
                "error": "API settings not configured. Please configure API key and URL first."
            }
        
        # Fetch latest models list if API settings exist
        from app.core.translator import fetch_available_models
        models = fetch_available_models()
        
        # Update global variable
        global AVAILABLE_MODELS
        AVAILABLE_MODELS = models
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"Models list fetch error: {e}")
        # Return empty models list on error
        return {
            "models": {}, 
            "error": f"Failed to fetch models: {str(e)}"
        }

@app.get("/api/models/refresh")
async def refresh_models():
    """Refresh available models list"""
    try:
        from app.config import api_settings_exist
        
        # Check API settings
        if not api_settings_exist():
            raise HTTPException(
                status_code=400, 
                detail="API settings not configured. Please configure API settings first."
            )
        
        from app.core.translator import fetch_available_models
        models = fetch_available_models()
        
        # Update global variable
        global AVAILABLE_MODELS
        AVAILABLE_MODELS = models
        
        return {"models": models, "message": "Models refreshed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Models list update error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh models: {str(e)}")

@app.get("/api/languages")
async def get_languages():
    """Return list of supported languages"""
    return {"languages": LANGUAGES}

@app.get("/api/status")
async def get_status():
    """Return application status"""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "debug_mode": DEBUG,
    }

@app.post("/api/save-api-settings")
async def save_api_settings_endpoint(request: Request, api_key: str = Form(...), api_url: str = Form(...)):
    """Save API settings endpoint"""
    logger.info("Received API settings save request")

    if not api_key or len(api_key.strip()) < 20:
        logger.error("Invalid API key format")
        raise HTTPException(status_code=400, detail="Invalid API key")
        
    if not api_url or not api_url.startswith("http"):
        logger.error("Invalid API URL format")
        raise HTTPException(status_code=400, detail="Invalid API URL")

    try:
        from app.config import save_api_settings
        result = save_api_settings(api_key, api_url)
        logger.info(f"API settings save result: {result}")

        if result:
            # Check if API settings are actually set after successful save
            from app.config import api_settings_exist
            if api_settings_exist():
                return JSONResponse(
                    status_code=200, content={"message": "API settings saved successfully"}
                )
            else:
                logger.error("API settings were not properly saved")
                raise HTTPException(
                    status_code=500, detail="API settings were not properly saved"
                )
        else:
            logger.error("Failed to save API settings")
            raise HTTPException(status_code=500, detail="Failed to save API settings")

    except Exception as e:
        logger.error(f"API settings save error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check-api-settings")
async def check_api_settings():
    """Check if API settings are configured"""
    try:
        # Reload environment variables
        from app.config import load_env_file, api_settings_exist, get_api_key, get_api_url

        load_env_file()

        has_api_settings = api_settings_exist()
        api_key = get_api_key()
        api_url = get_api_url()

        # Output masked API key to log (for debugging)
        masked_key = (
            "****"
            if not api_key
            else f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        )
        logger.debug(f"API settings check: exists={has_api_settings}, key={masked_key}, url={api_url}")

        return {"has_api_settings": has_api_settings, "api_url": api_url}
    except Exception as e:
        logger.error(f"API settings check error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An error occurred while checking the API settings"
        )

@app.post("/api/translate")
async def translate_file(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    api_key: str = Form(None),
    model: str = Form(None),
    source_lang: str = Form("en"),
    target_lang: str = Form("ja"),
    ai_instruction: str = Form(""),
    client_id: str = Form(...),
):
    """
    Handle file translation requests
    
    注意: このエンドポイントはドキュメント翻訳専用です。
         言語自動検出は実行されません。
    """
    # Auto-select model if not specified
    if model is None:
        model = get_default_model()
    
    logger.debug("[Document Translation] Translation request received:")
    logger.debug(f"File name: {file.filename}")
    logger.debug(f"Model: {model}")
    logger.debug(f"Source language: {source_lang}")
    logger.debug(f"Target language: {target_lang}")
    logger.debug(f"Client ID: {client_id}")

    # Initialize client's translation state
    active_translations[client_id] = {"cancelled": False}
    
    # Initialize variables
    extracted_filename = None
    translated_filename = None

    try:
        # Get API key from environment
        stored_api_key = get_api_key()
        logger.debug(f"Stored API key: {'exists' if stored_api_key else 'not set'}")

        # Select API key (UI provided key takes precedence)
        final_api_key = api_key or stored_api_key

        # API key validation
        if not final_api_key:
            logger.error("API key is not set")
            raise HTTPException(status_code=401, detail="API key is not set")

        # File validation
        if not file.filename:
            logger.error("File name is empty")
            raise HTTPException(status_code=400, detail="File name is empty")

        if not file.filename.lower().endswith((".pptx", ".docx", ".pdf", ".xlsx")):
            logger.error(f"Unsupported file format: {file.filename}")
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Only PPTX, DOCX, PDF, XLSX are supported.",
            )

        # Model validation
        if model not in AVAILABLE_MODELS:
            logger.error(f"Unsupported model: {model}")
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

        # Language validation
        if source_lang not in LANGUAGES or target_lang not in LANGUAGES:
            logger.error(
                f"Unsupported language: source={source_lang}, target={target_lang}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language: source={source_lang}, target={target_lang}",
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        safe_filename = file.filename.replace(" ", "_")

        # Generate output filename (without unique ID)
        base_filename = Path(safe_filename).stem
        output_filename = f"{base_filename}_translated{Path(safe_filename).suffix}"
        output_path = DOWNLOAD_DIR / f"{file_id}_{output_filename}"

        # Define input and output file paths
        input_path = UPLOAD_DIR / f"{file_id}_{safe_filename}"

        logger.info(
            f"[Document Translation] Starting translation process: {safe_filename}, Model: {model}, Languages: {source_lang} -> {target_lang}"
        )

        try:
            # Save file
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.debug(f"File saved: {input_path}")

            # Function to monitor client connection state
            async def check_client_connected():
                try:
                    while True:
                        if await request.is_disconnected():
                            logger.warning(f"Client {client_id} disconnected")
                            active_translations[client_id]["cancelled"] = True
                            return
                        await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Connection check error: {e}")

            # Monitor client connection in background
            background_tasks.add_task(check_client_connected)

            # Progress callback function
            async def progress_callback(progress: float):
                try:
                    # Skip progress send if cancelled
                    if (
                        client_id in active_translations
                        and active_translations[client_id]["cancelled"]
                    ):
                        logger.debug(
                            f"Progress update skipped for cancelled translation: {client_id}"
                        )
                        return

                    message = get_progress_message(progress)
                    await manager.send_progress(client_id, progress, message)
                    logger.debug(f"Progress sent: {progress}, {message}")
                except Exception as e:
                    logger.error(f"Progress send error: {e}")

            # Thread-safe progress callback wrapper
            def safe_progress_callback(progress: float):
                """Thread-safe progress callback"""
                try:
                    # Skip progress send if cancelled
                    if (
                        client_id in active_translations
                        and active_translations[client_id]["cancelled"]
                    ):
                        logger.debug(
                            "Progress callback skipped for cancelled translation"
                        )
                        return

                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(progress_callback(progress))
                except Exception as e:
                    logger.error(f"Progress callback error: {e}")
                finally:
                    loop.close()

            logger.info(f"Starting translation: {input_path} -> {output_path}")

            # Send initial progress
            await manager.send_progress(client_id, 0.01, "Preparing translation...")

            # Function to check cancellation state
            def check_cancelled():
                return (
                    client_id in active_translations
                    and active_translations[client_id]["cancelled"]
                )

            # Execute translation
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: translate_document(
                        str(input_path),
                        str(output_path),
                        final_api_key,
                        model,
                        source_lang,
                        target_lang,
                        safe_progress_callback,
                        True,
                        ai_instruction,
                        check_cancelled,
                    ),
                )

                # Return early if cancelled
                if check_cancelled():
                    logger.info(f"Translation for client {client_id} was cancelled")
                    if input_path.exists():
                        input_path.unlink()
                    return {"success": False, "message": "Translation was cancelled"}

                # Process result
                if isinstance(result, tuple) and len(result) == 3:
                    extracted_file, translated_file, docx_path = result
                    # DOCX file returned (PDF conversion failed)
                    
                    # Set filename correctly
                    docx_filename = os.path.basename(docx_path)
                    # Generate actual filename including file ID
                    actual_docx_filename = f"{file_id}_{docx_filename}"
                    actual_docx_path = DOWNLOAD_DIR / actual_docx_filename
                    
                    # Copy DOCX file to correct location
                    if os.path.exists(docx_path) and not os.path.exists(actual_docx_path):
                        import shutil
                        shutil.copy2(docx_path, actual_docx_path)
                        logger.info(f"DOCX file copied: {docx_path} -> {actual_docx_path}")
                    
                    # Set filenames
                    extracted_filename = (
                        Path(extracted_file).name if extracted_file else None
                    )
                    translated_filename = (
                        Path(translated_file).name if translated_file else None
                    )
                    
                    await manager.send_progress(
                        client_id, 1.0, "PDF conversion failed: Providing DOCX file"
                    )
                    return {
                        "success": True,
                        "warning": "Failed to convert to PDF. Providing DOCX file instead.",
                        "file_id": file_id,
                        "original_filename": file.filename,
                        "translated_filename": docx_filename,
                        "download_url": f"/api/download/{actual_docx_filename}",
                        "extracted_text_url": (
                            f"/api/download/{extracted_filename}"
                            if extracted_filename
                            else None
                        ),
                        "translated_text_url": (
                            f"/api/download/{translated_filename}"
                            if translated_filename
                            else None
                        ),
                        "model": model,
                        "source_language": source_lang,
                        "target_language": target_lang,
                    }
                else:
                    # Normal translation result (2 elements)
                    extracted_file, translated_file = result
                    
                    # Set filenames
                    extracted_filename = (
                        Path(extracted_file).name if extracted_file else None
                    )
                    translated_filename = (
                        Path(translated_file).name if translated_file else None
                    )

                    logger.info(f"Translation completed: {output_path}")

                    # Send final progress
                    await manager.send_progress(client_id, 1.0, "Translation Completed")

                    return {
                        "success": True,
                        "file_id": file_id,
                        "original_filename": file.filename,
                        "translated_filename": output_filename,
                        "download_url": f"/api/download/{file_id}_{output_filename}",
                        "extracted_text_url": (
                            f"/api/download/{extracted_filename}"
                            if extracted_filename
                            else None
                        ),
                        "translated_text_url": (
                            f"/api/download/{translated_filename}"
                            if translated_filename
                            else None
                        ),
                        "model": model,
                        "source_language": source_lang,
                        "target_language": target_lang,
                    }

            except ValueError as ve:
                # PDF conversion error case
                if "Failed to convert DOCX to PDF" in str(ve) or "PDFからDOCXへの変換に失敗" in str(ve) or "PDF翻訳に失敗しました" in str(ve):
                    # Generate DOCX filename
                    docx_filename = output_filename.replace(".pdf", ".docx")
                    docx_output_path = DOWNLOAD_DIR / f"{file_id}_{docx_filename}"
                    
                    # Check if DOCX file exists
                    if docx_output_path.exists():
                        await manager.send_progress(
                            client_id, 1.0, "PDF conversion failed: Providing DOCX file instead"
                        )

                        return {
                            "success": True,
                            "warning": "Failed to convert to PDF. Providing DOCX file instead.",
                            "file_id": file_id,
                            "original_filename": file.filename,
                            "translated_filename": docx_filename,
                            "download_url": f"/api/download/{file_id}_{docx_filename}",
                            "extracted_text_url": (
                                f"/api/download/{extracted_filename}"
                                if extracted_filename
                                else None
                            ),
                            "translated_text_url": (
                                f"/api/download/{translated_filename}"
                                if translated_filename
                                else None
                            ),
                            "model": model,
                            "source_language": source_lang,
                            "target_language": target_lang,
                        }
                    else:
                        logger.error(f"Translation process error: {ve}", exc_info=True)
                        await manager.send_progress(
                            client_id, 0, f"Translation Error: {str(ve)}"
                        )
                        raise HTTPException(
                            status_code=500,
                            detail=f"An error occurred during translation: {str(ve)}",
                        )
                else:
                    logger.error(f"Translation process error: {ve}", exc_info=True)
                    await manager.send_progress(
                        client_id, 0, f"Translation Error: {str(ve)}"
                    )
                    raise HTTPException(
                        status_code=500,
                        detail=f"An error occurred during translation: {str(ve)}",
                    )

            except Exception as translate_error:
                logger.error(
                    f"Translation process error: {translate_error}", exc_info=True
                )
                await manager.send_progress(
                    client_id, 0, f"Translation Error: {str(translate_error)}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"An error occurred during translation: {str(translate_error)}",
                )

        except IOError as io_error:
            logger.error(f"File save error: {io_error}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"An error occurred while saving the file: {str(io_error)}",
            )

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )

    finally:
        # Delete input file
        try:
            if "input_path" in locals() and input_path.exists():
                input_path.unlink()
                logger.debug(f"Temporary file deleted: {input_path}")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")

        # Clean up client's translation state
        if client_id in active_translations:
            del active_translations[client_id]
            logger.debug(f"Cleaned up translation state for client {client_id}")

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Handle file downloads"""
    logger.debug(f"Download request received: {filename}")

    # Remove unique ID from filename
    if "_" in filename:
        # Get filename after first underscore
        original_filename = "_".join(filename.split("_")[1:])
    else:
        original_filename = filename

    # Handle extracted text file download
    if original_filename.endswith("_extracted.txt"):
        file_path = DOWNLOAD_DIR / filename
        logger.debug(f"Extracted text file path: {file_path}")

        if not file_path.exists():
            logger.warning(f"Extracted text file not found: {file_path}")
            raise HTTPException(status_code=404, detail="Extracted text file not found")

        return FileResponse(
            path=file_path,
            filename=original_filename,
            media_type="text/plain",
        )

    # Handle translated text file download
    elif original_filename.endswith("_translated.txt"):
        file_path = DOWNLOAD_DIR / filename
        logger.debug(f"Translated text file path: {file_path}")

        if not file_path.exists():
            logger.warning(f"Translated text file not found: {file_path}")
            raise HTTPException(
                status_code=404, detail="Translated text file not found"
            )

        return FileResponse(
            path=file_path,
            filename=original_filename,
            media_type="text/plain",
        )

    # Handle translated document download
    else:
        file_path = DOWNLOAD_DIR / filename
        logger.debug(f"Translated document file path: {file_path}")

        if not file_path.exists():
            logger.warning(f"Translated document file not found: {file_path}")
            raise HTTPException(
                status_code=404, detail="Translated document file not found"
            )

        # Set media type based on file extension
        file_ext = Path(original_filename).suffix.lower()
        media_types = {
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pdf": "application/pdf",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

        media_type = media_types.get(file_ext, "application/octet-stream")

        return FileResponse(
            path=file_path,
            filename=original_filename,
            media_type=media_type,
        )

# ===== Text Translation Endpoints（言語自動検出・履歴管理機能付き） =====

@app.post("/api/detect-language")
async def detect_language_endpoint(request: Request):
    """
    テキストから言語を自動検出（テキスト翻訳専用）
    
    注意: このエンドポイントはテキスト翻訳でのみ使用されます。
         ドキュメント翻訳では使用されません。
    """
    try:
        data = await request.json()
        text = data.get('text', '')
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        detected_lang = detect_language(text)
        lang_name = get_language_name(detected_lang)
        suggested_target = suggest_target_language(detected_lang)
        
        logger.debug(f"[Text Translation] Language detected: {detected_lang} ({lang_name})")
        
        return {
            "success": True,
            "detected_language": detected_lang,
            "language_name": lang_name,
            "suggested_target": suggested_target
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Text Translation] Language detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/translate-text")
async def translate_text_endpoint(
    text: str = Form(...),
    source_lang: str = Form("en"),
    target_lang: str = Form("ja"),
    model: str = Form(None),
    auto_detect: bool = Form(True),
):
    """
    テキスト翻訳エンドポイント（言語検出・履歴保存機能付き）
    
    注意: 言語自動検出はテキスト翻訳でのみ動作します。
         ドキュメント翻訳では使用されません。
    """
    try:
        # モデル自動選択
        if model is None:
            model = get_default_model()
        
        # 言語検出（テキスト翻訳のみ）
        was_auto_detected = False
        original_source_lang = source_lang
        
        if auto_detect:
            detected_lang = detect_language(text)
            
            # 検出された言語がソース言語と異なる場合は更新
            if detected_lang != source_lang:
                logger.info(f"[Text Translation] Auto-detected language: {detected_lang} (was: {source_lang})")
                source_lang = detected_lang
                # ターゲット言語も自動調整
                target_lang = suggest_target_language(source_lang)
                was_auto_detected = True
        
        # 言語ペアの検証
        is_valid, error_msg = validate_language_pair(source_lang, target_lang)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # API設定取得
        api_key = get_api_key()
        if not api_key:
            raise HTTPException(status_code=401, detail="API key is not set")
        
        logger.info(f"[Text Translation] Starting translation: {source_lang} -> {target_lang}, Model: {model}")
        
        # テキスト翻訳実行
        from app.core.text_translator import translate_text_chunks
        translated_text = translate_text_chunks(
            text=text,
            api_key=api_key,
            model=model,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # データベースに保存（テキスト翻訳専用）
        try:
            with get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO text_translations 
                    (timestamp, source_lang, target_lang, source_text, translated_text, model, auto_detected)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    get_language_name(source_lang),
                    get_language_name(target_lang),
                    text,
                    translated_text,
                    model,
                    1 if was_auto_detected else 0
                ))
                conn.commit()
            logger.debug("[Text Translation] Translation saved to database")
        except Exception as db_error:
            logger.warning(f"[Text Translation] Failed to save to database: {db_error}")
            # データベース保存失敗は致命的エラーではない
        
        return {
            "success": True,
            "translated": translated_text,
            "source_lang": get_language_name(source_lang),
            "target_lang": get_language_name(target_lang),
            "detected_lang": source_lang if was_auto_detected else None,
            "auto_detected": was_auto_detected,
            "model": model
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Text Translation] Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/text-translation-history")
async def get_text_translation_history():
    """テキスト翻訳履歴を取得（テキスト翻訳専用）"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM text_translations 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            entries = [dict(row) for row in cursor.fetchall()]
        
        logger.debug(f"[Text Translation] Retrieved {len(entries)} history entries")
        return {"success": True, "entries": entries}
        
    except Exception as e:
        logger.error(f"[Text Translation] Error fetching history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear-text-translation-history")
async def clear_text_translation_history():
    """テキスト翻訳履歴をクリア（テキスト翻訳専用）"""
    try:
        with get_db_connection() as conn:
            conn.execute('DELETE FROM text_translations')
            conn.commit()
        
        logger.info("[Text Translation] History cleared")
        return {"success": True}
        
    except Exception as e:
        logger.error(f"[Text Translation] Error clearing history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export-text-translation-history")
async def export_text_translation_history():
    """テキスト翻訳履歴をCSVでエクスポート（テキスト翻訳専用）- 修正版"""
    try:
        with get_db_connection() as conn:
            cursor = conn.execute('''
                SELECT 
                    timestamp,
                    source_lang,
                    target_lang,
                    source_text,
                    translated_text,
                    model,
                    auto_detected
                FROM text_translations 
                ORDER BY timestamp DESC
            ''')
            
            output = StringIO()
            writer = csv.writer(output)
            
            # ヘッダー
            writer.writerow([
                'Timestamp', 
                'Source Language', 
                'Target Language',
                'Original Text', 
                'Translated Text', 
                'Model', 
                'Auto Detected'
            ])
            
            # データ - インデックスでアクセス
            for row in cursor.fetchall():
                writer.writerow([
                    row[0],  # timestamp
                    row[1],  # source_lang
                    row[2],  # target_lang
                    row[3],  # source_text
                    row[4],  # translated_text
                    row[5],  # model
                    'Yes' if row[6] else 'No'  # auto_detected
                ])
        
        csv_data = '\ufeff' + output.getvalue()
        csv_data = csv_data.encode('utf-8')
        
        logger.info("[Text Translation] History exported successfully")
        
        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=text_translation_history.csv"
            }
        )
        
    except Exception as e:
        logger.error(f"[Text Translation] Error exporting history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Main application entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
