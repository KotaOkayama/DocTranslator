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
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

import os
import uuid
import asyncio
import subprocess
from pathlib import Path
import logging
import sys
from typing import Dict, Optional
from datetime import datetime

# Import configuration module
from app.config import save_api_key, get_api_key, api_key_exists, save_api_settings, api_settings_exist

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


# Import translation module
try:
    from app.core.translator import translate_document, LANGUAGES
    from app.config import api_settings_exist
    
    # API設定の存在チェック
    if api_settings_exist():
        # API設定がある場合のみモデル一覧を動的取得
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
        # API設定がない場合はモデル一覧を空にする
        logger.info("No API settings found. Models list will be empty until API is configured.")
        AVAILABLE_MODELS = {}
    
    logger.info(f"Supported languages: {list(LANGUAGES.keys())}")
    
except ImportError as e:
    logger.error(f"Failed to import translation module: {e}")
    # インポートエラーの場合も空にする
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
    # 予期しないエラーでもモデル一覧は空にする
    AVAILABLE_MODELS = {}
    logger.warning("Models list set to empty due to initialization error")


# Debug information for static file directory
logger.debug(f"Static file directory: {STATIC_DIR}")
logger.debug(f"Static directory exists: {os.path.exists(STATIC_DIR)}")
logger.debug(f"Static directory contents: {os.listdir(STATIC_DIR)}")

# Create application
app = FastAPI(
    title="DocTranslator",
    description="Document Translation Service",
    version="0.1.0",
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

# 翻訳タスク管理用の辞書
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

            # クライアントの切断時に翻訳タスクをキャンセル
            if client_id in active_translations:
                active_translations[client_id]["cancelled"] = True
                logger.info(
                    f"Translation for client {client_id} marked as cancelled due to disconnect"
                )

    async def send_progress(self, client_id: str, progress: float, message: str):
        if client_id in self.active_connections:
            try:
                # クライアントの翻訳がキャンセルされていたら進捗を送信しない
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

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup process"""
    logger.info("DocTranslator application has started")
    logger.info(f"Debug mode: {DEBUG}")
    logger.info(f"Log level: {LOG_LEVEL}")

    # 仮想ディスプレイの開始
    try:
        # 既存のXvfbプロセスを確認
        result = subprocess.run(["pgrep", "Xvfb"], capture_output=True)
        if result.returncode != 0:
            # Xvfbが動いていない場合は開始
            subprocess.Popen(
                ["Xvfb", ":99", "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # 少し待機
            await asyncio.sleep(2)
            logger.info("仮想ディスプレイ (Xvfb) を開始しました")
        else:
            logger.info("仮想ディスプレイは既に動作中です")
    except Exception as e:
        logger.warning(f"仮想ディスプレイの開始に失敗: {e}")

    # LibreOfficeのテスト
    try:
        result = subprocess.run(
            ["libreoffice", "--headless", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"LibreOffice確認成功: {result.stdout.strip()}")
        else:
            logger.warning(f"LibreOffice確認失敗: {result.stderr}")
    except Exception as e:
        logger.error(f"LibreOffice確認エラー: {e}")

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

    logger.info("Application preparation completed")

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

# 翻訳タスクをキャンセルするエンドポイント
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
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.now().isoformat()}

# Root page endpoint
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main page"""
    try:
        index_path = os.path.join(STATIC_DIR, "index.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        logger.error(f"index.html not found: {index_path}")
        return HTMLResponse(
            content="""
        <html>
            <head>
                <title>DocTranslator</title>
            </head>
            <body>
                <h1>DocTranslator</h1>
                <p>Static files not found.</p>
            </body>
        </html>
        """
        )

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Handle WebSocket connections"""
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            # クライアントからのメッセージを処理
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


# API endpoints

@app.get("/api/models")
async def get_models():
    """Return list of available translation models"""
    try:
        from app.config import api_settings_exist
        
        # API設定がない場合は空のモデル一覧を返す
        if not api_settings_exist():
            logger.debug("No API settings found. Returning empty models list.")
            return {
                "models": {}, 
                "error": "API settings not configured. Please configure API key and URL first."
            }
        
        # API設定がある場合は最新のモデル一覧を取得
        from app.core.translator import fetch_available_models
        models = fetch_available_models()
        
        # グローバル変数も更新
        global AVAILABLE_MODELS
        AVAILABLE_MODELS = models
        
        return {"models": models}
        
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {e}")
        # エラーの場合も空のモデル一覧を返す
        return {
            "models": {}, 
            "error": f"Failed to fetch models: {str(e)}"
        }



@app.get("/api/models/refresh")
async def refresh_models():
    """Refresh available models list"""
    try:
        from app.config import api_settings_exist
        
        # API設定チェック
        if not api_settings_exist():
            raise HTTPException(
                status_code=400, 
                detail="API settings not configured. Please configure API settings first."
            )
        
        from app.core.translator import fetch_available_models
        models = fetch_available_models()
        
        # グローバル変数を更新
        global AVAILABLE_MODELS
        AVAILABLE_MODELS = models
        
        return {"models": models, "message": "Models refreshed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"モデル一覧更新エラー: {e}")
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
        "version": "0.1.0",
        "timestamp": datetime.now().isoformat(),
        "debug_mode": DEBUG,
    }

# API設定エンドポイントを追加
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
            # 保存成功後、API設定が実際に設定されているか確認
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
        # 環境変数を再読み込み
        from app.config import load_env_file, api_settings_exist, get_api_key, get_api_url

        load_env_file()

        has_api_settings = api_settings_exist()
        api_key = get_api_key()
        api_url = get_api_url()

        # マスクされたAPIキーをログに出力（デバッグ用）
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
    model: str = Form("claude-3-5-haiku"),
    source_lang: str = Form("en"),
    target_lang: str = Form("ja"),
    ai_instruction: str = Form(""),
    client_id: str = Form(...),
):
    """Handle file translation requests"""
    logger.debug("Translation request received:")
    logger.debug(f"File name: {file.filename}")
    logger.debug(f"Model: {model}")
    logger.debug(f"Source language: {source_lang}")
    logger.debug(f"Target language: {target_lang}")
    logger.debug(f"Client ID: {client_id}")

    # クライアントの翻訳状態を初期化
    active_translations[client_id] = {"cancelled": False}
    
    # 変数を初期化
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
            f"Starting translation process: {safe_filename}, Model: {model}, Languages: {source_lang} -> {target_lang}"
        )

        try:
            # Save file
            with open(input_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            logger.debug(f"File saved: {input_path}")

            # クライアント接続状態を監視する関数
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

            # バックグラウンドでクライアント接続を監視
            background_tasks.add_task(check_client_connected)

            # Progress callback function
            async def progress_callback(progress: float):
                try:
                    # 中止されていたら進捗送信をスキップ
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
                    # 中止されていたら進捗送信をスキップ
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

            # 中止状態をチェックする関数
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

                # 中止された場合は早期リターン
                if check_cancelled():
                    logger.info(f"Translation for client {client_id} was cancelled")
                    if input_path.exists():
                        input_path.unlink()
                    return {"success": False, "message": "Translation was cancelled"}

                # 結果の処理
                if isinstance(result, tuple) and len(result) == 3:
                    extracted_file, translated_file, docx_path = result
                    # DOCXファイルが返された場合（PDF変換失敗）
                    
                    # ファイル名を正しく設定（修正）
                    docx_filename = os.path.basename(docx_path)
                    # ファイルIDを含む実際のファイル名を生成
                    actual_docx_filename = f"{file_id}_{docx_filename}"
                    actual_docx_path = DOWNLOAD_DIR / actual_docx_filename
                    
                    # DOCXファイルを正しい場所にコピー
                    if os.path.exists(docx_path) and not os.path.exists(actual_docx_path):
                        import shutil
                        shutil.copy2(docx_path, actual_docx_path)
                        logger.info(f"DOCXファイルをコピーしました: {docx_path} -> {actual_docx_path}")
                    
                    # ファイル名を設定
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
                    # 通常の翻訳結果（2つの要素）
                    extracted_file, translated_file = result
                    
                    # ファイル名を設定
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
                    # DOCXファイル名を生成
                    docx_filename = output_filename.replace(".pdf", ".docx")
                    docx_output_path = DOWNLOAD_DIR / f"{file_id}_{docx_filename}"
                    
                    # DOCXファイルが存在するかチェック
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

        # クライアントの翻訳状態をクリーンアップ
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
            filename=original_filename,  # Filename without unique ID
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
            filename=original_filename,  # Filename without unique ID
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
            filename=original_filename,  # Filename without unique ID
            media_type=media_type,
        )

# Main application entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
