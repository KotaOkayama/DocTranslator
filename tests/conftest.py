import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from dotenv import load_dotenv
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path
import shutil


# テスト用の一時的な.envファイルを作成する関数
def create_temp_env_file(original_env_path):
    temp_env_path = original_env_path.parent / ".env.test"
    if original_env_path.exists():
        shutil.copy2(original_env_path, temp_env_path)
    return temp_env_path


@pytest.fixture(autouse=True)
def env_setup():
    """テスト用の環境変数を設定"""
    # 元の.envファイルのパスを保存
    original_env_path = Path(__file__).parent.parent / ".env"
    temp_env_path = create_temp_env_file(original_env_path)

    # 元の環境変数を保存
    original_env = dict(os.environ)

    # dotenvのset_keyとsave_api_keyをモック化
    with patch("dotenv.set_key") as _, patch(
        "app.config.ENV_FILE", temp_env_path
    ), patch("app.config.save_api_key") as _:

        # テスト用の環境変数を設定
        test_env = {
            "TESTING": "true",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "GENAI_HUB_API_KEY": original_env.get("GENAI_HUB_API_KEY", "test_key"),
        }
        os.environ.update(test_env)

        yield

        # テスト後のクリーンアップ
        os.environ.clear()
        os.environ.update(original_env)

        # 一時的な.envファイルを削除
        if temp_env_path.exists():
            temp_env_path.unlink()


@pytest.fixture
def client():
    """テストクライアントを提供するフィクスチャ"""
    return TestClient(app)


@pytest.fixture
def test_files(tmp_path):
    """テストファイルのパスを提供"""
    # テスト用のディレクトリ構造を作成
    test_files_dir = tmp_path / "test_files"
    test_files_dir.mkdir()

    # テストファイルを作成
    docx_file = test_files_dir / "test.docx"
    pptx_file = test_files_dir / "test.pptx"
    pdf_file = test_files_dir / "test.pdf"

    # サンプルコンテンツを書き込む
    docx_file.write_bytes(b"test docx content")
    pptx_file.write_bytes(b"test pptx content")
    pdf_file.write_bytes(b"test pdf content")

    return {"docx": docx_file, "pptx": pptx_file, "pdf": pdf_file}


@pytest.fixture
def mock_websocket():
    """WebSocketのモック"""
    mock = Mock()
    mock.send_json = Mock()
    return mock


@pytest.fixture
def setup_test_env(tmp_path):
    """テスト環境のセットアップ"""
    # テスト用ディレクトリの作成
    uploads_dir = tmp_path / "uploads"
    downloads_dir = tmp_path / "downloads"
    logs_dir = tmp_path / "logs"

    uploads_dir.mkdir()
    downloads_dir.mkdir()
    logs_dir.mkdir()

    # 元の環境変数を保存
    original_env = dict(os.environ)
    original_api_key = os.environ.get("GENAI_HUB_API_KEY")

    # テスト用の環境変数を設定
    test_env = {
        "UPLOAD_DIR": str(uploads_dir),
        "DOWNLOAD_DIR": str(downloads_dir),
        "LOG_DIR": str(logs_dir),
        "TESTING": "true",
    }
    os.environ.update(test_env)

    # API キー関連の操作をモック化
    with patch("app.config.get_api_key", return_value=original_api_key), patch(
        "app.config.save_api_key"
    ) as mock_save_api_key, patch("app.config.api_key_exists", return_value=True):

        mock_save_api_key.return_value = True

        yield {
            "uploads_dir": uploads_dir,
            "downloads_dir": downloads_dir,
            "logs_dir": logs_dir,
        }

    # テスト後のクリーンアップ
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_file_content():
    """サンプルファイルの内容"""
    return b"Sample file content for testing"


@pytest.fixture
def mock_api_settings():
    """API設定操作のモック"""
    with patch("app.config.get_api_key", return_value="test_api_key"), patch(
        "app.config.get_api_url", return_value="https://api.anthropic.com/v1/messages"
    ), patch("app.config.save_api_settings", return_value=True), patch(
        "app.config.api_settings_exist", return_value=True
    ):
        yield


@pytest.fixture
def mock_env_file():
    """環境変数ファイルの操作をモック化"""
    with patch("dotenv.set_key") as mock_set_key:
        yield mock_set_key
