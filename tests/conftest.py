import pytest
from fastapi.testclient import TestClient
from app.main import app
import os
from unittest.mock import patch


@pytest.fixture
def client():
    """テストクライアントを提供するフィクスチャ"""
    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_api_and_translation():
    """API設定と翻訳機能を自動的にモック化"""
    with patch("app.config.api_settings_exist", return_value=True), \
         patch("app.config.get_api_key", return_value="test_api_key_12345678901234567890"), \
         patch("app.config.get_api_url", return_value="https://api.test.com/v1/chat/completions"), \
         patch("app.core.translator.fetch_available_models", return_value={
             "claude-3-5-haiku": "Claude 3.5 Haiku",
             "claude-3-5-sonnet-v2": "Claude 3.5 Sonnet V2",
             "claude-3-7-sonnet": "Claude 3.7 Sonnet",
             "claude-4-sonnet": "Claude 4 Sonnet",
         }), \
         patch("app.main.translate_document", return_value=("extracted.txt", "translated.txt")), \
         patch("app.main.manager.send_progress", return_value=None):
        yield


@pytest.fixture
def sample_file_content():
    """サンプルファイルの内容"""
    return b"Sample file content for testing"
