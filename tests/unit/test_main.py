import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_read_health(client):
    """ヘルスチェックエンドポイントのテスト"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "version" in data
    assert "timestamp" in data


def test_get_models(client):
    """利用可能なモデル一覧取得のテスト"""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], dict)
    assert "claude-4-sonnet" in data["models"]


def test_get_languages(client):
    """サポート言語一覧取得のテスト"""
    response = client.get("/api/languages")
    assert response.status_code == 200
    data = response.json()
    assert "languages" in data
    assert isinstance(data["languages"], dict)
    assert "ja" in data["languages"]
    assert "en" in data["languages"]
