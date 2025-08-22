import pytest
from app.core.translator import (
    optimize_text_for_translation,
    LANGUAGES,
    format_model_name,
    filter_translation_models,
)


def test_optimize_text_for_translation():
    """テキスト最適化機能のテスト"""
    text = "Hello\n\nWorld\n\n"
    optimized = optimize_text_for_translation(text)
    assert "Hello" in optimized
    assert "World" in optimized


def test_supported_languages():
    """サポート言語の確認"""
    assert isinstance(LANGUAGES, dict)
    assert "ja" in LANGUAGES
    assert "en" in LANGUAGES
    assert LANGUAGES["ja"] == "Japanese"
    assert LANGUAGES["en"] == "English"


def test_format_model_name():
    """モデル名整形のテスト"""
    # Claude系モデル
    assert format_model_name("claude-3-5-haiku") == "Claude 3.5 Haiku"
    assert format_model_name("claude-4-sonnet") == "Claude 4 Sonnet"
    
    # Llama系モデル
    assert format_model_name("llama3-1-70b") == "Llama 3.1 70B"
    
    # その他
    assert format_model_name("other-model") == "Other Model"


def test_filter_translation_models():
    """モデルフィルタリングのテスト"""
    models = [
        "claude-3-5-haiku",
        "claude-4-sonnet", 
        "copilot-model",
        "documentation-helper",
        "llama3-1-70b"
    ]
    
    filtered = filter_translation_models(models)
    
    # 翻訳に適したモデルのみが残る
    assert "claude-3-5-haiku" in filtered
    assert "claude-4-sonnet" in filtered
    assert "llama3-1-70b" in filtered
    
    # 除外されるモデル
    assert "copilot-model" not in filtered
    assert "documentation-helper" not in filtered
