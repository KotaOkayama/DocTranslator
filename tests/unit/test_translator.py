import pytest
from app.core.translator import (
    translate_text_with_cache,
    optimize_text_for_translation,
    AVAILABLE_MODELS,
    LANGUAGES,
)


def test_optimize_text_for_translation():
    """テキスト最適化機能のテスト"""
    # 空白行を含むテキスト
    text = """
    Hello

    World

    """
    optimized = optimize_text_for_translation(text)
    assert optimized == "    Hello\n    World"

    # 数字のみの行を含むテキスト
    text = """
    Line 1
    123456
    Line 2
    """
    optimized = optimize_text_for_translation(text)
    assert "Line 1" in optimized
    assert "123456" in optimized
    assert "Line 2" in optimized


def test_available_models():
    """利用可能なモデルの確認"""
    assert isinstance(AVAILABLE_MODELS, dict)
    assert "claude-4-sonnet" in AVAILABLE_MODELS
    assert "claude-3-5-haiku" in AVAILABLE_MODELS


def test_supported_languages():
    """サポート言語の確認"""
    assert isinstance(LANGUAGES, dict)
    assert "ja" in LANGUAGES
    assert "en" in LANGUAGES
    assert LANGUAGES["ja"] == "Japanese"
    assert LANGUAGES["en"] == "English"


@pytest.mark.asyncio
async def test_translate_text_with_cache():
    """キャッシュを使用したテキスト翻訳のテスト"""
    test_text = "Hello, World!"
    api_key = "test_key"

    # 最初の翻訳（キャッシュなし）
    translated = translate_text_with_cache(
        text=test_text,
        api_key=api_key,
        model="claude-3-5-haiku",
        source_lang="en",
        target_lang="ja",
    )
    assert translated
    assert isinstance(translated, str)

    # 2回目の翻訳（キャッシュあり）
    cached_translation = translate_text_with_cache(
        text=test_text,
        api_key=api_key,
        model="claude-3-5-haiku",
        source_lang="en",
        target_lang="ja",
    )
    assert cached_translation == translated
