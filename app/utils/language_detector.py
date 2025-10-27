#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
言語自動検出ユーティリティ（テキスト翻訳専用）

テキスト翻訳でのみ使用される言語自動判別機能を提供
ドキュメント翻訳では使用されません
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 言語パターン定義
LANGUAGE_PATTERNS = {
    'ja': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]',  # ひらがな、カタカナ、漢字
    'ko': r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]',  # ハングル
    'zh': r'[\u4E00-\u9FFF]',  # 中国語（漢字）
    'hi': r'[\u0900-\u097F]',  # デーヴァナーガリー文字
    'th': r'[\u0E00-\u0E7F]',  # タイ文字
    'vi': r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',  # ベトナム語
    'fr': r'[àâäéèêëïîôöùûüÿç]',  # フランス語
    'de': r'[äöüßÄÖÜ]',  # ドイツ語
    'es': r'[ñáéíóúü¿¡]'  # スペイン語
}

# 検出閾値（この割合以上の文字が含まれていれば、その言語と判定）
DETECTION_THRESHOLD = 0.3

def detect_language(text: str, threshold: float = DETECTION_THRESHOLD) -> str:
    """
    テキストから言語を自動判別（テキスト翻訳専用）
    
    注意: この関数はテキスト翻訳でのみ使用されます。
         ドキュメント翻訳では使用されません。
    
    Args:
        text: 判別対象のテキスト
        threshold: 判定閾値（デフォルト: 0.3 = 30%）
        
    Returns:
        言語コード（'ja', 'en', 'ko'等）
        判別できない場合は 'en' を返す
    """
    if not text or not text.strip():
        logger.debug("Empty text provided for language detection, defaulting to 'en'")
        return 'en'
    
    # 空白を除いた総文字数
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        logger.debug("No non-whitespace characters, defaulting to 'en'")
        return 'en'
    
    logger.debug(f"[Text Translation] Detecting language for text of length {total_chars}")
    
    # 各言語の文字数をカウント
    for lang_code, pattern in LANGUAGE_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        char_count = len(matches)
        
        if char_count > 0:
            ratio = char_count / total_chars
            logger.debug(f"[Text Translation] Language {lang_code}: {char_count}/{total_chars} = {ratio:.2%}")
            
            # 閾値以上であればその言語と判定
            if ratio >= threshold:
                logger.info(f"[Text Translation] Detected language: {lang_code} ({ratio:.2%})")
                return lang_code
    
    # どの言語も閾値に達しない場合は英語とする
    logger.debug("[Text Translation] No language reached threshold, defaulting to 'en'")
    return 'en'

def get_language_name(lang_code: str) -> str:
    """
    言語コードから言語名を取得
    
    Args:
        lang_code: 言語コード
        
    Returns:
        言語名
    """
    from app.core.translator import LANGUAGES
    return LANGUAGES.get(lang_code, lang_code)

def suggest_target_language(source_lang: str) -> str:
    """
    ソース言語に基づいて適切なターゲット言語を提案（テキスト翻訳専用）
    
    Args:
        source_lang: ソース言語コード
        
    Returns:
        推奨ターゲット言語コード
    """
    # 英語以外 → 英語
    # 英語 → 日本語
    if source_lang == 'en':
        return 'ja'
    else:
        return 'en'

def validate_language_pair(source_lang: str, target_lang: str) -> tuple[bool, Optional[str]]:
    """
    言語ペアの妥当性を検証
    
    Args:
        source_lang: ソース言語コード
        target_lang: ターゲット言語コード
        
    Returns:
        (is_valid, error_message)
    """
    from app.core.translator import LANGUAGES
    
    if source_lang not in LANGUAGES:
        return False, f"Unsupported source language: {source_lang}"
    
    if target_lang not in LANGUAGES:
        return False, f"Unsupported target language: {target_lang}"
    
    if source_lang == target_lang:
        return False, "Source and target languages must be different"
    
    return True, None

# エクスポート
__all__ = [
    'detect_language',
    'get_language_name',
    'suggest_target_language',
    'validate_language_pair',
    'LANGUAGE_PATTERNS',
    'DETECTION_THRESHOLD'
]
