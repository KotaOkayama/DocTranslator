#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PPTX/DOCX/PDF Text Translation Module

This module extracts text from PPTX/DOCX/PDF files, translates it using the GenAI Hub API, and re-imports the translated text into new files.
It maintains formatting and hyperlinks as much as possible.
"""

import json
import os
import time
import logging
import re
import subprocess
import tempfile
import shutil
from typing import Optional, Tuple, Callable, Dict, List

import PyPDF2
import pdfplumber
from docx import Document
from docx.shared import Inches
import pandas as pd

# dotenv import
from dotenv import load_dotenv

import pandas as pd
import requests

import concurrent.futures

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/translator.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# GenAI Hub API URL
def get_api_url():
    """Get API URL from environment variable or use default"""
    from app.config import get_api_url
    return get_api_url()

def get_api_key():
    """Get API key from environment variable"""
    from app.config import get_api_key
    return get_api_key()

# 翻訳に適さないモデルのフィルタリング用キーワード
EXCLUDED_MODEL_KEYWORDS = [
    "copilot",
    "documentation", 
    "jira",
    "ngsiem",
    "dom-data-extractor",
    "wiki",
    "compintel",
    "coding"
]

def filter_translation_models(models: List[str]) -> List[str]:
    """
    翻訳に適したモデルのみをフィルタリングします
    
    Args:
        models: モデル名のリスト
        
    Returns:
        翻訳に適したモデル名のリスト
    """
    filtered_models = []
    
    for model in models:
        model_lower = model.lower()
        
        # 除外キーワードが含まれているかチェック
        should_exclude = any(keyword in model_lower for keyword in EXCLUDED_MODEL_KEYWORDS)
        
        if not should_exclude:
            filtered_models.append(model)
    
    logger.info(f"モデルフィルタリング結果: {len(models)} -> {len(filtered_models)} モデル")
    logger.debug(f"除外されたモデル: {set(models) - set(filtered_models)}")
    
    return filtered_models

def format_model_name(model_name: str) -> str:
    """
    GenAI Hub特有のモデル名を適切に整形します
    
    Args:
        model_name: 元のモデル名
        
    Returns:
        整形されたモデル名
    """
    # Claude系モデルの整形
    if model_name.startswith("claude-"):
        # claude-3-5-sonnet-v2 -> Claude 3.5 Sonnet V2
        # claude-4-sonnet -> Claude 4 Sonnet
        # claude-3-7-sonnet -> Claude 3.7 Sonnet
        parts = model_name.split("-")
        if len(parts) >= 3:
            version = parts[1]  # 3, 4
            subversion = parts[2] if len(parts) > 2 else ""  # 5, 7
            model_type = parts[3] if len(parts) > 3 else parts[2]  # sonnet, haiku, opus
            variant = parts[4] if len(parts) > 4 else ""  # v2
            
            formatted = f"Claude {version}"
            if subversion and subversion.isdigit():
                formatted += f".{subversion}"
            formatted += f" {model_type.title()}"
            if variant:
                formatted += f" {variant.upper()}"
            
            return formatted
    
    # Llama系モデルの整形
    elif model_name.startswith("llama"):
        # llama3-1-70b -> Llama 3.1 70B
        # llama4-maverick-17b -> Llama 4 Maverick 17B
        parts = model_name.split("-")
        if len(parts) >= 2:
            version_part = parts[0].replace("llama", "")  # 3, 4
            
            formatted = f"Llama {version_part}"
            
            # バージョン番号の処理
            if len(parts) > 1 and parts[1].isdigit():
                formatted += f".{parts[1]}"
                remaining_parts = parts[2:]
            else:
                remaining_parts = parts[1:]
            
            # 残りの部分を処理
            for part in remaining_parts:
                if part.endswith("b") and part[:-1].isdigit():
                    # サイズ情報 (70b -> 70B)
                    formatted += f" {part.upper()}"
                else:
                    # その他の情報 (maverick, scout等)
                    formatted += f" {part.title()}"
            
            return formatted
    
    # その他のモデルは最初の文字を大文字にして返す
    return model_name.replace("-", " ").title()

def fetch_available_models() -> Dict[str, str]:
    """
    GenAI HUBから利用可能なモデル一覧を取得します
    
    Returns:
        モデル辞書 {model_id: display_name}
        
    Raises:
        ValueError: API呼び出しに失敗した場合
        ConnectionError: 接続に失敗した場合
    """
    logger.info("GenAI HUBからモデル一覧を取得しています...")
    
    # API設定を取得
    api_key = get_api_key()
    api_url = get_api_url()
    
    if not api_key:
        raise ValueError("API キーが設定されていません。設定画面でAPI キーを設定してください。")
    
    if not api_url:
        raise ValueError("API URLが設定されていません。設定画面でAPI URLを設定してください。")
    
    # モデル一覧取得用のエンドポイントを構築
    # チャットエンドポイントからモデル一覧エンドポイントに変換
    if api_url.endswith("/chat/completions"):
        models_url = api_url.replace("/chat/completions", "/models")
    elif api_url.endswith("/v1/chat/completions"):
        models_url = api_url.replace("/v1/chat/completions", "/v1/models")
    else:
        # フォールバック: URLの末尾に/modelsを追加
        models_url = api_url.rstrip("/") + "/models"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        logger.debug(f"モデル一覧取得URL: {models_url}")
        response = requests.get(
            models_url,
            headers=headers,
            timeout=30
        )
        
        logger.debug(f"API レスポンスステータス: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"モデル一覧取得エラー: ステータスコード {response.status_code}")
            logger.error(f"レスポンス: {response.text}")
            raise ValueError(f"モデル一覧の取得に失敗しました。ステータスコード: {response.status_code}")
        
        result = response.json()
        logger.debug(f"API レスポンス: {result}")
        
        # レスポンス形式を判定して適切にパース
        models_list = None
        
        # 新しい形式（OpenAI互換）: data フィールドを使用
        if "data" in result and isinstance(result["data"], list):
            logger.debug("OpenAI互換形式のレスポンスを検出")
            models_list = [model["id"] for model in result["data"] if "id" in model]
        
        # 旧形式: content フィールドを使用（後方互換性のため）
        elif "content" in result and isinstance(result["content"], list):
            logger.debug("旧形式のレスポンスを検出")
            models_list = result["content"]
        
        # どちらの形式でもない場合
        else:
            logger.error(f"未知のレスポンス形式: {result}")
            raise ValueError("モデル一覧の取得に失敗しました。レスポンス形式が不正です。")
        
        if not isinstance(models_list, list):
            logger.error(f"モデル一覧がリスト形式ではありません: {type(models_list)}")
            raise ValueError("モデル一覧の取得に失敗しました。レスポンス形式が不正です。")
        
        logger.info(f"取得したモデル数: {len(models_list)}")
        logger.debug(f"取得したモデル: {models_list}")
        
        # 翻訳に適したモデルのみをフィルタリング
        filtered_models = filter_translation_models(models_list)
        
        if not filtered_models:
            logger.warning("翻訳に適したモデルが見つかりませんでした")
            raise ValueError("翻訳に適したモデルが見つかりませんでした。")
        
        # モデル辞書を作成し、アルファベット順にソート
        models_dict = {}
        for model in filtered_models:
            display_name = format_model_name(model)
            models_dict[model] = display_name
        
        # モデルIDでアルファベット順にソート
        sorted_models_dict = dict(sorted(models_dict.items()))
        
        logger.info(f"利用可能な翻訳モデル（アルファベット順）: {list(sorted_models_dict.keys())}")
        
        return sorted_models_dict
        
    except requests.exceptions.RequestException as e:
        logger.error(f"モデル一覧取得リクエストエラー: {e}")
        raise ConnectionError(f"GenAI HUBへの接続に失敗しました: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"JSON デコードエラー: {e}")
        logger.error(f"レスポンス: {response.text if 'response' in locals() else 'なし'}")
        raise ValueError(f"モデル一覧の取得に失敗しました。レスポンスの解析エラー: {str(e)}")
    except Exception as e:
        logger.error(f"モデル一覧取得エラー: {e}")
        raise ValueError(f"モデル一覧の取得に失敗しました: {str(e)}")



# フォールバック用のデフォルトモデル（API呼び出しが失敗した場合に使用）
# DEFAULT_MODELS = {
#     "claude-3-5-haiku": "Claude 3.5 Haiku",
#     "claude-3-5-sonnet-v2": "Claude 3.5 Sonnet V2", 
#     "claude-3-7-sonnet": "Claude 3.7 Sonnet",
#     "claude-4-sonnet": "Claude 4 Sonnet",
# }

# アルファベット順にソート
# DEFAULT_MODELS = dict(sorted(DEFAULT_MODELS.items()))

def get_available_models() -> Dict[str, str]:
    """
    利用可能なモデル一覧を取得します（フォールバック無し）
    
    Returns:
        モデル辞書 {model_id: display_name}
    Raises:
        ValueError: API呼び出しに失敗した場合
        ConnectionError: 接続に失敗した場合
    """

    return fetch_available_models()

# Language options
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

# Translation Settings
TRANSLATION_CONFIG = {
    "chunk_size": 1500,  # Maximum length of text to split
    "batch_size": 5,  # Maximum number of texts to batch process
    "parallel_workers": 4,  # Maximum number of parallel processes
    "use_cache": True,  # Whether to use cache
    "optimize_text": True,  # Whether to optimize text
    "api_timeout": 30,  # API call timeout (seconds)
    "retry_count": 2,  # Number of retries on error
}

# Global Translation Cache
TRANSLATION_CACHE = {}

def save_translation_cache(cache_file: str = "translation_cache.json"):
    """Save translation cache to file"""
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(TRANSLATION_CACHE, f, ensure_ascii=False, indent=2)
        logger.info(f"Translation cache saved: {len(TRANSLATION_CACHE)} entries")
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")

def load_translation_cache(cache_file: str = "translation_cache.json"):
    """Loading translation cache from file"""
    global TRANSLATION_CACHE
    try:
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                TRANSLATION_CACHE = json.load(f)
            logger.info(
                f"翻訳キャッシュをロードしました: {len(TRANSLATION_CACHE)} entries"
            )
    except Exception as e:
        logger.error(f"キャッシュのロードに失敗しました: {e}")
        TRANSLATION_CACHE = {}

# キャッシュをロード
load_translation_cache()

def translate_text_with_genai_hub(
    text: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    ai_instruction: str = "",
) -> str:
    """
    GenAI Hub API を使用してテキストを翻訳します。
    """
    if not text.strip():
        return ""

    # 環境変数からAPI キーとURLを取得
    if not api_key:
        api_key = os.environ.get("GENAI_HUB_API_KEY")

    # API URLを環境変数から取得
    api_url = os.environ.get("GENAI_HUB_API_URL")

    if not api_key:
        raise ValueError(
            "The API key is not set. Please check the environment variable GENAI_HUB_API_KEY."
        )
    
    if not api_url:
        raise ValueError(
            "The API URL is not set. Please check the environment variable GENAI_HUB_API_URL."
        )

    # Log only the first and last few characters of the API key (for security purposes)
    masked_api_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    logger.info(
        f"翻訳: {source_lang} -> {target_lang}, モデル: {model}, API キー: {masked_api_key}, API URL: {api_url}"
    )
    logger.debug(f"翻訳テキスト: {text[:100]}...")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    # Create a prompt that includes translation instructions
    prompt = f"""以下のテキストを {LANGUAGES.get(source_lang, source_lang)} から {LANGUAGES.get(target_lang, target_lang)} に翻訳してください。
元のテキストのフォーマットを維持し、翻訳のみを行ってください。
翻訳以外の説明や注釈は不要です。
改行やスペースなどの書式も可能な限り維持してください。
"""

    # 補足指示があれば追加
    if ai_instruction:
        prompt += f"\n補足指示: {ai_instruction}\n"

    prompt += f"""
テキスト:
{text}

翻訳:"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Set a low temperature for consistent translation
        "max_tokens": 4000,
    }

    # Retry counter
    retry_count = 0
    max_retries = TRANSLATION_CONFIG["retry_count"]

    while retry_count <= max_retries:
        try:
            logger.debug(f"API リクエスト送信: {api_url}")
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=TRANSLATION_CONFIG["api_timeout"],
            )

            # レスポンスの詳細をログに記録
            logger.debug(f"API レスポンスステータス: {response.status_code}")

            # エラーチェック
            if response.status_code != 200:
                logger.error(f"API エラー: ステータスコード {response.status_code}")
                logger.error(f"レスポンス: {response.text}")

                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2**retry_count  # 指数バックオフ
                    logger.info(
                        f"リトライ {retry_count}/{max_retries} を {wait_time} 秒後に実行します"
                    )
                    time.sleep(wait_time)
                    continue

                return f"[翻訳エラー: API ステータスコード {response.status_code}]"

            response.raise_for_status()

            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                logger.error(f"API レスポンスに 'choices' がありません: {result}")

                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2**retry_count
                    logger.info(
                        f"リトライ {retry_count}/{max_retries} を {wait_time} 秒後に実行します"
                    )
                    time.sleep(wait_time)
                    continue

                return "[翻訳エラー: 無効なレスポンス]"

            translated_text = result["choices"][0]["message"]["content"].strip()

            # "翻訳:" などのプレフィックスがある場合は削除
            if "翻訳:" in translated_text:
                translated_text = translated_text.split("翻訳:", 1)[1].strip()

            logger.debug(f"翻訳結果: {translated_text[:100]}...")
            return translated_text

        except requests.exceptions.RequestException as e:
            logger.error(f"API リクエストエラー: {e}")

            if retry_count < max_retries:
                retry_count += 1
                wait_time = 2**retry_count
                logger.info(
                    f"リトライ {retry_count}/{max_retries} を {wait_time} 秒後に実行します"
                )
                time.sleep(wait_time)
                continue

            return f"[翻訳エラー: リクエスト失敗 - {str(e)}]"

        except json.JSONDecodeError as e:
            logger.error(f"JSON デコードエラー: {e}")
            logger.error(
                f"レスポンス: {response.text if 'response' in locals() else 'なし'}"
            )

            if retry_count < max_retries:
                retry_count += 1
                wait_time = 2**retry_count
                logger.info(
                    f"リトライ {retry_count}/{max_retries} を {wait_time} 秒後に実行します"
                )
                time.sleep(wait_time)
                continue

            return "[翻訳エラー: 無効な JSON レスポンス]"
        except Exception as e:
            logger.error(f"翻訳エラー: {e}")
            logger.error(
                f"レスポンス: {response.text if 'response' in locals() else 'なし'}"
            )

            if retry_count < max_retries:
                retry_count += 1
                wait_time = 2**retry_count
                logger.info(
                    f"リトライ {retry_count}/{max_retries} を {wait_time} 秒後に実行します"
                )
                time.sleep(wait_time)
                continue

            return f"[翻訳エラー: {str(e)}]"

def translate_text_with_cache(
    text: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    ai_instruction: str = "",
) -> str:
    """
    キャッシュを使用してテキストを翻訳します。

    Args:
        text: 翻訳するテキスト
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        ai_instruction: AIへの補足指示

    Returns:
        翻訳されたテキスト
    """
    if not text.strip():
        return ""

    # キャッシュキーを作成（補足指示も含める）
    cache_key = f"{text}|{model}|{source_lang}|{target_lang}|{ai_instruction}"

    # キャッシュをチェック
    if cache_key in TRANSLATION_CACHE:
        logger.debug(f"キャッシュから翻訳を取得: {text[:30]}...")
        return TRANSLATION_CACHE[cache_key]

    # キャッシュにない場合は翻訳
    translated = translate_text_with_genai_hub(
        text, api_key, model, source_lang, target_lang, ai_instruction
    )

    # キャッシュに保存
    TRANSLATION_CACHE[cache_key] = translated

    return translated

def optimize_text_for_translation(text: str) -> str:
    """
    翻訳のためにテキストを最適化します。

    Args:
        text: 最適化するテキスト

    Returns:
        最適化されたテキスト
    """
    # 空白行を削除
    lines = [line for line in text.split("\n") if line.strip()]

    # 数字や記号だけの行をスキップ
    filtered_lines = []
    for line in lines:
        # 翻訳が必要なテキストかチェック（アルファベット、漢字、ひらがな、カタカナなどを含む）
        if any(c.isalpha() for c in line):
            filtered_lines.append(line)
        else:
            # 数字や記号だけの行はそのまま追加
            filtered_lines.append(line)

    return "\n".join(filtered_lines)

def translate_dataframe(
    df: pd.DataFrame,
    api_key: str,
    model: str,
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> pd.DataFrame:
    """
    DataFrameのテキストを並列翻訳します。

    Args:
        df: 翻訳するテキストを含むDataFrame
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        progress_callback: 進捗を報告するコールバック関数
        ai_instruction: AIへの補足指示
        check_cancelled: 中止状態をチェックする関数

    Returns:
        翻訳されたテキストを含むDataFrame
    """
    logger.info(f"DataFrame 内のテキストを並列翻訳します: {len(df)} 項目")

    # テキストを最適化
    if TRANSLATION_CONFIG["optimize_text"]:
        optimized_texts = [
            optimize_text_for_translation(text) for text in df["original_text"]
        ]
    else:
        optimized_texts = df["original_text"].tolist()

    # 並列処理の設定
    max_workers = TRANSLATION_CONFIG.get("parallel_workers", 4)
    batch_size = TRANSLATION_CONFIG.get("batch_size", 5)

    # 進捗状況を追跡するためのカウンター
    total_items = len(df)
    completed_items = 0
    translations = [""] * total_items

    # スレッドセーフな進捗更新関数
    def update_progress(count=1):
        nonlocal completed_items
        completed_items += count
        if progress_callback:
            try:
                progress = 0.3 + (completed_items / total_items * 0.5)
                progress_callback(min(progress, 0.8))
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")

    # 単一テキスト翻訳のラッパー関数
    def translate_single_text(index, text):
        try:
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info(f"翻訳が中止されました (項目 {index+1})")
                return index, "[翻訳中止]"

            translated_text = translate_text_with_cache(
                text, api_key, model, source_lang, target_lang, ai_instruction
            )
            return index, translated_text
        except Exception as e:
            logger.error(f"テキスト翻訳エラー (項目 {index+1}): {e}")
            return index, f"[翻訳エラー: {str(e)}]"

    # バッチ処理を使用した並列翻訳
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # バッチごとに処理
        for batch_start in range(0, total_items, batch_size):
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info("翻訳が中止されました")
                break

            batch_end = min(batch_start + batch_size, total_items)
            batch_texts = optimized_texts[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            # バッチの翻訳タスクを送信
            batch_futures = [
                executor.submit(translate_single_text, index, text)
                for index, text in zip(batch_indices, batch_texts)
            ]

            # 結果を収集
            for future in concurrent.futures.as_completed(batch_futures):
                # 中止チェック
                if check_cancelled and check_cancelled():
                    logger.info("翻訳が中止されました")
                    break

                try:
                    index, translated_text = future.result()
                    translations[index] = translated_text
                    update_progress()
                except Exception as e:
                    logger.error(f"バッチ処理エラー: {e}")

    # 翻訳結果をDataFrameに追加
    df["translated_text"] = translations

    logger.info(f"並列翻訳完了: {len(df)} 項目")

    if progress_callback:
        try:
            progress_callback(0.8)  # 翻訳完了
        except Exception as e:
            logger.warning(f"進捗コールバックエラー: {e}")

    return df

def save_text_files(df: pd.DataFrame, output_dir: str, base_filename: str) -> tuple:
    """
    抽出したテキストと翻訳したテキストをファイルに保存します。

    Args:
        df: テキストデータを含む DataFrame
        output_dir: 出力ディレクトリ
        base_filename: 基本ファイル名

    Returns:
        (抽出テキストファイルパス, 翻訳テキストファイルパス)
    """
    logger.info(
        f"テキストファイルを保存しています: ディレクトリ {output_dir}, ベース名 {base_filename}"
    )

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # ファイル名を生成
    extracted_file = os.path.join(output_dir, f"{base_filename}_extracted.txt")
    translated_file = os.path.join(output_dir, f"{base_filename}_translated.txt")

    # 抽出テキストを保存
    with open(extracted_file, "w", encoding="utf-8") as f:
        for i, row in df.iterrows():
            # ファイル形式に応じたフォーマット
            if "element_type" in row:
                if row["element_type"] == "paragraph":
                    f.write(f"=== 段落 {row.get('para_id', i) + 1} ===\n")
                elif row["element_type"] == "table_cell":
                    f.write(
                        f"=== 表 {row.get('table_id', 0) + 1}, 行 {row.get('row_id', 0) + 1}, セル {row.get('cell_id', 0) + 1} ===\n"
                    )
                elif row["element_type"] == "pdf_text":
                    f.write(
                        f"=== ページ {row.get('page_num', 0) + 1}, ブロック {row.get('block_num', 0) + 1}, 行 {row.get('line_num', 0) + 1} ===\n"
                    )
                elif row["element_type"] == "pdf_paragraph":  # 追加
                    f.write(
                        f"=== ページ {row.get('page_num', 0) + 1}, 段落 {row.get('para_num', 0) + 1} ===\n"
                    )
            elif "slide_num" in row:
                f.write(
                    f"=== スライド {row['slide_num'] + 1}, 形状 {row['shape_id'] + 1} ===\n"
                )
            else:
                f.write(f"=== 項目 {i + 1} ===\n")

            # 元のテキストを書き込み（この行が抜けていました）
            f.write(f"{row['original_text']}\n\n")

            # ハイパーリンク情報があれば追加
            if "hyperlink" in row and row["hyperlink"]:
                f.write(f"[ハイパーリンク: {row['hyperlink']}]\n\n")

    logger.info(f"抽出テキストを保存しました: {extracted_file}")

    # 翻訳テキストを保存（'translated_text'列がある場合のみ）
    if "translated_text" in df.columns:
        # 翻訳テキストが空でないか確認
        if (
            not df["translated_text"].isna().all()
            and not (df["translated_text"] == "").all()
        ):
            with open(translated_file, "w", encoding="utf-8") as f:
                for i, row in df.iterrows():
                    # ファイル形式に応じたフォーマット
                    if "element_type" in row:
                        if row["element_type"] == "paragraph":
                            f.write(f"=== 段落 {row.get('para_id', i) + 1} ===\n")
                        elif row["element_type"] == "table_cell":
                            f.write(
                                f"=== 表 {row.get('table_id', 0) + 1}, 行 {row.get('row_id', 0) + 1}, セル {row.get('cell_id', 0) + 1} ===\n"
                            )
                        elif row["element_type"] == "pdf_text":
                            f.write(
                                f"=== ページ {row.get('page_num', 0) + 1}, ブロック {row.get('block_num', 0) + 1}, 行 {row.get('line_num', 0) + 1} ===\n"
                            )
                        elif row["element_type"] == "pdf_paragraph":  # 追加
                            f.write(
                                f"=== ページ {row.get('page_num', 0) + 1}, 段落 {row.get('para_num', 0) + 1} ===\n"
                            )
                    elif "slide_num" in row:
                        f.write(
                            f"=== スライド {row['slide_num'] + 1}, 形状 {row['shape_id'] + 1} ===\n"
                        )
                    else:
                        f.write(f"=== 項目 {i + 1} ===\n")

                    # 翻訳テキストがNoneまたは空文字列の場合は元のテキストを使用
                    text_to_write = (
                        row["translated_text"]
                        if pd.notna(row["translated_text"]) and row["translated_text"]
                        else "[未翻訳]"
                    )
                    f.write(f"{text_to_write}\n\n")

                    # ハイパーリンク情報があれば追加
                    if "hyperlink" in row and row["hyperlink"]:
                        f.write(f"[ハイパーリンク: {row['hyperlink']}]\n\n")

            logger.info(f"翻訳テキストを保存しました: {translated_file}")
            return extracted_file, translated_file
        else:
            logger.warning(
                "翻訳テキストが空です。翻訳テキストファイルは保存されません。"
            )
            return extracted_file, None
    else:
        logger.warning(
            "DataFrame に 'translated_text' 列がありません。翻訳テキストは保存されません。"
        )
        return extracted_file, None

#
# PPTX 処理関数
#

def extract_text_from_pptx(
    pptx_path: str, progress_callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    PPTX ファイルからテキストを抽出し、DataFrame として返します。

    Args:
        pptx_path: PPTX ファイルのパス
        progress_callback: 進捗を報告するコールバック関数

    Returns:
        テキストデータを含む DataFrame
    """
    logger.info(f"PPTX ファイル '{pptx_path}' からテキストを抽出しています...")

    try:
        from pptx import Presentation

        prs = Presentation(pptx_path)
        text_data = []

        total_slides = len(prs.slides)
        logger.info(f"スライド数: {total_slides}")

        # スライド番号、形状ID、元のテキストを保存
        for slide_num, slide in enumerate(prs.slides):
            if progress_callback:
                try:
                    progress_callback(
                        slide_num / total_slides * 0.3
                    )  # 抽出は全体の30%と想定
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")

            for shape_id, shape in enumerate(slide.shapes):
                if hasattr(shape, "text") and shape.text.strip():
                    text = shape.text.strip()
                    logger.debug(
                        f"スライド {slide_num+1}, 形状 {shape_id+1}: {text[:50]}..."
                    )

                    # ハイパーリンク情報を取得
                    hyperlink = None
                    if hasattr(shape, "click_action") and hasattr(
                        shape.click_action, "hyperlink"
                    ):
                        if (
                            hasattr(shape.click_action.hyperlink, "address")
                            and shape.click_action.hyperlink.address
                        ):
                            hyperlink = shape.click_action.hyperlink.address

                    text_data.append(
                        {
                            "slide_num": slide_num,
                            "shape_id": shape_id,
                            "original_text": text,
                            "hyperlink": hyperlink,
                        }
                    )

        # データフレームに変換
        df = pd.DataFrame(text_data)

        logger.info(f"抽出完了: {len(df)} 個のテキスト要素が見つかりました")

        if progress_callback:
            try:
                progress_callback(0.3)  # 抽出完了
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")

        return df

    except ImportError:
        logger.error("python-pptxがインストールされていません")
        raise ImportError(
            "python-pptxがインストールされていません。pip install python-pptx でインストールしてください。"
        )
    except Exception as e:
        logger.error(f"PPTXからのテキスト抽出エラー: {e}")
        raise

def reimport_text_to_pptx(
    pptx_path: str,
    df: pd.DataFrame,
    output_path: str,
    progress_callback: Optional[Callable] = None,
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> None:
    """
    翻訳されたテキストを PPTX ファイルに再インポートします。
    書式とハイパーリンクを可能な限り維持します。

    Args:
        pptx_path: 元の PPTX ファイルのパス
        df: 翻訳されたテキストを含む DataFrame
        output_path: 出力 PPTX ファイルのパス
        progress_callback: 進捗を報告するコールバック関数
        check_cancelled: 中止状態をチェックする関数
    """
    logger.info(
        f"翻訳されたテキストを PPTX ファイルに再インポートしています: {output_path}"
    )

    try:
        from pptx import Presentation

        # プレゼンテーションを開く
        prs = Presentation(pptx_path)

        total_items = len(df)

        # 各スライドと形状に翻訳テキストを適用
        for i, (index, row) in enumerate(df.iterrows()):
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info("PPTX再インポートが中止されました")
                return

            if progress_callback:
                try:
                    # 再インポートは全体の80%〜100%と想定
                    progress = 0.8 + (i / total_items * 0.2)
                    progress_callback(progress)
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")

            slide_num = row["slide_num"]
            shape_id = row["shape_id"]
            translated_text = row["translated_text"]
            hyperlink = row.get("hyperlink")

            logger.debug(
                f"スライド {slide_num+1}, 形状 {shape_id+1} に翻訳テキストを適用中..."
            )

            # 該当するスライドと形状を特定してテキストを置換
            if slide_num < len(prs.slides):
                slide = prs.slides[slide_num]
                shapes = list(slide.shapes)
                if shape_id < len(shapes):
                    shape = shapes[shape_id]
                    if hasattr(shape, "text_frame"):
                        # 翻訳テキストを段落ごとに分割
                        translated_paragraphs = translated_text.split("\n")
                        original_paragraphs = [p for p in shape.text_frame.paragraphs]

                        # 既存の段落数と翻訳後の段落数を比較
                        if len(original_paragraphs) >= len(translated_paragraphs):
                            # 既存の段落に翻訳テキストを適用（書式を維持）
                            for p_idx, p_text in enumerate(translated_paragraphs):
                                if p_idx < len(original_paragraphs):
                                    # 元の段落の書式を保持しながらテキストを置換
                                    original_p = original_paragraphs[p_idx]

                                    # 段落内のランの処理
                                    if len(original_p.runs) > 0:
                                        # 最初のランにテキストを設定
                                        original_p.runs[0].text = p_text

                                        # 残りのランを空にする
                                        for run in original_p.runs[1:]:
                                            run.text = ""
                                    else:
                                        # ランがない場合は段落のテキストを直接設定
                                        original_p.text = p_text

                            # 余分な段落を空にする
                            for p_idx in range(
                                len(translated_paragraphs), len(original_paragraphs)
                            ):
                                original_paragraphs[p_idx].text = ""
                        else:
                            # 翻訳後の段落数が多い場合
                            # 既存の段落を更新
                            for p_idx, original_p in enumerate(original_paragraphs):
                                if p_idx < len(translated_paragraphs):
                                    if len(original_p.runs) > 0:
                                        original_p.runs[0].text = translated_paragraphs[
                                            p_idx
                                        ]
                                        for run in original_p.runs[1:]:
                                            run.text = ""
                                    else:
                                        original_p.text = translated_paragraphs[p_idx]

                            # 不足している段落を追加
                            for p_idx in range(
                                len(original_paragraphs), len(translated_paragraphs)
                            ):
                                p = shape.text_frame.add_paragraph()
                                p.text = translated_paragraphs[p_idx]

                                # 可能であれば最初の段落のスタイルをコピー
                                if original_paragraphs and hasattr(
                                    original_paragraphs[0], "alignment"
                                ):
                                    p.alignment = original_paragraphs[0].alignment

                        # ハイパーリンクの処理
                        if hyperlink:
                            try:
                                # python-pptxの制限により、完全なハイパーリンクの再設定は難しい場合がある
                                # 可能な場合は、クリックアクションのハイパーリンクを設定
                                if hasattr(shape, "click_action"):
                                    logger.debug(f"ハイパーリンクを維持: {hyperlink}")
                                    shape.click_action.hyperlink.address = hyperlink
                            except Exception as e:
                                logger.warning(
                                    f"ハイパーリンクの設定に失敗しました: {e}"
                                )

        # 新しいファイルとして保存
        logger.info(f"翻訳された PPTX ファイルを保存しています: {output_path}")
        prs.save(output_path)

        if progress_callback:
            try:
                progress_callback(1.0)  # 完了
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")

    except Exception as e:
        logger.error(f"PPTX再インポートエラー: {e}")
        raise

#
# DOCX 処理関数
#

def extract_text_from_docx(
    docx_path: str, progress_callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    Word(docx)ファイルからテキストを抽出し、DataFrameとして返します。

    Args:
        docx_path: Wordファイルのパス
        progress_callback: 進捗を報告するコールバック関数

    Returns:
        テキストデータを含むDataFrame
    """
    logger.info(f"Word文書 '{docx_path}' からテキストを抽出しています...")

    try:
        from docx import Document

        doc = Document(docx_path)
        text_data = []

        # 段落と表の総数を計算（進捗表示用）
        total_elements = len(doc.paragraphs)
        for table in doc.tables:
            for row in table.rows:
                total_elements += len(row.cells)

        logger.info(f"要素数: {total_elements}")

        # 段落からテキストを抽出
        element_count = 0
        for para_id, para in enumerate(doc.paragraphs):
            if progress_callback:
                try:
                    progress_callback(
                        element_count / total_elements * 0.3
                    )  # 抽出は全体の30%と想定
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")
                element_count += 1

            text = para.text.strip()
            if text:
                logger.debug(f"段落 {para_id+1}: {text[:50]}...")

                # スタイル情報を取得
                style_name = para.style.name if para.style else "Normal"
                alignment = para.alignment if hasattr(para, "alignment") else None

                # 段落内のランの書式情報を収集
                runs_info = []
                for run_id, run in enumerate(para.runs):
                    if run.text.strip():
                        run_info = {
                            "text": run.text,
                            "bold": run.bold,
                            "italic": run.italic,
                            "underline": run.underline,
                        }
                        runs_info.append(run_info)

                text_data.append(
                    {
                        "element_type": "paragraph",
                        "para_id": int(para_id),  # 確実に整数を使用
                        "original_text": text,
                        "style_name": style_name,
                        "alignment": alignment,
                        "runs_info": runs_info,
                        "table_id": None,
                        "row_id": None,
                        "cell_id": None,
                    }
                )

        # 表からテキストを抽出
        for table_id, table in enumerate(doc.tables):
            for row_id, row in enumerate(table.rows):
                for cell_id, cell in enumerate(row.cells):
                    if progress_callback:
                        try:
                            progress_callback(element_count / total_elements * 0.3)
                        except Exception as e:
                            logger.warning(f"進捗コールバックエラー: {e}")
                        element_count += 1

                    text = cell.text.strip()
                    if text:
                        logger.debug(
                            f"表 {table_id+1}, 行 {row_id+1}, セル {cell_id+1}: {text[:50]}..."
                        )

                        # セル内の段落のスタイル情報を収集
                        cell_paras_info = []
                        for p_id, p in enumerate(cell.paragraphs):
                            if p.text.strip():
                                para_info = {
                                    "text": p.text,
                                    "style_name": p.style.name if p.style else "Normal",
                                    "alignment": (
                                        p.alignment if hasattr(p, "alignment") else None
                                    ),
                                }
                                cell_paras_info.append(para_info)

                        text_data.append(
                            {
                                "element_type": "table_cell",
                                "para_id": None,
                                "original_text": text,
                                "style_name": None,
                                "alignment": None,
                                "runs_info": None,
                                "table_id": int(table_id),  # 確実に整数を使用
                                "row_id": int(row_id),  # 確実に整数を使用
                                "cell_id": int(cell_id),  # 確実に整数を使用
                                "cell_paras_info": cell_paras_info,
                            }
                        )

        # データフレームに変換
        df = pd.DataFrame(text_data)

        logger.info(f"抽出完了: {len(df)} 個のテキスト要素が見つかりました")

        if progress_callback:
            try:
                progress_callback(0.3)  # 抽出完了
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")

        return df

    except ImportError:
        logger.error("python-docxがインストールされていません")
        raise ImportError(
            "python-docxがインストールされていません。pip install python-docx でインストールしてください。"
        )
    except Exception as e:
        logger.error(f"DOCXからのテキスト抽出エラー: {e}")
        raise

def reimport_text_to_docx(
    docx_path: str,
    df: pd.DataFrame,
    output_path: str,
    progress_callback: Optional[Callable] = None,
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> None:
    """
    翻訳されたテキストをWord文書に再インポートします。
    書式を可能な限り維持します。

    Args:
        docx_path: 元のWordファイルのパス
        df: 翻訳されたテキストを含むDataFrame
        output_path: 出力Wordファイルのパス
        progress_callback: 進捗を報告するコールバック関数
        check_cancelled: 中止状態をチェックする関数
    """
    logger.info(f"翻訳されたテキストをWord文書に再インポートしています: {output_path}")

    try:
        from docx import Document

        # 文書を開く
        doc = Document(docx_path)

        total_items = len(df)

        # 段落の翻訳テキストを適用
        para_df = df[df["element_type"] == "paragraph"]
        for i, (_, row) in enumerate(para_df.iterrows()):
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info("DOCX再インポートが中止されました")
                return

            if progress_callback:
                try:
                    # 再インポートは全体の80%〜100%と想定
                    progress = 0.8 + (i / total_items * 0.2)
                    progress_callback(progress)
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")

            # para_idが浮動小数点数の場合は整数に変換
            try:
                para_id = int(row["para_id"])
            except (TypeError, ValueError):
                logger.warning(
                    f"段落ID '{row['para_id']}' を整数に変換できません。スキップします。"
                )
                continue

            translated_text = row["translated_text"]

            logger.debug(f"段落 {para_id+1} に翻訳テキストを適用中...")

            # 該当する段落を特定してテキストを置換
            if para_id < len(doc.paragraphs):
                para = doc.paragraphs[para_id]

                # 段落のテキストをクリア
                for run in para.runs:
                    run.text = ""

                # 翻訳テキストを追加
                if para.runs:
                    # 既存のランがある場合は最初のランにテキストを設定
                    para.runs[0].text = translated_text
                else:
                    # ランがない場合は新しいランを追加
                    run = para.add_run(translated_text)

                    # 元の書式情報があれば適用
                    if (
                        "runs_info" in row
                        and row["runs_info"]
                        and len(row["runs_info"]) > 0
                    ):
                        run_info = row["runs_info"][0]
                        if run_info.get("bold"):
                            run.bold = True
                        if run_info.get("italic"):
                            run.italic = True
                        if run_info.get("underline"):
                            run.underline = True

        # 表のセルの翻訳テキストを適用
        table_df = df[df["element_type"] == "table_cell"]
        for i, (_, row) in enumerate(table_df.iterrows()):
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info("DOCX再インポートが中止されました")
                return

            if progress_callback:
                try:
                    # 再インポートは全体の80%〜100%と想定
                    progress = 0.8 + ((len(para_df) + i) / total_items * 0.2)
                    progress_callback(progress)
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")

            # 各IDを整数に変換
            try:
                table_id = int(row["table_id"])
                row_id = int(row["row_id"])
                cell_id = int(row["cell_id"])
            except (TypeError, ValueError):
                logger.warning(
                    f"表ID '{row['table_id']}', 行ID '{row['row_id']}', セルID '{row['cell_id']}' を整数に変換できません。スキップします。"
                )
                continue

            translated_text = row["translated_text"]

            logger.debug(
                f"表 {table_id+1}, 行 {row_id+1}, セル {cell_id+1} に翻訳テキストを適用中..."
            )

            # 該当するセルを特定してテキストを置換
            try:
                if table_id < len(doc.tables):
                    table = doc.tables[table_id]
                    if row_id < len(table.rows):
                        row_obj = table.rows[row_id]
                        if cell_id < len(row_obj.cells):
                            cell = row_obj.cells[cell_id]

                            # セルのテキストをクリア
                            cell.text = ""

                            # 翻訳テキストを追加
                            # セル内の段落情報があれば、それに基づいて段落を作成
                            if "cell_paras_info" in row and row["cell_paras_info"]:
                                paragraphs = translated_text.split("\n")
                                for p_idx, p_text in enumerate(paragraphs):
                                    if p_idx == 0:
                                        # 最初の段落は既に存在する
                                        p = cell.paragraphs[0]
                                    else:
                                        # 追加の段落を作成
                                        p = cell.add_paragraph()

                                    p.text = p_text

                                    # 元の段落情報があれば、スタイルを適用
                                    if p_idx < len(row["cell_paras_info"]):
                                        para_info = row["cell_paras_info"][p_idx]
                                        if para_info.get("alignment") is not None:
                                            p.alignment = para_info["alignment"]
                            else:
                                # 段落情報がない場合は単純にテキストを設定
                                cell.text = translated_text
            except Exception as e:
                logger.warning(f"表セルの更新中にエラーが発生しました: {e}")
                continue

        # 新しいファイルとして保存
        logger.info(f"翻訳されたWord文書を保存しています: {output_path}")
        doc.save(output_path)

        if progress_callback:
            try:
                progress_callback(1.0)  # 完了
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")

    except Exception as e:
        logger.error(f"DOCX再インポートエラー: {e}")
        raise

#
# PDF 処理関数（translator.py.old.pyの処理フローに合わせて修正）
#

def convert_pdf_to_docx(pdf_path: str, docx_path: str) -> str:
    """
    PDFファイルをDOCXに変換します。
    pdf2docxライブラリを優先し、失敗時はLibreOfficeを使用します。
    
    Args:
        pdf_path: 入力PDFファイルのパス
        docx_path: 出力DOCXファイルのパス
        
    Returns:
        変換されたDOCXファイルのパス
    """
    logger.info(f"PDFをDOCXに変換しています: {pdf_path} -> {docx_path}")
    
    # 方法1: pdf2docxライブラリを使用
    try:
        from pdf2docx import Converter
        
        logger.info("pdf2docxを使用してPDFをDOCXに変換しています...")
        
        # PDFをDOCXに変換
        cv = Converter(pdf_path)
        cv.convert(docx_path)
        cv.close()
        
        # 変換結果を確認
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
            logger.info(f"pdf2docxによるPDF→DOCX変換が完了しました: {docx_path}")
            return docx_path
        else:
            logger.warning("pdf2docxでDOCXが作成されませんでした")
            raise Exception("DOCXが作成されませんでした")
            
    except ImportError:
        logger.warning("pdf2docxがインストールされていません。LibreOfficeを試行します。")
    except Exception as e:
        logger.warning(f"pdf2docxによるPDF→DOCX変換に失敗しました: {e}. LibreOfficeを試行します。")
    
    # 方法2: LibreOfficeを使用してPDFをDOCXに変換
    try:
        logger.info("LibreOfficeを使用してPDFをDOCXに変換しています...")
        
        # LibreOfficeのパスを検索
        libreoffice_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
            "/usr/local/bin/soffice",
            "/usr/bin/soffice",
            "libreoffice",
            "soffice"
        ]
        
        libreoffice_cmd = None
        for path in libreoffice_paths:
            try:
                if os.path.exists(path) or path in ["libreoffice", "soffice"]:
                    # コマンドの存在確認
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        libreoffice_cmd = path
                        break
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        if not libreoffice_cmd:
            raise FileNotFoundError("LibreOfficeが見つかりません")
        
        # LibreOfficeを使用してPDFをDOCXに変換
        output_dir = os.path.dirname(docx_path)
        cmd = [
            libreoffice_cmd,
            "--headless", "--convert-to", "docx",
            "--outdir", output_dir, 
            pdf_path
        ]
        
        result = subprocess.run(cmd, check=True, timeout=120, 
                              capture_output=True, text=True)
        
        # LibreOfficeは元のファイル名を使用するため、必要に応じてリネーム
        generated_docx = os.path.join(
            output_dir,
            os.path.splitext(os.path.basename(pdf_path))[0] + ".docx"
        )
        
        if generated_docx != docx_path and os.path.exists(generated_docx):
            shutil.move(generated_docx, docx_path)
        
        if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
            logger.info(f"LibreOfficeによるPDF→DOCX変換が完了しました: {docx_path}")
            return docx_path
        else:
            logger.warning("LibreOfficeでDOCXが作成されませんでした")
            raise Exception("DOCXが作成されませんでした")
            
    except Exception as e:
        logger.warning(f"LibreOfficeによるPDF→DOCX変換に失敗しました: {e}")
        
        # 方法3: PDFから直接テキストを抽出してDOCXを作成
        try:
            logger.info("PDFから直接テキストを抽出してDOCXを作成しています...")
            
            # PDFからテキストを抽出
            text_content = []
            
            # pdfplumberを試す
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text and text.strip():
                            text_content.append(f"=== Page {page_num + 1} ===\n{text}\n")
            except Exception as e:
                logger.warning(f"pdfplumberでの抽出に失敗: {e}")
                
                # PyPDF2を試す
                try:
                    import PyPDF2
                    with open(pdf_path, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        for page_num, page in enumerate(pdf_reader.pages):
                            text = page.extract_text()
                            if text and text.strip():
                                text_content.append(f"=== Page {page_num + 1} ===\n{text}\n")
                except Exception as e2:
                    logger.error(f"PyPDF2での抽出も失敗: {e2}")
                    raise ValueError(f"PDFからのテキスト抽出に失敗しました: {str(e2)}")
            
            if not text_content:
                raise ValueError("PDFからテキストを抽出できませんでした")
            
            # DOCXファイルを作成
            from docx import Document
            doc = Document()
            
            for content in text_content:
                paragraphs = content.split('\n')
                for paragraph_text in paragraphs:
                    if paragraph_text.strip():
                        doc.add_paragraph(paragraph_text)
            
            doc.save(docx_path)
            
            if os.path.exists(docx_path) and os.path.getsize(docx_path) > 0:
                logger.info(f"テキスト抽出によるDOCX作成が完了しました: {docx_path}")
                return docx_path
            else:
                raise ValueError("DOCXファイルの作成に失敗しました")
                
        except Exception as e2:
            logger.error(f"テキスト抽出によるDOCX作成も失敗しました: {e2}")
            raise ValueError(f"PDFからDOCXへの変換に失敗しました: {str(e2)}")

def convert_docx_to_pdf(docx_path: str, pdf_path: str) -> str:
    """
    DOCXファイルをPDFに変換します。
    複数の方法を試行してフォールバック処理を行います。
    
    Args:
        docx_path: 入力DOCXファイルのパス
        pdf_path: 出力PDFファイルのパス
        
    Returns:
        変換されたPDFファイルのパス
    """
    import sys
    import threading
    import time
    
    logger.info(f"DOCXをPDFに変換しています: {docx_path} -> {pdf_path}")
    
    # 方法1: docx2pdfを試す
    try:
        from docx2pdf import convert
        
        logger.info("docx2pdfを使用してDOCXをPDFに変換しています...")
        
        def convert_with_timeout():
            try:
                convert(docx_path, pdf_path)
                return True
            except Exception as e:
                logger.error(f"docx2pdf変換エラー: {e}")
                return False
        
        # 別スレッドで変換を実行（タイムアウト付き）
        thread = threading.Thread(target=convert_with_timeout)
        thread.daemon = True
        thread.start()
        
        # 最大60秒待機
        timeout = 60
        start_time = time.time()
        while thread.is_alive() and time.time() - start_time < timeout:
            time.sleep(1)
        
        if thread.is_alive():
            logger.warning(f"docx2pdfがタイムアウトしました（{timeout}秒）")
            raise TimeoutError(f"docx2pdfがタイムアウトしました（{timeout}秒）")
        
        # 変換結果を確認
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            logger.info(f"docx2pdfによるDOCX変換が完了しました: {pdf_path}")
            return pdf_path
        else:
            logger.warning("docx2pdfでPDFが作成されませんでした")
            raise Exception("PDFが作成されませんでした")
            
    except ImportError:
        logger.warning("docx2pdfがインストールされていません。代替方法を試みます。")
    except Exception as e:
        logger.warning(f"docx2pdfによる変換に失敗しました: {e}. 代替方法を試みます。")
    
    # 方法2: LibreOfficeを試す
    try:
        logger.info("LibreOfficeを使用してDOCXをPDFに変換しています...")
        
        # LibreOfficeのパスを検索
        libreoffice_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",  # macOS
            "/usr/local/bin/soffice",
            "/usr/bin/soffice",
            "libreoffice",
            "soffice"
        ]
        
        libreoffice_cmd = None
        for path in libreoffice_paths:
            try:
                if os.path.exists(path) or path in ["libreoffice", "soffice"]:
                    # コマンドの存在確認
                    result = subprocess.run([path, "--version"], 
                                          capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        libreoffice_cmd = path
                        break
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
                continue
        
        if not libreoffice_cmd:
            raise FileNotFoundError("LibreOfficeが見つかりません")
        
        # LibreOfficeを使用してDOCXをPDFに変換
        cmd = [
            libreoffice_cmd,
            "--headless", "--convert-to", "pdf",
            "--outdir", os.path.dirname(pdf_path), 
            docx_path
        ]
        
        result = subprocess.run(cmd, check=True, timeout=120, 
                              capture_output=True, text=True)
        
        # LibreOfficeは元のファイル名を使用するため、必要に応じてリネーム
        generated_pdf = os.path.join(
            os.path.dirname(pdf_path),
            os.path.splitext(os.path.basename(docx_path))[0] + ".pdf"
        )
        
        if generated_pdf != pdf_path and os.path.exists(generated_pdf):
            shutil.move(generated_pdf, pdf_path)
        
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            logger.info(f"LibreOfficeによるDOCX変換が完了しました: {pdf_path}")
            return pdf_path
        else:
            logger.warning("LibreOfficeでPDFが作成されませんでした")
            raise Exception("PDFが作成されませんでした")
            
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning(f"LibreOfficeによる変換に失敗しました: {e}")
    
    # 方法3: reportlabを使用した簡易PDF生成
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from docx import Document
        
        logger.info("reportlabを使用して簡易PDFを生成しています...")
        
        # DOCXからテキストを抽出
        doc = Document(docx_path)
        text_content = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                text_content.append(para.text)
        
        # 表からもテキストを抽出
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_content.append(" | ".join(row_text))
        
        # PDFを生成
        pdf_doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        flowables = []
        
        for text in text_content:
            try:
                # 問題のある文字を除去
                safe_text = ''.join(c for c in text if ord(c) < 65536)
                p = Paragraph(safe_text, styles["Normal"])
                flowables.append(p)
                flowables.append(Spacer(1, 12))
            except Exception as e:
                logger.warning(f"段落の作成に失敗しました: {e}")
                continue
        
        pdf_doc.build(flowables)
        
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            logger.info(f"reportlabによる簡易PDF生成が完了しました: {pdf_path}")
            logger.warning("注意: 簡易PDFはフォーマットが制限されています")
            return pdf_path
        else:
            logger.warning("reportlabでPDFが作成されませんでした")
            raise Exception("PDFが作成されませんでした")
            
    except ImportError:
        logger.warning("reportlabがインストールされていません")
    except Exception as e:
        logger.warning(f"reportlabによる変換に失敗しました: {e}")
    
    # すべての方法が失敗した場合
    logger.error("すべてのPDF変換方法が失敗しました")
    
    # 最終手段: DOCXファイルをそのまま返す
    docx_copy_path = pdf_path.replace(".pdf", ".docx")
    shutil.copy(docx_path, docx_copy_path)
    logger.warning(f"PDF変換に失敗したため、DOCXファイルをコピーしました: {docx_copy_path}")
    
    # エラーメッセージを含むテキストファイルを作成
    error_txt_path = pdf_path.replace(".pdf", "_error.txt")
    with open(error_txt_path, "w", encoding="utf-8") as f:
        f.write(f"PDF変換に失敗しました。DOCXファイルが代わりに保存されています: {docx_copy_path}\n")
        f.write("すべての変換方法（docx2pdf, LibreOffice, reportlab）が失敗しました。\n")
    
    return docx_copy_path

def translate_pdf(
    input_path: str,
    output_path: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    save_text_files_flag: bool = True,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,
) -> tuple:
    """
    PDFファイルを翻訳します。
    translator.py.old.pyの処理フローに従います：
    1. convert_pdf_to_docx()でPDFをDOCXに変換
    2. translate_docx()でDOCX翻訳を実行
    3. convert_docx_to_pdf()で翻訳済みDOCXをPDFに変換
    
    Args:
        input_path: 入力PDFファイルのパス
        output_path: 出力PDFファイルのパス
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        progress_callback: 進捗を報告するコールバック関数
        save_text_files_flag: テキストファイルを保存するかどうか
        ai_instruction: AIへの補足指示
        check_cancelled: 中止状態をチェックする関数
        
    Returns:
        (抽出テキストファイルパス, 翻訳テキストファイルパス)
    """
    logger.info(f"PDF翻訳を開始します: {input_path} -> {output_path}")
    
    if ai_instruction:
        logger.info(f"AIへの補足指示: {ai_instruction}")
    
    # 一時ファイルのパスを生成
    temp_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_docx_input = os.path.join(temp_dir, f"{base_name}_temp_input.docx")
    temp_docx_output = os.path.join(temp_dir, f"{base_name}_temp_output.docx")
    
    try:
        # 中止チェック
        if check_cancelled and check_cancelled():
            logger.info("翻訳が中止されました")
            return None, None
        
        # 進捗状況の更新
        if progress_callback:
            progress_callback(0.05)
            
        # 1. PDFをDOCXに変換
        logger.info("PDFをDOCXに変換しています...")
        convert_pdf_to_docx(input_path, temp_docx_input)
        
        # 中止チェック
        if check_cancelled and check_cancelled():
            logger.info("翻訳が中止されました")
            return None, None
        
        if progress_callback:
            progress_callback(0.15)
        
        # 2. DOCXを翻訳
        logger.info("DOCXを翻訳しています...")
        extracted_file, translated_file = translate_docx(
            temp_docx_input,
            temp_docx_output,
            api_key,
            model,
            source_lang,
            target_lang,
            # 進捗コールバックを調整して15%〜85%の範囲にする
            lambda p: progress_callback(0.15 + p * 0.7) if progress_callback else None,
            save_text_files_flag,
            ai_instruction,
            check_cancelled
        )
        
        # 中止チェック
        if check_cancelled and check_cancelled():
            logger.info("翻訳が中止されました")
            return extracted_file, translated_file
        
        # 3. 翻訳済みDOCXをPDFに変換
        logger.info("翻訳済みDOCXをPDFに変換しています...")
        final_output = convert_docx_to_pdf(temp_docx_output, output_path)
        
        if progress_callback:
            progress_callback(1.0)
            
        # 最終出力がDOCXファイルの場合（PDF変換に失敗した場合）
        if final_output.endswith('.docx'):
            logger.warning("PDF変換に失敗しました。DOCXファイルが提供されます。")
            # DOCXファイルを適切な場所に移動
            if final_output != output_path.replace('.pdf', '.docx'):
                shutil.move(final_output, output_path.replace('.pdf', '.docx'))
        
        logger.info(f"PDF翻訳が完了しました: {output_path}")
        return extracted_file, translated_file
        
    except Exception as e:
        logger.error(f"PDF翻訳エラー: {e}")
        raise
    finally:
        # 一時ファイルの削除
        try:
            if os.path.exists(temp_docx_input):
                os.remove(temp_docx_input)
                logger.debug(f"一時ファイルを削除しました: {temp_docx_input}")
            if os.path.exists(temp_docx_output):
                os.remove(temp_docx_output)
                logger.debug(f"一時ファイルを削除しました: {temp_docx_output}")
        except Exception as e:
            logger.warning(f"一時ファイルの削除に失敗しました: {e}")

def translate_xlsx(
    input_path: str,
    output_path: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    save_text_files_flag: bool = True,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,
) -> tuple:
    """
    XLSXファイルを翻訳します。

    Args:
        input_path: 入力Excelファイルのパス
        output_path: 出力Excelファイルのパス
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        progress_callback: 進捗を報告するコールバック関数
        save_text_files_flag: テキストファイルを保存するかどうか
        ai_instruction: AIへの補足指示
        check_cancelled: 中止状態をチェックする関数

    Returns:
        (抽出テキストファイルパス, 翻訳テキストファイルパス)
    """
    logger.info(f"Excel翻訳を開始: {input_path} -> {output_path}")

    try:
        # 進捗表示
        if progress_callback:
            progress_callback(0.1)

        # Excelファイルを読み込み
        excel = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")
        total_sheets = len(excel)
        
        # テキストデータを抽出してDataFrameに変換
        text_data = []
        for sheet_idx, (sheet_name, df) in enumerate(excel.items()):
            if check_cancelled and check_cancelled():
                return None, None
                
            for row_idx, row in df.iterrows():
                for col_idx, value in row.items():
                    if isinstance(value, str) and value.strip():
                        text_data.append({
                            'sheet_name': sheet_name,
                            'row': row_idx,
                            'column': col_idx,
                            'original_text': value.strip()
                        })
            
            if progress_callback:
                progress_callback(0.1 + (sheet_idx / total_sheets * 0.2))

        # テキストデータをDataFrameに変換
        text_df = pd.DataFrame(text_data)

        # テキストファイルを保存
        output_dir = os.path.dirname(output_path)
        base_filename = os.path.splitext(os.path.basename(output_path))[0]
        extracted_file = None
        translated_file = None

        if save_text_files_flag:
            extracted_file, _ = save_text_files(text_df, output_dir, base_filename)

        # テキストを翻訳
        text_df = translate_dataframe(
            text_df,
            api_key,
            model,
            source_lang,
            target_lang,
            lambda p: progress_callback(0.3 + p * 0.5) if progress_callback else None,
            ai_instruction,
            check_cancelled
        )

        if save_text_files_flag and 'translated_text' in text_df.columns:
            _, translated_file = save_text_files(text_df, output_dir, base_filename)

        # 翻訳結果をExcelに反映
        translated_excel = excel.copy()
        for _, row in text_df.iterrows():
            if check_cancelled and check_cancelled():
                return extracted_file, translated_file
                
            if 'translated_text' in row and pd.notna(row['translated_text']):
                sheet = translated_excel[row['sheet_name']]
                sheet.at[row['row'], row['column']] = row['translated_text']

        # 翻訳したExcelファイルを保存
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            for sheet_name, df in translated_excel.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        if progress_callback:
            progress_callback(1.0)

        return extracted_file, translated_file

    except Exception as e:
        logger.error(f"Excel翻訳エラー: {e}")
        raise

def translate_pptx(
    input_path: str,
    output_path: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    save_text_files_flag: bool = True,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> tuple:
    """
    PPTX ファイルを翻訳します。

    Args:
        input_path: 入力 PPTX ファイルのパス
        output_path: 出力 PPTX ファイルのパス
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        progress_callback: 進捗を報告するコールバック関数
        save_text_files_flag: テキストファイルを保存するかどうか
        ai_instruction: AIへの補足指示
        check_cancelled: 中止状態をチェックする関数

    Returns:
        (抽出テキストファイルパス, 翻訳テキストファイルパス)
    """
    logger.info(f"PPTX 翻訳を開始します: {input_path} -> {output_path}")
    logger.info(f"言語: {source_lang} -> {target_lang}, モデル: {model}")

    if ai_instruction:
        logger.info(f"AIへの補足指示: {ai_instruction}")

    # 環境変数からAPI キーを取得
    if not api_key:
        api_key = os.environ.get("GENAI_HUB_API_KEY")

    if not api_key:
        raise ValueError(
            "API キーが設定されていません。環境変数 GENAI_HUB_API_KEY を確認してください。"
        )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return None, None

    # 1. テキスト抽出
    df = extract_text_from_pptx(input_path, progress_callback)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return None, None

    # 出力ディレクトリとベースファイル名を取得
    output_dir = os.path.dirname(output_path)
    base_filename = os.path.splitext(os.path.basename(output_path))[0]

    # 抽出したテキストを保存
    extracted_file = None
    translated_file = None

    if save_text_files_flag:
        extracted_file, _ = save_text_files(df, output_dir, base_filename)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, None

    # 2. テキスト翻訳
    df = translate_dataframe(
        df,
        api_key,
        model,
        source_lang,
        target_lang,
        progress_callback,
        ai_instruction,
        check_cancelled,
    )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, None

    # 翻訳したテキストを保存
    if save_text_files_flag and "translated_text" in df.columns:
        _, translated_file = save_text_files(df, output_dir, base_filename)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, translated_file

    # 3. 翻訳テキストの再インポート
    reimport_text_to_pptx(
        input_path, df, output_path, progress_callback, check_cancelled
    )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, translated_file

    logger.info(f"PPTX 翻訳が完了しました: {output_path}")

    return extracted_file, translated_file

def translate_docx(
    input_path: str,
    output_path: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    save_text_files_flag: bool = True,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> tuple:
    """
    Word(docx)ファイルを翻訳します。

    Args:
        input_path: 入力Wordファイルのパス
        output_path: 出力Wordファイルのパス
        api_key: GenAI Hub API キー
        model: 使用するモデル名
        source_lang: 元の言語コード
        target_lang: 翻訳先の言語コード
        progress_callback: 進捗を報告するコールバック関数
        save_text_files_flag: テキストファイルを保存するかどうか
        ai_instruction: AIへの補足指示
        check_cancelled: 中止状態をチェックする関数

    Returns:
        (抽出テキストファイルパス, 翻訳テキストファイルパス)
    """
    logger.info(f"Word文書翻訳を開始します: {input_path} -> {output_path}")
    logger.info(f"言語: {source_lang} -> {target_lang}, モデル: {model}")

    if ai_instruction:
        logger.info(f"AIへの補足指示: {ai_instruction}")

    # 環境変数からAPI キーを取得
    if not api_key:
        api_key = os.environ.get("GENAI_HUB_API_KEY")

    if not api_key:
        raise ValueError(
            "API キーが設定されていません。環境変数 GENAI_HUB_API_KEY を確認してください。"
        )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return None, None

    # 1. テキスト抽出
    df = extract_text_from_docx(input_path, progress_callback)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return None, None

    # 出力ディレクトリとベースファイル名を取得
    output_dir = os.path.dirname(output_path)
    base_filename = os.path.splitext(os.path.basename(output_path))[0]

    # 抽出したテキストを保存
    extracted_file = None
    translated_file = None

    if save_text_files_flag:
        extracted_file, _ = save_text_files(df, output_dir, base_filename)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, None

    # 2. テキスト翻訳
    df = translate_dataframe(
        df,
        api_key,
        model,
        source_lang,
        target_lang,
        progress_callback,
        ai_instruction,
        check_cancelled,
    )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, None

    # 翻訳したテキストを保存
    if save_text_files_flag and "translated_text" in df.columns:
        _, translated_file = save_text_files(df, output_dir, base_filename)

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, translated_file

    # 3. 翻訳テキストの再インポート
    reimport_text_to_docx(
        input_path, df, output_path, progress_callback, check_cancelled
    )

    # 中止チェック
    if check_cancelled and check_cancelled():
        logger.info("翻訳が中止されました")
        return extracted_file, translated_file

    logger.info(f"Word文書翻訳が完了しました: {output_path}")

    return extracted_file, translated_file

def check_libreoffice() -> Tuple[bool, str]:
    """
    LibreOfficeの可用性をチェックします。

    Returns:
        (利用可能かどうか, バージョン情報またはエラーメッセージ)
    """
    try:
        result = subprocess.run(
            ["libreoffice", "--version"], capture_output=True, text=True
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            logger.info(f"LibreOffice version: {version}")
            return True, version
        else:
            error_msg = f"LibreOfficeコマンドはエラーを返しました: {result.stderr}"
            logger.error(error_msg)
            return False, error_msg

    except FileNotFoundError:
        error_msg = "LibreOfficeコマンドが見つかりません。インストールされていない可能性があります。"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"LibreOfficeチェックエラー: {e}"
        logger.error(error_msg)
        return False, error_msg

def detect_file_type(file_path: str) -> str:
    """
    ファイルの拡張子からファイル形式を検出します。

    Args:
        file_path: ファイルパス

    Returns:
        ファイル形式 ('pptx', 'docx', 'pdf')
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pptx":
        return "pptx"
    elif ext == ".docx":
        return "docx"
    elif ext == ".pdf":
        return "pdf"
    elif ext == ".xlsx":
        return "xlsx"
    else:
        raise ValueError(f"サポートされていないファイル形式です: {ext}")

def translate_document(
    input_path: str,
    output_path: str,
    api_key: str = None,
    model: str = "claude-3-5-haiku",
    source_lang: str = "en",
    target_lang: str = "ja",
    progress_callback: Optional[Callable] = None,
    save_text_files_flag: bool = True,
    ai_instruction: str = "",
    check_cancelled: Optional[Callable] = None,
) -> tuple:
    """
    ドキュメント（PPTX, DOCX, PDF, XLSX）を翻訳します。
    """
    logger.info(f"ドキュメント翻訳を開始: {input_path} -> {output_path}")

    # 起動時にキャッシュをロード
    if TRANSLATION_CONFIG["use_cache"] and not TRANSLATION_CACHE:
        load_translation_cache()

    file_type = detect_file_type(input_path)

    try:
        # 各処理ステップで中止状態をチェック
        if check_cancelled and check_cancelled():
            logger.info("Translation cancelled by user")
            return None, None

        # ファイル形式に応じた処理
        if file_type == "xlsx":
            result = translate_xlsx(
                input_path,
                output_path,
                api_key,
                model,
                source_lang,
                target_lang,
                progress_callback,
                save_text_files_flag,
                ai_instruction,
                check_cancelled
            )
        elif file_type == "pptx":
            result = translate_pptx(
                input_path,
                output_path,
                api_key,
                model,
                source_lang,
                target_lang,
                progress_callback,
                save_text_files_flag,
                ai_instruction,
                check_cancelled
            )
        elif file_type == "docx":
            result = translate_docx(
                input_path,
                output_path,
                api_key,
                model,
                source_lang,
                target_lang,
                progress_callback,
                save_text_files_flag,
                ai_instruction,
                check_cancelled
            )
        elif file_type == "pdf":
            result = translate_pdf(
                input_path,
                output_path,
                api_key,
                model,
                source_lang,
                target_lang,
                progress_callback,
                save_text_files_flag,
                ai_instruction,
                check_cancelled
            )
        else:
            raise ValueError(f"サポートされていないファイル形式です: {file_type}")

        # 翻訳完了後にキャッシュを保存
        if TRANSLATION_CONFIG["use_cache"]:
            save_translation_cache()

        return result

    except Exception as e:
        logger.error(f"翻訳エラー: {e}")
        if TRANSLATION_CONFIG["use_cache"]:
            save_translation_cache()
        raise

# テスト用コード
if __name__ == "__main__":
    # API キーを環境変数から取得
    api_key = os.environ.get("GENAI_HUB_API_KEY")

    if not api_key:
        print("環境変数 GENAI_HUB_API_KEY が設定されていません。")
        api_key = input("GenAI Hub API キーを入力してください: ")

    # 簡単なテスト翻訳
    test_text = "Hello, world. This is a translation test."
    print(f"テスト翻訳: {test_text}")

    translated = translate_text_with_cache(
        test_text, api_key, model="claude-3-5-haiku", source_lang="en", target_lang="ja"
    )

    print(f"翻訳結果: {translated}")

# エクスポートする関数と変数
__all__ = [
    "get_available_models",
    "fetch_available_models",
    "LANGUAGES",
    "TRANSLATION_CONFIG",
    "translate_pptx",
    "translate_docx",
    "translate_pdf",
    "translate_xlsx",
    "translate_document",
    "load_translation_cache",
    "save_translation_cache",
]
