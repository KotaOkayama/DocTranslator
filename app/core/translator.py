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

import subprocess
import tempfile
import shutil
from typing import Optional, Tuple, Callable

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

# Available models
AVAILABLE_MODELS = {
    "claude-4-sonnet": "Claude 4 Sonnet",  # 追加
    "claude-3-7-sonnet": "Claude 3.7 Sonnet",
    "claude-3-5-sonnet-v2": "Claude 3.5 Sonnet V2",
    "claude-3-5-haiku": "Claude 3.5 Haiku",
}

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
# PDF 処理関数
#


import PyPDF2
import pdfplumber
from docx import Document
from docx.shared import Inches
import pandas as pd

def extract_text_from_pdf_direct(
    pdf_path: str, progress_callback: Optional[Callable] = None
) -> pd.DataFrame:
    """
    PDFから直接テキストを抽出し、DataFrameとして返します。
    """
    logger.info(f"PDFから直接テキストを抽出: {pdf_path}")
    
    text_data = []
    
    try:
        # pdfplumberを使用してテキストを抽出
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"ページ数: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages):
                if progress_callback:
                    try:
                        progress_callback(page_num / total_pages * 0.3)
                    except Exception as e:
                        logger.warning(f"進捗コールバックエラー: {e}")
                
                # ページからテキストを抽出
                text = page.extract_text()
                if text and text.strip():
                    # テキストを段落に分割
                    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                    
                    for para_num, paragraph in enumerate(paragraphs):
                        if paragraph:
                            logger.debug(f"ページ {page_num+1}, 段落 {para_num+1}: {paragraph[:50]}...")
                            
                            text_data.append({
                                "page_num": page_num,
                                "para_num": para_num,
                                "original_text": paragraph,
                                "element_type": "pdf_paragraph",
                            })
    
    except Exception as e:
        logger.error(f"pdfplumberでの抽出に失敗: {e}")
        
        # PyPDF2を代替として使用
        try:
            logger.info("PyPDF2を使用して再試行...")
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    if progress_callback:
                        try:
                            progress_callback(page_num / total_pages * 0.3)
                        except Exception as e:
                            logger.warning(f"進捗コールバックエラー: {e}")
                    
                    text = page.extract_text()
                    if text and text.strip():
                        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                        
                        for para_num, paragraph in enumerate(paragraphs):
                            if paragraph:
                                text_data.append({
                                    "page_num": page_num,
                                    "para_num": para_num,
                                    "original_text": paragraph,
                                    "element_type": "pdf_paragraph",
                                })
        
        except Exception as e2:
            logger.error(f"PyPDF2での抽出も失敗: {e2}")
            raise ValueError(f"PDFからのテキスト抽出に失敗しました: {str(e2)}")
    
    # データフレームに変換
    df = pd.DataFrame(text_data)
    
    logger.info(f"抽出完了: {len(df)} 個のテキスト要素が見つかりました")
    
    if progress_callback:
        try:
            progress_callback(0.3)  # 抽出完了
        except Exception as e:
            logger.warning(f"進捗コールバックエラー: {e}")
    
    return df


def create_docx_from_pdf_text(
    df: pd.DataFrame,
    output_path: str,
    progress_callback: Optional[Callable] = None,
    check_cancelled: Optional[Callable] = None,
) -> None:
    """
    PDFから抽出したテキストを使用してDOCXファイルを作成します。
    """
    logger.info(f"PDFテキストからDOCXファイルを作成: {output_path}")
    
    try:
        # 新しいWord文書を作成
        doc = Document()
        
        # ページごとにテキストを整理
        pages = df.groupby('page_num')
        total_pages = len(pages)
        
        for page_idx, (page_num, page_data) in enumerate(pages):
            # 中止チェック
            if check_cancelled and check_cancelled():
                logger.info("DOCX作成が中止されました")
                return
            
            if progress_callback:
                try:
                    progress = 0.8 + (page_idx / total_pages * 0.2)
                    progress_callback(progress)
                except Exception as e:
                    logger.warning(f"進捗コールバックエラー: {e}")
            
            # ページヘッダーを追加
            if page_num > 0:
                doc.add_page_break()
            
            page_header = doc.add_heading(f'Page {page_num + 1}', level=2)
            
            # 段落を追加
            for _, row in page_data.iterrows():
                if 'translated_text' in row and pd.notna(row['translated_text']) and row['translated_text']:
                    text_to_add = row['translated_text']
                else:
                    text_to_add = row['original_text']
                
                paragraph = doc.add_paragraph(text_to_add)
        
        # ファイルを保存
        doc.save(output_path)
        logger.info(f"DOCXファイルが作成されました: {output_path}")
        
        if progress_callback:
            try:
                progress_callback(1.0)  # 完了
            except Exception as e:
                logger.warning(f"進捗コールバックエラー: {e}")
    
    except Exception as e:
        logger.error(f"DOCX作成エラー: {e}")
        raise

def translate_pdf_direct(
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
    PDFファイルを直接処理して翻訳します
    """
    logger.info(f"PDF直接翻訳を開始します: {input_path} -> {output_path}")
    
    # 一時的なDOCXファイルのパスを生成
    output_dir = os.path.dirname(output_path)
    base_filename = os.path.splitext(os.path.basename(output_path))[0]
    temp_docx = os.path.join(output_dir, f"{base_filename}_temp.docx")
    
    try:
        # 1. PDFから直接テキスト抽出
        df = extract_text_from_pdf_direct(input_path, progress_callback)

        # 中止チェック
        if check_cancelled and check_cancelled():
            logger.info("翻訳が中止されました")
            return None, None

        # 抽出したテキストを保存
        extracted_file = None
        translated_file = None

        if save_text_files_flag:
            extracted_file, _ = save_text_files(df, output_dir, base_filename)

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

        # 翻訳したテキストを保存
        if save_text_files_flag and "translated_text" in df.columns:
            _, translated_file = save_text_files(df, output_dir, base_filename)

        # 3. 翻訳テキストからDOCXファイルを作成
        logger.info(f"DOCXファイルを作成中: {temp_docx}")
        create_docx_from_pdf_text(
            df, temp_docx, progress_callback, check_cancelled
        )

        # DOCXファイルが正常に作成されたか確認
        if not os.path.exists(temp_docx):
            raise ValueError(f"DOCXファイルの作成に失敗しました: {temp_docx}")
        
        docx_size = os.path.getsize(temp_docx)
        logger.info(f"DOCXファイルが作成されました: {temp_docx} ({docx_size} bytes)")

        # 4. DOCXからPDFに変換
        try:
            logger.info(f"DOCXをPDFに変換開始: {temp_docx} -> {output_path}")
            convert_docx_to_pdf_with_libreoffice(temp_docx, output_path)
            
            # PDFファイルが正常に生成されたことを確認
            if os.path.exists(output_path):
                pdf_size = os.path.getsize(output_path)
                if pdf_size > 0:
                    logger.info(f"PDF変換が完了しました: {output_path} ({pdf_size} bytes)")
                    # 一時DOCXファイルを削除
                    if os.path.exists(temp_docx):
                        os.remove(temp_docx)
                        logger.debug(f"一時DOCXファイルを削除: {temp_docx}")
                    return extracted_file, translated_file
                else:
                    logger.warning("生成されたPDFファイルが空です")
                    raise ValueError("生成されたPDFファイルが空です")
            else:
                logger.warning("PDFファイルが生成されませんでした")
                raise ValueError("PDFファイルが生成されませんでした")
                
        except Exception as e:
            logger.error(f"PDF変換エラー: {e}")
            logger.warning("PDF変換に失敗しました。DOCXファイルを提供します。")
            
            # DOCXファイルを最終出力として使用
            docx_output_path = output_path.replace('.pdf', '.docx')
            if temp_docx != docx_output_path:
                shutil.move(temp_docx, docx_output_path)
                logger.info(f"DOCXファイルを移動: {temp_docx} -> {docx_output_path}")
            
            return extracted_file, translated_file, docx_output_path

    except Exception as e:
        logger.error(f"PDF処理エラー: {e}")
        # 一時ファイルのクリーンアップ
        if os.path.exists(temp_docx):
            os.remove(temp_docx)
        raise



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
    """PDFファイルを翻訳します（直接処理を優先、LibreOfficeは代替手段）"""
    logger.info(f"PDF翻訳を開始します: {input_path} -> {output_path}")

    try:
        # まず直接処理を試行
        logger.info("PDFから直接テキストを抽出して翻訳を試行します...")
        return translate_pdf_direct(
            input_path,
            output_path,
            api_key,
            model,
            source_lang,
            target_lang,
            progress_callback,
            save_text_files_flag,
            ai_instruction,
            check_cancelled,
        )
    
    except Exception as direct_error:
        logger.warning(f"直接処理に失敗しました: {direct_error}")
        logger.info("LibreOfficeを使用した変換を試行します...")
        
        try:
            # 一時ファイル用のディレクトリ
            with tempfile.TemporaryDirectory() as temp_dir:
                # 中止チェック
                if check_cancelled and check_cancelled():
                    logger.info("翻訳が中止されました")
                    return None, None

                # 進捗更新
                if progress_callback:
                    progress_callback(0.1)

                # 1. PDF → DOCX変換（LibreOfficeを使用）
                temp_docx = os.path.join(temp_dir, "temp.docx")
                logger.info(f"PDFをDOCXに変換: {input_path} -> {temp_docx}")
                
                # LibreOfficeを使用してPDFをDOCXに変換
                convert_pdf_to_docx_with_libreoffice(input_path, temp_docx)

                # 中止チェック
                if check_cancelled and check_cancelled():
                    logger.info("翻訳が中止されました")
                    return None, None

                # 進捗更新
                if progress_callback:
                    progress_callback(0.2)

                # 2. DOCXを翻訳
                temp_translated_docx = os.path.join(temp_dir, "translated.docx")
                extracted_file_temp, translated_file_temp = translate_docx(
                    temp_docx,
                    temp_translated_docx,
                    api_key,
                    model,
                    source_lang,
                    target_lang,
                    lambda p: (
                        progress_callback(0.2 + p * 0.6) if progress_callback else None
                    ),
                    save_text_files_flag,
                    ai_instruction,
                    check_cancelled,
                )

                # 中止チェック
                if check_cancelled and check_cancelled():
                    logger.info("翻訳が中止されました")
                    return None, None

                # 一時ファイルを出力ディレクトリにコピー
                output_dir = os.path.dirname(output_path)
                base_filename = os.path.splitext(os.path.basename(output_path))[0]

                extracted_file = None
                translated_file = None

                try:
                    if extracted_file_temp and os.path.exists(extracted_file_temp):
                        extracted_file = os.path.join(
                            output_dir, f"{base_filename}_extracted.txt"
                        )
                        shutil.copy2(extracted_file_temp, extracted_file)
                        logger.info(
                            f"抽出テキストファイルをコピーしました: {extracted_file}"
                        )

                    if translated_file_temp and os.path.exists(translated_file_temp):
                        translated_file = os.path.join(
                            output_dir, f"{base_filename}_translated.txt"
                        )
                        shutil.copy2(translated_file_temp, translated_file)
                        logger.info(
                            f"翻訳テキストファイルをコピーしました: {translated_file}"
                        )
                except Exception as e:
                    logger.error(f"ファイルコピー中にエラーが発生しました: {e}")
                    # コピーに失敗しても処理は続行

                # 中止チェック
                if check_cancelled and check_cancelled():
                    logger.info("翻訳が中止されました")
                    return extracted_file, translated_file

                # 3. DOCX → PDF変換（LibreOfficeを使用）
                logger.info(
                    f"翻訳済みDOCXをPDFに変換: {temp_translated_docx} -> {output_path}"
                )
                convert_docx_to_pdf_with_libreoffice(temp_translated_docx, output_path)

                # 進捗更新
                if progress_callback:
                    progress_callback(1.0)

                logger.info(f"PDF翻訳が完了しました: {output_path}")
                return extracted_file, translated_file

        except Exception as libreoffice_error:
            logger.error(f"LibreOffice変換も失敗しました: {libreoffice_error}")
            
            # 最後の手段として、DOCXファイルのみ提供
            try:
                logger.info("DOCXファイルのみ提供します...")
                docx_output_path = output_path.replace('.pdf', '.docx')
                
                return translate_pdf_direct(
                    input_path,
                    docx_output_path,
                    api_key,
                    model,
                    source_lang,
                    target_lang,
                    progress_callback,
                    save_text_files_flag,
                    ai_instruction,
                    check_cancelled,
                )
            except Exception as final_error:
                logger.error(f"すべての変換方法が失敗しました: {final_error}")
                raise ValueError(f"PDF翻訳に失敗しました: {str(final_error)}")


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


def convert_pdf_to_docx_with_libreoffice(input_pdf: str, output_docx: str) -> None:
    """
    LibreOfficeを使用してPDFファイルをDOCXに変換します。
    """
    try:
        # 出力ディレクトリを取得
        output_dir = os.path.dirname(output_docx)
        
        # 出力ディレクトリの存在確認と権限設定
        os.makedirs(output_dir, exist_ok=True)
        os.chmod(output_dir, 0o777)

        # 入力ファイルの絶対パスを取得
        input_pdf_abs = os.path.abspath(input_pdf)
        
        # 一時作業ディレクトリを作成
        work_dir = os.path.join(output_dir, "work")
        os.makedirs(work_dir, exist_ok=True)
        os.chmod(work_dir, 0o777)

        # LibreOfficeコマンドを構築（より詳細なオプション）
        cmd = [
            "libreoffice",
            "--headless",
            "--invisible",
            "--nodefault",
            "--nolockcheck",
            "--nologo",
            "--norestore",
            "--convert-to", "docx",
            "--outdir", output_dir,
            input_pdf_abs
        ]

        logger.debug(f"LibreOfficeコマンド: {' '.join(cmd)}")

        # 環境変数を設定
        env = os.environ.copy()
        env.update({
            "HOME": work_dir,
            "TMPDIR": work_dir,
            "LOGNAME": "libreoffice",
            "USER": "libreoffice",
            "USERNAME": "libreoffice",
            "DISPLAY": ":99",  # 仮想ディスプレイ
        })

        # コマンドを実行し、結果を取得
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=120,  # タイムアウトを延長
            env=env,
            cwd=work_dir
        )

        # 実行結果のログ出力
        logger.debug(f"LibreOffice return code: {result.returncode}")
        if result.stdout:
            logger.debug(f"LibreOffice stdout: {result.stdout}")
        if result.stderr:
            logger.debug(f"LibreOffice stderr: {result.stderr}")

        # 生成されたDOCXファイル名
        generated_docx = os.path.join(
            output_dir, os.path.splitext(os.path.basename(input_pdf))[0] + ".docx"
        )

        # ファイルの存在確認
        if os.path.exists(generated_docx):
            os.chmod(generated_docx, 0o666)
            logger.info(f"DOCX file generated successfully: {generated_docx}")
        else:
            # 詳細なデバッグ情報
            logger.error(f"Output directory contents: {os.listdir(output_dir)}")
            if os.path.exists(work_dir):
                logger.error(f"Work directory contents: {os.listdir(work_dir)}")
            
            # 代替方法を試行（Writer形式で変換）
            logger.info("Trying alternative conversion method...")
            alt_cmd = [
                "libreoffice",
                "--headless",
                "--convert-to", "odt",
                "--outdir", output_dir,
                input_pdf_abs
            ]
            
            alt_result = subprocess.run(
                alt_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                env=env,
                cwd=work_dir
            )
            
            odt_file = os.path.join(
                output_dir, os.path.splitext(os.path.basename(input_pdf))[0] + ".odt"
            )
            
            if os.path.exists(odt_file):
                # ODTからDOCXに変換
                docx_cmd = [
                    "libreoffice",
                    "--headless",
                    "--convert-to", "docx",
                    "--outdir", output_dir,
                    odt_file
                ]
                
                subprocess.run(docx_cmd, env=env, cwd=work_dir, timeout=60)
                
                # ODTファイルを削除
                os.remove(odt_file)
            
            if not os.path.exists(generated_docx):
                raise ValueError(f"DOCX file was not generated: {generated_docx}")

        # 指定された出力パスに移動（必要な場合）
        if generated_docx != output_docx and os.path.exists(generated_docx):
            shutil.move(generated_docx, output_docx)
            os.chmod(output_docx, 0o666)

        logger.info(f"LibreOfficeによるPDF→DOCX変換が完了しました: {output_docx}")

    except subprocess.TimeoutExpired:
        logger.error("LibreOffice conversion timed out")
        raise ValueError("PDF変換がタイムアウトしました")
    except subprocess.CalledProcessError as e:
        logger.error(f"LibreOfficeによる変換エラー: {e}")
        logger.error(f"標準出力: {e.stdout}")
        logger.error(f"標準エラー: {e.stderr}")
        raise ValueError(f"PDFからDOCXへの変換に失敗しました: {str(e)}")
    except Exception as e:
        logger.error(f"PDF→DOCX変換エラー: {e}")
        raise ValueError(f"PDFからDOCXへの変換に失敗しました: {str(e)}")
    finally:
        # 一時作業ディレクトリを削除
        try:
            if 'work_dir' in locals() and os.path.exists(work_dir):
                shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"一時ディレクトリの削除に失敗しました: {e}")

def convert_pdf_to_text_fallback(input_pdf: str) -> str:
    """
    PDFからテキストを直接抽出する代替手段
    """
    try:
        import PyPDF2
        
        with open(input_pdf, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text
    except ImportError:
        logger.warning("PyPDF2 is not installed. Cannot extract text from PDF.")
        return ""
    except Exception as e:
        logger.error(f"PDF text extraction failed: {e}")
        return ""

def convert_docx_to_pdf_with_libreoffice(input_docx: str, output_pdf: str) -> None:
    """
    LibreOfficeを使用してDOCXファイルをPDFに変換します。
    """
    import tempfile
    import time
    import signal
    
    try:
        # 入力ファイルの存在確認
        if not os.path.exists(input_docx):
            raise ValueError(f"Input DOCX file not found: {input_docx}")
        
        input_size = os.path.getsize(input_docx)
        logger.info(f"PDF変換開始 - 入力ファイル: {input_docx} ({input_size} bytes)")
        
        # 出力ディレクトリの準備
        output_dir = os.path.dirname(output_pdf)
        os.makedirs(output_dir, exist_ok=True)
        
        # 一時作業ディレクトリを作成
        with tempfile.TemporaryDirectory(prefix="libreoffice_", dir="/tmp") as temp_dir:
            logger.info(f"一時ディレクトリ: {temp_dir}")
            
            # 入力ファイルを一時ディレクトリにコピー
            temp_input = os.path.join(temp_dir, "input.docx")
            shutil.copy2(input_docx, temp_input)
            logger.info(f"入力ファイルをコピー: {temp_input}")
            
            # LibreOffice用の環境変数
            env = os.environ.copy()
            env.update({
                "HOME": temp_dir,
                "TMPDIR": temp_dir,
                "USER": "root",
                "DISPLAY": ":99",
                "SAL_USE_VCLPLUGIN": "svp",
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            })
            
            # 複数の変換方法を試行
            conversion_attempts = [
                {
                    "name": "Standard PDF conversion",
                    "cmd": [
                        "libreoffice",
                        "--headless",
                        "--invisible",
                        "--nodefault",
                        "--nolockcheck",
                        "--nologo",
                        "--norestore",
                        "--convert-to", "pdf",
                        "--outdir", temp_dir,
                        temp_input
                    ]
                },
                {
                    "name": "Writer PDF Export",
                    "cmd": [
                        "libreoffice",
                        "--headless",
                        "--invisible",
                        "--nodefault",
                        "--nolockcheck",
                        "--nologo",
                        "--norestore",
                        "--convert-to", "pdf:writer_pdf_Export",
                        "--outdir", temp_dir,
                        temp_input
                    ]
                },
                {
                    "name": "Alternative soffice command",
                    "cmd": [
                        "soffice",
                        "--headless",
                        "--invisible",
                        "--nodefault",
                        "--nolockcheck",
                        "--nologo",
                        "--norestore",
                        "--convert-to", "pdf",
                        "--outdir", temp_dir,
                        temp_input
                    ]
                },
                {
                    "name": "Simple conversion",
                    "cmd": [
                        "libreoffice",
                        "--convert-to", "pdf",
                        "--outdir", temp_dir,
                        temp_input
                    ]
                }
            ]
            
            success = False
            last_error = None
            
            for attempt in conversion_attempts:
                try:
                    logger.info(f"試行中: {attempt['name']}")
                    logger.debug(f"コマンド: {' '.join(attempt['cmd'])}")
                    
                    # プロセス実行
                    process = subprocess.Popen(
                        attempt['cmd'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        cwd=temp_dir,
                        preexec_fn=os.setsid  # プロセスグループを作成
                    )
                    
                    try:
                        # タイムアウト付きで待機
                        stdout, stderr = process.communicate(timeout=180)
                        return_code = process.returncode
                        
                        logger.debug(f"Return code: {return_code}")
                        if stdout:
                            logger.debug(f"Stdout: {stdout}")
                        if stderr:
                            logger.debug(f"Stderr: {stderr}")
                        
                        # 出力ファイルをチェック
                        temp_pdf = os.path.join(temp_dir, "input.pdf")
                        
                        # 少し待機してファイルシステムの同期を待つ
                        time.sleep(1)
                        
                        if os.path.exists(temp_pdf):
                            pdf_size = os.path.getsize(temp_pdf)
                            logger.info(f"PDFファイル生成: {temp_pdf} ({pdf_size} bytes)")
                            
                            if pdf_size > 100:  # 最小サイズチェック（100バイト以上）
                                # 最終出力先にコピー
                                shutil.copy2(temp_pdf, output_pdf)
                                os.chmod(output_pdf, 0o666)
                                
                                final_size = os.path.getsize(output_pdf)
                                logger.info(f"PDF変換完了: {output_pdf} ({final_size} bytes)")
                                success = True
                                break
                            else:
                                logger.warning(f"生成されたPDFファイルが小さすぎます: {pdf_size} bytes")
                                if os.path.exists(temp_pdf):
                                    os.remove(temp_pdf)
                        else:
                            logger.warning(f"PDFファイルが生成されませんでした: {temp_pdf}")
                            logger.debug(f"一時ディレクトリの内容: {os.listdir(temp_dir)}")
                    
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{attempt['name']} がタイムアウトしました")
                        # プロセスグループ全体を終了
                        try:
                            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                            time.sleep(2)
                            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        except:
                            pass
                        process.wait()
                        continue
                        
                except Exception as e:
                    logger.warning(f"{attempt['name']} でエラー: {e}")
                    last_error = e
                    continue
            
            if not success:
                error_msg = f"すべての変換方法が失敗しました。最後のエラー: {last_error}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
    except Exception as e:
        logger.error(f"PDF変換エラー: {e}")
        raise ValueError(f"DOCXからPDFへの変換に失敗しました: {str(e)}")


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
    check_cancelled: Optional[Callable] = None,  # 中止チェック関数を追加
) -> tuple:
    """
    ドキュメント（PPTX, DOCX, PDF）を翻訳します。

    Args:
        input_path: 入力ファイルのパス
        output_path: 出力ファイルのパス
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
    # 起動時にキャッシュをロード
    if TRANSLATION_CONFIG["use_cache"] and not TRANSLATION_CACHE:
        load_translation_cache()

    file_type = detect_file_type(input_path)

    try:
        # 各処理ステップで中止状態をチェック
        if check_cancelled and check_cancelled():
            logger.info("Translation cancelled by user")
            return None, None

        if file_type == "pptx":
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
                check_cancelled,  # 中止チェック関数を渡す
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
                check_cancelled,  # 中止チェック関数を渡す
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
                check_cancelled,  # 中止チェック関数を渡す
            )
        else:
            raise ValueError(f"サポートされていないファイル形式です: {file_type}")

        # 翻訳完了後にキャッシュを保存
        if TRANSLATION_CONFIG["use_cache"]:
            save_translation_cache()

        return result
    except Exception as e:
        logger.error(f"翻訳エラー: {e}")
        # エラー発生時もキャッシュを保存
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
    "AVAILABLE_MODELS",
    "LANGUAGES",
    "TRANSLATION_CONFIG",
    "translate_pptx",
    "translate_docx",
    "translate_pdf",
    "translate_document",
    "load_translation_cache",
    "save_translation_cache",
]
