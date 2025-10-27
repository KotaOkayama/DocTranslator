# app/core/text_translator.py
"""
Text Translation Module for LangTranslator

This module handles text-only translation using GenAI Hub API.
Supports long text splitting and chunk-based translation.
"""

import re
import time
import logging
import requests
from typing import List, Optional

from app.utils.language_detector import (
    detect_language as detect_language_util,
    suggest_target_language,
    validate_language_pair
)

logger = logging.getLogger("doctranslator")

def split_text_into_chunks(text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Split text into appropriate chunks for translation.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk (default: 1000 characters)
        
    Returns:
        List of text chunks
    """
    logger.debug(f"Splitting text of length {len(text)} into chunks (max size: {max_chunk_size})")
    
    # Split by paragraphs (empty lines as delimiters)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        # If paragraph is very long, split further by sentences
        if len(paragraph) > max_chunk_size:
            logger.debug(f"Paragraph too long ({len(paragraph)} chars), splitting by sentences")
            
            # Split by sentences (including punctuation)
            sentences = re.split(r'([.!?。！？]\s*)', paragraph)
            i = 0
            while i < len(sentences):
                if i + 1 < len(sentences):
                    # Combine sentence with its punctuation
                    sentence = sentences[i] + sentences[i + 1]
                    i += 2
                else:
                    sentence = sentences[i]
                    i += 1
                
                if len(current_chunk) + len(sentence) <= max_chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                        logger.debug(f"Created chunk of size {len(current_chunk)}")
                    current_chunk = sentence
        else:
            # Add paragraph to current chunk or save as new chunk
            if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    logger.debug(f"Created chunk of size {len(current_chunk)}")
                current_chunk = paragraph
    
    # Add remaining text as chunk
    if current_chunk:
        chunks.append(current_chunk)
        logger.debug(f"Created final chunk of size {len(current_chunk)}")
    
    logger.info(f"Text split into {len(chunks)} chunks")
    return chunks

def translate_chunk(
    chunk: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    api_url: str
) -> str:
    """
    Translate a single text chunk using GenAI Hub API.
    
    Args:
        chunk: Text chunk to translate
        api_key: GenAI Hub API key
        model: Model name to use
        source_lang: Source language code
        target_lang: Target language code
        api_url: GenAI Hub API URL
        
    Returns:
        Translated text
        
    Raises:
        Exception: If translation fails
    """
    from app.core.translator import LANGUAGES
    
    logger.debug(f"Translating chunk of length {len(chunk)} with model {model}")
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Accept': 'application/json'
    }
    
    # Construct payload for GenAI Hub
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"You are a professional translator. Translate the following text from {LANGUAGES.get(source_lang, source_lang)} to {LANGUAGES.get(target_lang, target_lang)}. Provide only the direct translation without any additional explanations, comments, or formatting. Maintain the original text structure including line breaks and spacing."
            },
            {
                "role": "user",
                "content": chunk
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
        "stream": False
    }
    
    try:
        logger.debug(f"Sending API request to: {api_url}")
        
        response = requests.post(
            api_url, 
            headers=headers, 
            json=payload, 
            timeout=60
        )
        
        logger.debug(f"API response status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"API error: status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            response.raise_for_status()
        
        result = response.json()
        logger.debug(f"API response parsed successfully")
        
        # Parse GenAI Hub response structure
        if 'choices' in result and len(result['choices']) > 0:
            if 'message' in result['choices'][0]:
                translated_text = result['choices'][0]['message']['content'].strip()
                logger.debug(f"Translation successful (length: {len(translated_text)})")
                return translated_text
            elif 'text' in result['choices'][0]:
                translated_text = result['choices'][0]['text'].strip()
                logger.debug(f"Translation successful (length: {len(translated_text)})")
                return translated_text
        
        # Alternative response formats
        if 'content' in result:
            translated_text = result['content'].strip()
            logger.debug(f"Translation successful (length: {len(translated_text)})")
            return translated_text
        if 'text' in result:
            translated_text = result['text'].strip()
            logger.debug(f"Translation successful (length: {len(translated_text)})")
            return translated_text
        if 'response' in result:
            translated_text = result['response'].strip()
            logger.debug(f"Translation successful (length: {len(translated_text)})")
            return translated_text
        
        logger.error(f"Unexpected response format: {result}")
        raise Exception(f"Unexpected response format from API")
        
    except requests.exceptions.Timeout:
        logger.error("API request timeout")
        raise Exception("Translation request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request error: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Error response: {e.response.text}")
        raise Exception(f"API request failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during translation: {str(e)}")
        raise

def detect_language(text: str) -> str:
    """
    テキストから言語を自動検出（テキスト翻訳専用）
    
    注意: この関数はテキスト翻訳でのみ使用されます。
         ドキュメント翻訳では使用されません。
    
    Args:
        text: 判別対象のテキスト
        
    Returns:
        言語コード
    """
    logger.info("[Text Translation] Detecting language from text...")
    detected = detect_language_util(text)
    logger.info(f"[Text Translation] Detected language: {detected}")
    return detected

def translate_text_chunks(
    text: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    max_chunk_size: int = 1000,
    retry_count: int = 2,
    retry_delay: int = 2
) -> str:
    """
    Translate text by splitting into chunks if necessary.
    
    Args:
        text: Text to translate
        api_key: GenAI Hub API key
        model: Model name to use
        source_lang: Source language code
        target_lang: Target language code
        max_chunk_size: Maximum size of each chunk (default: 1000)
        retry_count: Number of retries on error (default: 2)
        retry_delay: Delay between retries in seconds (default: 2)
        
    Returns:
        Translated text
        
    Raises:
        ValueError: If API key or URL is not set
        Exception: If translation fails
    """
    from app.config import get_api_url
    
    logger.info(f"Starting text translation: {len(text)} characters, {source_lang} -> {target_lang}, Model: {model}")
    
    # Validate inputs
    if not text or not text.strip():
        logger.warning("Empty text provided for translation")
        return ""
    
    if not api_key:
        logger.error("API key is not set")
        raise ValueError("API key is not set. Please configure API key in settings.")
    
    api_url = get_api_url()
    if not api_url:
        logger.error("API URL is not set")
        raise ValueError("API URL is not set. Please configure API URL in settings.")
    
    # Mask API key for logging
    masked_api_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    logger.debug(f"Using API key: {masked_api_key}, API URL: {api_url}")
    
    try:
        # Check if text needs to be split
        if len(text) > 1500:
            logger.info(f"Text is long ({len(text)} chars), splitting into chunks")
            
            # Split text into chunks
            chunks = split_text_into_chunks(text, max_chunk_size)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            translated_chunks = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Translating chunk {i+1}/{len(chunks)}")
                
                # Retry logic for each chunk
                for attempt in range(retry_count + 1):
                    try:
                        chunk_result = translate_chunk(
                            chunk, 
                            api_key, 
                            model, 
                            source_lang, 
                            target_lang, 
                            api_url
                        )
                        translated_chunks.append(chunk_result)
                        logger.info(f"Chunk {i+1}/{len(chunks)} translated successfully")
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if attempt < retry_count:
                            wait_time = retry_delay * (attempt + 1)
                            logger.warning(f"Chunk {i+1} translation failed (attempt {attempt+1}/{retry_count+1}): {e}")
                            logger.info(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # Final attempt failed
                            error_msg = f"Error translating chunk {i+1}: {str(e)}"
                            logger.error(error_msg)
                            translated_chunks.append(f"[{error_msg}]")
                
                # Rate limiting: wait between chunks (except for last chunk)
                if i < len(chunks) - 1:
                    logger.debug(f"Waiting {retry_delay} seconds before next chunk...")
                    time.sleep(retry_delay)
            
            # Join translated chunks
            translated_text = "\n\n".join(translated_chunks)
            logger.info(f"All chunks translated. Total length: {len(translated_text)}")
            
            return translated_text
            
        else:
            # Short text: translate directly
            logger.info("Text is short, translating directly")
            
            # Retry logic
            for attempt in range(retry_count + 1):
                try:
                    translated_text = translate_chunk(
                        text, 
                        api_key, 
                        model, 
                        source_lang, 
                        target_lang, 
                        api_url
                    )
                    logger.info(f"Translation successful. Length: {len(translated_text)}")
                    return translated_text
                    
                except Exception as e:
                    if attempt < retry_count:
                        wait_time = retry_delay * (attempt + 1)
                        logger.warning(f"Translation failed (attempt {attempt+1}/{retry_count+1}): {e}")
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        # Final attempt failed
                        logger.error(f"Translation failed after {retry_count+1} attempts: {e}")
                        raise
    
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during text translation: {e}")
        raise

def detect_language(text: str) -> str:
    """
    Detect language from text content.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected language code (defaults to 'en' if uncertain)
    """
    if not text:
        return 'en'
    
    # Language character patterns
    language_patterns = {
        'ja': r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]',  # Hiragana, Katakana, Kanji
        'ko': r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]',  # Hangul
        'zh': r'[\u4E00-\u9FFF]',  # Chinese (Kanji)
        'hi': r'[\u0900-\u097F]',  # Devanagari
        'th': r'[\u0E00-\u0E7F]',  # Thai
        'vi': r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]',  # Vietnamese
        'fr': r'[àâäéèêëïîôöùûüÿç]',  # French
        'de': r'[äöüßÄÖÜ]',  # German
        'es': r'[ñáéíóúü¿¡]'  # Spanish
    }
    
    # Total character count (excluding whitespace)
    total_chars = len(re.sub(r'\s', '', text))
    if total_chars == 0:
        return 'en'
    
    # Count characters for each language
    for lang_code, pattern in language_patterns.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        char_count = len(matches)
        
        # If 30% or more characters match, identify as that language
        if char_count / total_chars >= 0.3:
            logger.debug(f"Language detected: {lang_code} ({char_count}/{total_chars} = {char_count/total_chars*100:.1f}%)")
            return lang_code
    
    # Default to English if no language reaches 30%
    logger.debug("No language detected with 30% threshold, defaulting to English")
    return 'en'

def optimize_text_for_translation(text: str) -> str:
    """
    Optimize text for translation by removing unnecessary whitespace.
    
    Args:
        text: Text to optimize
        
    Returns:
        Optimized text
    """
    # Remove excessive blank lines (keep max 1 blank line)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split('\n')]
    
    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    optimized_text = '\n'.join(lines)
    
    if len(optimized_text) != len(text):
        logger.debug(f"Text optimized: {len(text)} -> {len(optimized_text)} characters")
    
    return optimized_text

def validate_translation_params(
    text: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str
) -> tuple[bool, Optional[str]]:
    """
    Validate translation parameters.
    
    Args:
        text: Text to translate
        api_key: API key
        model: Model name
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        (is_valid, error_message)
    """
    from app.core.translator import LANGUAGES
    
    if not text or not text.strip():
        return False, "Text is empty"
    
    if not api_key or len(api_key) < 20:
        return False, "Invalid API key"
    
    if not model:
        return False, "Model is not specified"
    
    if source_lang not in LANGUAGES:
        return False, f"Unsupported source language: {source_lang}"
    
    if target_lang not in LANGUAGES:
        return False, f"Unsupported target language: {target_lang}"
    
    if source_lang == target_lang:
        return False, "Source and target languages must be different"
    
    return True, None

def translate_text_with_retry(
    text: str,
    api_key: str,
    model: str,
    source_lang: str,
    target_lang: str,
    api_url: str,
    max_retries: int = 3,
    base_delay: int = 2
) -> str:
    """
    Translate text with exponential backoff retry logic.
    
    Args:
        text: Text to translate
        api_key: GenAI Hub API key
        model: Model name
        source_lang: Source language code
        target_lang: Target language code
        api_url: API URL
        max_retries: Maximum number of retries (default: 3)
        base_delay: Base delay for exponential backoff (default: 2 seconds)
        
    Returns:
        Translated text
        
    Raises:
        Exception: If all retries fail
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return translate_chunk(text, api_key, model, source_lang, target_lang, api_url)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                # Exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"All {max_retries} translation attempts failed")
    
    raise last_error

def estimate_translation_time(text_length: int, chunk_size: int = 1000) -> int:
    """
    Estimate translation time in seconds.
    
    Args:
        text_length: Length of text to translate
        chunk_size: Size of each chunk
        
    Returns:
        Estimated time in seconds
    """
    if text_length <= 1500:
        # Short text: ~5 seconds
        return 5
    
    # Calculate number of chunks
    num_chunks = (text_length + chunk_size - 1) // chunk_size
    
    # Estimate: 5 seconds per chunk + 2 seconds delay between chunks
    estimated_time = (num_chunks * 5) + ((num_chunks - 1) * 2)
    
    logger.debug(f"Estimated translation time: {estimated_time} seconds for {num_chunks} chunks")
    
    return estimated_time

def format_translation_stats(
    original_length: int,
    translated_length: int,
    num_chunks: int,
    elapsed_time: float
) -> dict:
    """
    Format translation statistics.
    
    Args:
        original_length: Original text length
        translated_length: Translated text length
        num_chunks: Number of chunks processed
        elapsed_time: Time taken in seconds
        
    Returns:
        Statistics dictionary
    """
    return {
        "original_length": original_length,
        "translated_length": translated_length,
        "num_chunks": num_chunks,
        "elapsed_time": round(elapsed_time, 2),
        "chars_per_second": round(original_length / elapsed_time, 2) if elapsed_time > 0 else 0,
        "length_ratio": round(translated_length / original_length, 2) if original_length > 0 else 0
    }

def clean_translated_text(text: str) -> str:
    """
    Clean up translated text by removing common artifacts.
    
    Args:
        text: Translated text
        
    Returns:
        Cleaned text
    """
    # Remove common prefixes that AI might add
    prefixes_to_remove = [
        "Translation:",
        "Translated text:",
        "Here is the translation:",
        "Here's the translation:",
    ]
    
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            logger.debug(f"Removed prefix: {prefix}")
    
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    return text.strip()

# ===== Main Translation Function =====

def translate_text(
    text: str,
    api_key: str,
    model: str,
    source_lang: str = "en",
    target_lang: str = "ja",
    optimize: bool = True,
    clean_output: bool = True
) -> dict:
    """
    Main text translation function with full features.
    
    Args:
        text: Text to translate
        api_key: GenAI Hub API key
        model: Model name to use
        source_lang: Source language code (default: 'en')
        target_lang: Target language code (default: 'ja')
        optimize: Whether to optimize text before translation (default: True)
        clean_output: Whether to clean translated text (default: True)
        
    Returns:
        Dictionary containing translation result and metadata
        
    Raises:
        ValueError: If validation fails
        Exception: If translation fails
    """
    import time as time_module
    
    start_time = time_module.time()
    
    logger.info(f"=== Starting text translation ===")
    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Languages: {source_lang} -> {target_lang}")
    logger.info(f"Model: {model}")
    
    # Validate parameters
    is_valid, error_msg = validate_translation_params(
        text, api_key, model, source_lang, target_lang
    )
    
    if not is_valid:
        logger.error(f"Validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    # Optimize text if requested
    if optimize:
        original_length = len(text)
        text = optimize_text_for_translation(text)
        if len(text) != original_length:
            logger.info(f"Text optimized: {original_length} -> {len(text)} characters")
    
    # Translate
    try:
        translated_text = translate_text_chunks(
            text=text,
            api_key=api_key,
            model=model,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Clean output if requested
        if clean_output:
            original_translated_length = len(translated_text)
            translated_text = clean_translated_text(translated_text)
            if len(translated_text) != original_translated_length:
                logger.info(f"Translation cleaned: {original_translated_length} -> {len(translated_text)} characters")
        
        # Calculate statistics
        elapsed_time = time_module.time() - start_time
        num_chunks = (len(text) + 999) // 1000  # Estimate chunks
        
        stats = format_translation_stats(
            len(text),
            len(translated_text),
            num_chunks,
            elapsed_time
        )
        
        logger.info(f"=== Translation completed ===")
        logger.info(f"Statistics: {stats}")
        
        return {
            "success": True,
            "translated_text": translated_text,
            "statistics": stats,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model
        }
        
    except Exception as e:
        elapsed_time = time_module.time() - start_time
        logger.error(f"Translation failed after {elapsed_time:.2f} seconds: {e}")
        raise

# ===== Utility Functions =====

def get_supported_languages() -> dict:
    """
    Get list of supported languages.
    
    Returns:
        Dictionary of language codes and names
    """
    from app.core.translator import LANGUAGES
    return LANGUAGES

def is_long_text(text: str, threshold: int = 1500) -> bool:
    """
    Check if text is considered long.
    
    Args:
        text: Text to check
        threshold: Length threshold (default: 1500)
        
    Returns:
        True if text is long
    """
    return len(text) > threshold

def calculate_chunk_count(text: str, chunk_size: int = 1000) -> int:
    """
    Calculate number of chunks needed for text.
    
    Args:
        text: Text to analyze
        chunk_size: Size of each chunk
        
    Returns:
        Number of chunks
    """
    return (len(text) + chunk_size - 1) // chunk_size

# ===== Export Functions =====

__all__ = [
    "split_text_into_chunks",
    "translate_chunk",
    "translate_text_chunks",
    "detect_language",  # 新規追加（テキスト翻訳専用）
    "optimize_text_for_translation",
    "validate_translation_params",
    "translate_text_with_retry",
    "estimate_translation_time",
    "format_translation_stats",
    "clean_translated_text",
    "translate_text",
    "get_supported_languages",
    "is_long_text",
    "calculate_chunk_count",
]

# ===== Module Test Code =====

if __name__ == "__main__":
    # Simple test
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    api_key = os.getenv("GENAI_HUB_API_KEY")
    api_url = os.getenv("GENAI_HUB_API_URL")
    
    if not api_key or not api_url:
        print("Error: GENAI_HUB_API_KEY or GENAI_HUB_API_URL not set")
        print("Please set environment variables in .env file")
        exit(1)
    
    # Test translation
    test_text = "Hello, world! This is a test translation."
    print(f"Test text: {test_text}")
    print(f"Detecting language...")
    
    detected_lang = detect_language(test_text)
    print(f"Detected language: {detected_lang}")
    
    print(f"\nTranslating from English to Japanese...")
    
    try:
        result = translate_text(
            text=test_text,
            api_key=api_key,
            model="claude-3-5-haiku",  # Use default model
            source_lang="en",
            target_lang="ja"
        )
        
        print(f"\nTranslation successful!")
        print(f"Original: {test_text}")
        print(f"Translated: {result['translated_text']}")
        print(f"\nStatistics:")
        for key, value in result['statistics'].items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"\nTranslation failed: {e}")
        exit(1)
