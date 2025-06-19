import os
from pathlib import Path
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Get application directory
APP_DIR = Path(__file__).parent.parent
ENV_FILE = APP_DIR / ".env"


def load_env_file():
    """Load environment variables from .env file"""
    if ENV_FILE.exists():
        logger.debug(f"Loading environment variables from {ENV_FILE}")
        load_dotenv(ENV_FILE)
        logger.debug(
            f"Environment variables loaded: API Key: {os.environ.get('GENAI_HUB_API_KEY') is not None}, API URL: {os.environ.get('GENAI_HUB_API_URL') is not None}"
        )
    else:
        logger.warning(f".env file not found at {ENV_FILE}")


# Load environment variables at module import
load_env_file()


def save_api_key(api_key: str) -> bool:
    """
    Save API key to .env file and environment variable (backward compatibility)
    """
    return save_api_settings(api_key, get_api_url())


def save_api_settings(api_key: str, api_url: str) -> bool:
    """
    Save API key and URL to .env file and environment variables
    """
    try:
        # Save to .env file
        with open(ENV_FILE, "w") as f:
            f.write(f"GENAI_HUB_API_KEY={api_key}\n")
            f.write(f"GENAI_HUB_API_URL={api_url}\n")

        # Set environment variables
        os.environ["GENAI_HUB_API_KEY"] = api_key
        os.environ["GENAI_HUB_API_URL"] = api_url

        logger.info("API settings saved successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to save API settings: {e}")
        return False


def get_api_key() -> str:
    """Get API key from environment variable"""
    # Ensure environment variables are loaded
    if not os.environ.get("GENAI_HUB_API_KEY"):
        load_env_file()

    api_key = os.environ.get("GENAI_HUB_API_KEY", "")
    logger.debug(f"API key exists: {bool(api_key)}")
    return api_key


def get_api_url() -> str:
    """Get API URL from environment variable"""
    # Ensure environment variables are loaded
    if not os.environ.get("GENAI_HUB_API_URL"):
        load_env_file()

    api_url = os.environ.get("GENAI_HUB_API_URL", "")
    logger.debug(f"API URL: {api_url}")
    return api_url


def api_key_exists() -> bool:
    """Check if API key exists and is valid (backward compatibility)"""
    return api_settings_exist()


def api_settings_exist() -> bool:
    """Check if API key and URL exist and are valid"""
    api_key = get_api_key()
    api_url = get_api_url()
    
    key_valid = bool(api_key and len(api_key.strip()) >= 20)
    url_valid = bool(api_url and api_url.startswith("http"))
    
    logger.debug(f"API settings valid: Key: {key_valid}, URL: {url_valid}")
    return key_valid and url_valid
