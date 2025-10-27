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
        # Create directory if it doesn't exist
        ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
        
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

def validate_api_settings() -> bool:
    """
    Validate API settings
    
    Returns:
        True if settings are valid
    """
    try:
        api_key = get_api_key()
        api_url = get_api_url()
        
        if not api_key or not api_url:
            logger.debug("API key or URL is missing")
            return False
            
        # Simple connection test
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Test models endpoint
        if api_url.endswith("/chat/completions"):
            test_url = api_url.replace("/chat/completions", "/models")
        elif api_url.endswith("/v1/chat/completions"):
            test_url = api_url.replace("/v1/chat/completions", "/v1/models")
        else:
            test_url = api_url.rstrip("/") + "/models"
        
        logger.debug(f"Testing API connection to: {test_url}")
        response = requests.get(test_url, headers=headers, timeout=10)
        
        is_valid = response.status_code == 200
        logger.debug(f"API validation result: {is_valid} (status: {response.status_code})")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"API settings validation error: {e}")
        return False

def clear_api_settings() -> bool:
    """
    Clear API settings
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Remove from environment variables
        if "GENAI_HUB_API_KEY" in os.environ:
            del os.environ["GENAI_HUB_API_KEY"]
        if "GENAI_HUB_API_URL" in os.environ:
            del os.environ["GENAI_HUB_API_URL"]
        
        # Delete or empty .env file
        if ENV_FILE.exists():
            ENV_FILE.unlink()
            logger.info(f"Deleted .env file: {ENV_FILE}")
        
        logger.info("API settings cleared successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to clear API settings: {e}")
        return False

def get_config_info() -> dict:
    """
    Get current configuration information (for debugging)
    
    Returns:
        Configuration information dictionary
    """
    api_key = get_api_key()
    api_url = get_api_url()
    
    # Mask API key for security
    masked_key = ""
    if api_key:
        if len(api_key) > 8:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}"
        else:
            masked_key = "****"
    
    return {
        "env_file": str(ENV_FILE),
        "env_file_exists": ENV_FILE.exists(),
        "api_key_set": bool(api_key),
        "api_key_masked": masked_key,
        "api_key_length": len(api_key) if api_key else 0,
        "api_url": api_url,
        "api_url_set": bool(api_url),
        "settings_valid": api_settings_exist(),
        "app_dir": str(APP_DIR)
    }

def update_api_settings(api_key: str = None, api_url: str = None) -> bool:
    """
    Partially update existing API settings
    
    Args:
        api_key: New API key (None to keep existing value)
        api_url: New API URL (None to keep existing value)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Get current settings
        current_api_key = get_api_key()
        current_api_url = get_api_url()
        
        # Use new values or existing values
        final_api_key = api_key if api_key is not None else current_api_key
        final_api_url = api_url if api_url is not None else current_api_url
        
        # Save settings
        return save_api_settings(final_api_key, final_api_url)
        
    except Exception as e:
        logger.error(f"Failed to update API settings: {e}")
        return False

def test_api_connection() -> tuple[bool, str]:
    """
    Test API connection
    
    Returns:
        (success/failure, message)
    """
    try:
        api_key = get_api_key()
        api_url = get_api_url()
        
        if not api_key:
            return False, "API key is not set"
        
        if not api_url:
            return False, "API URL is not set"
        
        import requests
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # Test models endpoint
        if api_url.endswith("/chat/completions"):
            test_url = api_url.replace("/chat/completions", "/models")
        elif api_url.endswith("/v1/chat/completions"):
            test_url = api_url.replace("/v1/chat/completions", "/v1/models")
        else:
            test_url = api_url.rstrip("/") + "/models"
        
        logger.info(f"Testing API connection to: {test_url}")
        response = requests.get(test_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                if "content" in data and isinstance(data["content"], list):
                    model_count = len(data["content"])
                    return True, f"Connection successful. Found {model_count} models."
                else:
                    return True, "Connection successful, but unexpected response format."
            except Exception as e:
                return True, f"Connection successful, but failed to parse response: {str(e)}"
        else:
            return False, f"Connection failed with status {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        return False, "Connection timeout. Please check your network and API URL."
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Please check your network and API URL."
    except Exception as e:
        return False, f"Connection test failed: {str(e)}"

def get_default_model() -> str:
    """
    Automatically select the first available model
    
    Returns:
        Default model name
    """
    try:
        # Check if API settings exist
        if not api_settings_exist():
            logger.warning("API settings not found, using fallback model")
            return "claude-3-5-haiku"
        
        # Get available models
        from app.core.translator import fetch_available_models
        models = fetch_available_models()
        
        if models:
            # Return the first model in alphabetical order
            default_model = sorted(models.keys())[0]
            logger.info(f"Auto-selected default model: {default_model}")
            return default_model
        else:
            logger.warning("No available models found. Using fallback model")
            return "claude-3-5-haiku"
            
    except Exception as e:
        logger.error(f"Error getting default model: {e}")
        return "claude-3-5-haiku"  # Fallback on error

# Export functions
__all__ = [
    "save_api_key",
    "save_api_settings", 
    "get_api_key",
    "get_api_url",
    "api_key_exists",
    "api_settings_exist",
    "validate_api_settings",
    "clear_api_settings",
    "get_config_info",
    "load_env_file",
    "update_api_settings",
    "test_api_connection",
    "get_default_model",
]
