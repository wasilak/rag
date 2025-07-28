"""Gemini models management module."""
import os
import logging
from typing import Dict, List, Any
from google.ai import generativelanguage as glanguage
from google.oauth2 import service_account
from google.api_core import client_options
from google.auth import credentials as google_auth

logger = logging.getLogger("RAG")

# Predefined models for OpenAI compatibility API
OPENAI_COMPAT_MODELS = [
    {
        'id': 'gemini-1.5-pro',
        'name': 'gemini-1.5-pro',
        'display_name': 'Gemini 1.5 Pro',
        'provider': 'gemini',
        'capabilities': ['generateContent']
    },
    {
        'id': 'gemini-1.5-flash',
        'name': 'gemini-1.5-flash',
        'display_name': 'Gemini 1.5 Flash',
        'provider': 'gemini',
        'capabilities': ['generateContent']
    }
]

def get_gemini_models() -> List[Dict[str, Any]]:
    """Get list of available Gemini models.

    When using OpenAI compatibility API, returns predefined list.
    When using native API, attempts to fetch from API.

    Returns:
        List of model information dictionaries
    """
    # Check if we're using native API (service account or API key)
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_CLOUD_PROJECT"):
        return _get_native_models()
    else:
        # Using OpenAI compatibility API
        return OPENAI_COMPAT_MODELS

def _get_native_models() -> List[Dict[str, Any]]:
    """Get models list from native Gemini API.

    Returns:
        List of model information dictionaries
    """
    try:
        # Check for service account credentials first
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
            except Exception as e:
                logger.error(f"Error loading service account credentials: {e}")
                return OPENAI_COMPAT_MODELS
        else:
            # Fall back to API key
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("Neither GOOGLE_APPLICATION_CREDENTIALS nor GEMINI_API_KEY found in environment")
                return OPENAI_COMPAT_MODELS
            credentials = google_auth.AnonymousCredentials()

        # Initialize Gemini client
        client_opts = client_options.ClientOptions(
            api_endpoint="generativelanguage.googleapis.com",
            quota_project_id=os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        model_client = glanguage.ModelServiceClient(
            credentials=credentials,
            client_options=client_opts
        )

        # List available models
        request = glanguage.ListModelsRequest()
        response = model_client.list_models(request)

        # Format the response
        formatted_models: List[Dict[str, Any]] = []
        for model in response.models:
            model_id = model.name.split('/')[-1]
            formatted_models.append({
                'id': model_id,
                'name': model_id,
                'display_name': model.display_name,
                'description': model.description,
                'supported_generation_methods': [method for method in model.supported_generation_methods],
                'capabilities': ['generateContent'],  # All Gemini models support content generation
                'provider': 'gemini'
            })

        return formatted_models

    except Exception as e:
        logger.error(f"Error listing Gemini models via native API: {e}")
        # Fall back to predefined models
        return OPENAI_COMPAT_MODELS

def validate_gemini_model(model_name: str) -> bool:
    """Validate if a Gemini model exists.

    Args:
        model_name: Name of the model to validate

    Returns:
        True if model exists (either in predefined list or native API), False otherwise
    """
    available_models = get_gemini_models()
    return any(model['id'] == model_name or model['name'] == model_name
              for model in available_models)
