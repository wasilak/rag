import os
import logging
from typing import Dict, List, Optional
import ollama
import openai
from google.ai import generativelanguage as glanguage
from google.api_core import client_options

logger = logging.getLogger("RAG")


class ModelDefaults:
    """Default models for each LLM provider"""

    OLLAMA = {
        "chat": "qwen3:8b",
        "embedding": "nomic-embed-text"
    }

    OPENAI = {
        "chat": "gpt-4o",
        "embedding": "text-embedding-3-small"
    }

    GEMINI = {
        "chat": "gemini-1.5-flash",
        "embedding": "text-embedding-004"
    }


class ModelManager:
    """Manages model selection and validation for different LLM providers"""

    def __init__(self, ollama_host: str, ollama_port: int):
        self.defaults = ModelDefaults()
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port

    def get_default_model(self, provider: str, model_type: str) -> str:
        """Get default model for a provider and type

        Args:
            provider: LLM provider (ollama, openai, gemini)
            model_type: Type of model (chat, embedding)

        Returns:
            Default model name
        """
        provider_defaults = getattr(self.defaults, provider.upper(), None)
        if not provider_defaults:
            raise ValueError(f"Unknown provider: {provider}")

        if model_type not in provider_defaults:
            raise ValueError(f"Unknown model type: {model_type}")

        return provider_defaults[model_type]

    def list_models(self, provider: str) -> List[Dict]:
        """List available models for a provider

        Args:
            provider: LLM provider (ollama, openai, gemini)

        Returns:
            List of model information dictionaries
        """
        logger.info(f"Listing models for {provider}")

        if provider == "ollama":
            return self._list_ollama_models()
        elif provider == "openai":
            return self._list_openai_models()
        elif provider == "gemini":
            return self._list_gemini_models()
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def validate_model(self, provider: str, model_name: str, model_type: str) -> bool:
        """Validate if a model exists for a provider

        Args:
            provider: LLM provider (ollama, openai, gemini)
            model_name: Name of the model to validate
            model_type: Type of model (chat, embedding)

        Returns:
            True if model exists, False otherwise
        """
        logger.debug(f"Validating {model_type} model '{model_name}' for {provider}")

        try:
            available_models = self.list_models(provider)

            # Check if model exists in the list
            for model in available_models:
                model_id = model.get('id', '')
                model_full_name = model.get('name', '')

                # For Ollama, handle tag format (e.g., "nomic-embed-text:latest")
                if provider == "ollama":
                    model_base_name = model_id.split(':')[0] if ':' in model_id else model_id
                    if (model_id == model_name or
                        model_full_name == model_name or
                        model_base_name == model_name):
                        return True
                else:
                    # For other providers, exact match
                    if model_id == model_name or model_full_name == model_name:
                        # Additional validation for model type if available
                        return True

            logger.warning(f"Model '{model_name}' not found for {provider}")
            return False

        except Exception as e:
            logger.error(f"Error validating model '{model_name}' for {provider}: {e}")
            return False

    def get_validated_model(self, provider: str, model_name: Optional[str], model_type: str) -> str:
        """Get validated model name, falling back to default if needed

        Args:
            provider: LLM provider (ollama, openai, gemini)
            model_name: Requested model name (None for default)
            model_type: Type of model (chat, embedding)

        Returns:
            Validated model name
        """
        # Use default if no model specified
        if not model_name:
            model_name = self.get_default_model(provider, model_type)
            logger.debug(f"Using default {model_type} model for {provider}: {model_name}")
            return model_name

        # Check if custom model is valid
        if self.validate_model(provider, model_name, model_type):
            logger.debug(f"Using validated {model_type} model for {provider}: {model_name}")
            return model_name

        # Fall back to default if validation fails
        default_model = self.get_default_model(provider, model_type)
        logger.warning(f"Model '{model_name}' not valid for {provider}, using default: {default_model}")
        return default_model

    def _list_ollama_models(self) -> List[Dict]:
        """List Ollama models"""
        try:
            base_url = f"http://{self.ollama_host}:{self.ollama_port}"
            client = ollama.Client(host=base_url)
            response = client.list()

            # Format the response to match other providers
            formatted_models = []

            # Handle both dict and object response formats
            if hasattr(response, 'models'):
                models_list = response.models
            elif isinstance(response, dict) and 'models' in response:
                models_list = response['models']
            else:
                logger.warning(f"Unexpected Ollama response format: {response}")
                return []

            for model in models_list:
                # Handle both dict and object model formats
                if hasattr(model, 'model'):
                    name = model.model
                    size = getattr(model, 'size', 0)
                    modified_at = getattr(model, 'modified_at', '')
                elif isinstance(model, dict):
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0)
                    modified_at = model.get('modified_at', '')
                else:
                    name = str(model)
                    size = 0
                    modified_at = ''

                formatted_models.append({
                    'id': name,
                    'name': name,
                    'size': size,
                    'modified_at': modified_at,
                    'provider': 'ollama'
                })

            return formatted_models

        except Exception as e:
            logger.error(f"Error listing Ollama models: {e}")
            return []

    def _list_openai_models(self) -> List[Dict]:
        """List OpenAI models"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.error("OPENAI_API_KEY not found in environment")
                return []

            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()

            # Format the response
            formatted_models = []
            for model in models.data:
                formatted_models.append({
                    'id': model.id,
                    'name': model.id,
                    'created': model.created,
                    'owned_by': model.owned_by,
                    'provider': 'openai'
                })

            return formatted_models

        except Exception as e:
            logger.error(f"Error listing OpenAI models: {e}")
            return []

    def _list_gemini_models(self) -> List[Dict]:
        """List Gemini models"""
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.error("GEMINI_API_KEY not found in environment")
                return []

            # Create client with anonymous credentials and API key
            client_opts = client_options.ClientOptions(
                api_endpoint="generativelanguage.googleapis.com",
                api_key=api_key
            )
            model_client = glanguage.ModelServiceClient(
                client_options=client_opts,
            )
            request = glanguage.ListModelsRequest()
            response = model_client.list_models(request)

            formatted_models = []
            for model in response.models:
                model_id = model.name.split('/')[-1]
                formatted_models.append({
                    'id': model_id,
                    'name': model_id,
                    'display_name': model.display_name,
                    'description': model.description,
                    'provider': 'gemini'
                })

            return formatted_models

        except Exception as e:
            logger.error(f"Error listing Gemini models: {e}")
            return []


# Singleton instance
_model_manager_instance: Optional[ModelManager] = None

def get_model_manager(embedding_ollama_host: str, embedding_ollama_port: int) -> ModelManager:
    """Get singleton instance of ModelManager"""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager(embedding_ollama_host, embedding_ollama_port)
    return _model_manager_instance


def list_provider_models(provider: str, embedding_ollama_host: str, embedding_ollama_port: int) -> List[Dict]:
    """Convenience function to list models for a provider"""
    manager = get_model_manager(embedding_ollama_host, embedding_ollama_port)
    return manager.list_models(provider)


def validate_model_choice(provider: str, embedding_ollama_host: str, embedding_ollama_port: int, model_name: str, model_type: str) -> bool:
    """Convenience function to validate a model choice"""
    manager = get_model_manager(embedding_ollama_host, embedding_ollama_port)
    return manager.validate_model(provider, model_name, model_type)


def get_best_model(provider: str, embedding_ollama_host: str, embedding_ollama_port: int, model_name: Optional[str], model_type: str) -> str:
    """Convenience function to get the best available model"""
    manager = get_model_manager(embedding_ollama_host, embedding_ollama_port)
    return manager.get_validated_model(provider, model_name, model_type)
