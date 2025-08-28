"""
LLM Utilities

Centralized utilities for LLM operations with proper provider detection and response handling.
"""

import json
import os
from typing import Any

import openai

from ..config.logfire_config import get_logger, search_logger
from .credential_service import credential_service
from .llm_provider_service import _get_cached_settings, _set_cached_settings

logger = get_logger(__name__)


def _get_model_choice() -> str:
    """Get MODEL_CHOICE with direct fallback and provider awareness."""
    try:
        # Direct cache/env fallback
        model = None
        if credential_service._cache_initialized and "MODEL_CHOICE" in credential_service._cache:
            cached_model = credential_service._cache["MODEL_CHOICE"]
            if isinstance(cached_model, dict) and cached_model.get("is_encrypted"):
                try:
                    model = credential_service._decrypt_value(cached_model["encrypted_value"])
                except:
                    model = cached_model.get("value", "")
            else:
                model = cached_model
        
        if not model:
            model = os.getenv("MODEL_CHOICE", "gpt-4o-mini")
        
        # Get the current provider to validate model compatibility
        cache_key = "provider_config_llm"
        provider_config = _get_cached_settings(cache_key)
        if not provider_config and credential_service._cache_initialized:
            # Try to get provider from credential service cache
            llm_provider = credential_service._cache.get("LLM_PROVIDER", "openai")
            if isinstance(llm_provider, dict) and llm_provider.get("is_encrypted"):
                try:
                    llm_provider = credential_service._decrypt_value(llm_provider["encrypted_value"])
                except:
                    llm_provider = "openai"
            provider_name = llm_provider
        else:
            provider_name = provider_config.get("provider", "openai") if provider_config else "openai"
            
        # If using Ollama with an OpenAI model name, warn but continue
        if provider_name == "ollama" and model.startswith("gpt-"):
            logger.warning(
                f"Using OpenAI model '{model}' with Ollama provider. "
                f"Ensure this model exists in your Ollama installation."
            )
        # If using OpenAI with an Ollama model name, use a default OpenAI model
        elif provider_name == "openai" and not model.startswith(("gpt-", "o1-", "text-")):
            logger.warning(
                f"Model '{model}' may not be available on OpenAI. "
                f"Consider using 'gpt-4o-mini' or 'gpt-4o' for OpenAI."
            )
            # Don't override - let the user's choice stand but warn
        
        logger.debug(f"Using model choice: {model} for provider: {provider_name}")
        return model
    except Exception as e:
        logger.warning(f"Error getting model choice: {e}, using default")
        return "gpt-4o-mini"


def get_sync_llm_client(provider: str | None = None) -> openai.OpenAI:
    """
    Create a synchronous OpenAI-compatible client based on the configured provider.
    
    Args:
        provider: Optional provider override
        
    Returns:
        An OpenAI client configured for the selected provider
        
    Raises:
        ValueError: If provider configuration is invalid or missing
    """
    try:
        # Try to get provider configuration
        if provider:
            # Explicit provider requested
            provider_name = provider
            
            # Get API key and settings for the provider
            api_key = None
            base_url = None
            
            # Check cache for rag_settings
            cache_key = "rag_strategy_settings"
            rag_settings = _get_cached_settings(cache_key)
            if rag_settings is None:
                # Load rag settings from credential service if not cached
                if credential_service._cache_initialized:
                    # Try to get from the main cache
                    rag_keys = ["LLM_BASE_URL", "LLM_PROVIDER", "MODEL_CHOICE", "EMBEDDING_MODEL"]
                    rag_settings = {}
                    for key in rag_keys:
                        if key in credential_service._cache:
                            cached_val = credential_service._cache[key]
                            if isinstance(cached_val, dict) and cached_val.get("is_encrypted"):
                                try:
                                    rag_settings[key] = credential_service._decrypt_value(cached_val["encrypted_value"])
                                except:
                                    rag_settings[key] = cached_val.get("value", "")
                            else:
                                rag_settings[key] = cached_val
                    if rag_settings:
                        _set_cached_settings(cache_key, rag_settings)
                else:
                    rag_settings = {}
            
            # Get provider-specific settings
            if provider_name == "ollama":
                api_key = "ollama"  # Required but unused
                base_url = rag_settings.get("LLM_BASE_URL", "http://localhost:11434/v1")
            elif provider_name == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key and credential_service._cache_initialized:
                    cached_key = credential_service._cache.get("OPENAI_API_KEY")
                    if isinstance(cached_key, dict) and cached_key.get("is_encrypted"):
                        api_key = credential_service._decrypt_value(cached_key["encrypted_value"])
                    elif cached_key:
                        api_key = cached_key
            elif provider_name == "google":
                api_key = os.getenv("GOOGLE_API_KEY") 
                if not api_key and credential_service._cache_initialized:
                    cached_key = credential_service._cache.get("GOOGLE_API_KEY")
                    if isinstance(cached_key, dict) and cached_key.get("is_encrypted"):
                        api_key = credential_service._decrypt_value(cached_key["encrypted_value"])
                    elif cached_key:
                        api_key = cached_key
                base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        else:
            # Get configured provider - try cache first, then credential service
            cache_key = "provider_config_llm"
            provider_config = _get_cached_settings(cache_key)
            
            if provider_config is None and credential_service._cache_initialized:
                # Try to build provider config from credential service cache
                try:
                    # Get LLM provider setting
                    llm_provider = credential_service._cache.get("LLM_PROVIDER", "openai")
                    if isinstance(llm_provider, dict) and llm_provider.get("is_encrypted"):
                        try:
                            llm_provider = credential_service._decrypt_value(llm_provider["encrypted_value"])
                        except:
                            llm_provider = "openai"
                    
                    # Get API key for this provider
                    if llm_provider == "openai":
                        api_key = credential_service._cache.get("OPENAI_API_KEY")
                        if isinstance(api_key, dict) and api_key.get("is_encrypted"):
                            try:
                                api_key = credential_service._decrypt_value(api_key["encrypted_value"])
                            except:
                                api_key = os.getenv("OPENAI_API_KEY")
                        base_url = None
                    elif llm_provider == "ollama":
                        api_key = "ollama"
                        base_url_cached = credential_service._cache.get("LLM_BASE_URL")
                        if isinstance(base_url_cached, dict) and base_url_cached.get("is_encrypted"):
                            try:
                                base_url = credential_service._decrypt_value(base_url_cached["encrypted_value"])
                            except:
                                base_url = "http://localhost:11434/v1"
                        else:
                            base_url = base_url_cached or "http://localhost:11434/v1"
                    elif llm_provider == "google":
                        api_key = credential_service._cache.get("GOOGLE_API_KEY")
                        if isinstance(api_key, dict) and api_key.get("is_encrypted"):
                            try:
                                api_key = credential_service._decrypt_value(api_key["encrypted_value"])
                            except:
                                api_key = os.getenv("GOOGLE_API_KEY")
                        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
                    
                    # Cache the provider config for next time
                    provider_config = {
                        "provider": llm_provider,
                        "api_key": api_key,
                        "base_url": base_url
                    }
                    _set_cached_settings(cache_key, provider_config)
                    
                    provider_name = llm_provider
                    
                except Exception as e:
                    logger.warning(f"Failed to build provider config from cache: {e}")
                    provider_config = None
            
            if provider_config is None:
                # Ultimate fallback to OpenAI
                logger.warning("No provider config available, falling back to OpenAI")
                provider_name = "openai"
                api_key = os.getenv("OPENAI_API_KEY")
                base_url = None
            else:
                provider_name = provider_config["provider"]
                api_key = provider_config["api_key"]
                base_url = provider_config.get("base_url")
        
        logger.debug(f"Creating synchronous LLM client for provider: {provider_name}")
        
        # Create the appropriate client
        if provider_name == "openai":
            if not api_key:
                raise ValueError("OpenAI API key not found")
            client = openai.OpenAI(api_key=api_key)
            logger.debug("OpenAI client created successfully")
            
        elif provider_name == "ollama":
            client = openai.OpenAI(
                api_key=api_key or "ollama",
                base_url=base_url or "http://localhost:11434/v1",
            )
            logger.debug(f"Ollama client created successfully with base URL: {base_url}")
            
        elif provider_name == "google":
            if not api_key:
                raise ValueError("Google API key not found")
            client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            logger.debug("Google Gemini client created successfully")
            
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
            
        return client
        
    except Exception as e:
        logger.error(f"Error creating LLM client: {e}")
        # Try fallback to basic OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            logger.info("Falling back to basic OpenAI client")
            return openai.OpenAI(api_key=api_key)
        raise


def parse_llm_json_response(
    response_content: str, 
    expected_fields: list[str] | None = None,
    provider_name: str = "unknown"
) -> dict[str, Any]:
    """
    Parse JSON response from LLM with fallbacks for different provider formats.
    
    Args:
        response_content: Raw response from the LLM
        expected_fields: List of expected field names in the response
        provider_name: Name of the provider for better error messages
        
    Returns:
        Parsed dictionary from the JSON response
        
    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON
    """
    if not response_content:
        raise ValueError("Empty response from LLM")
    
    # Try direct JSON parsing first
    try:
        result = json.loads(response_content)
        
        # Validate expected fields if provided
        if expected_fields:
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                logger.warning(
                    f"Response from {provider_name} missing fields: {missing_fields}. "
                    f"Got: {list(result.keys())}"
                )
        
        return result
        
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parsing failed: {e}")
        
        # Try to extract JSON from markdown code blocks
        if "```json" in response_content:
            try:
                json_start = response_content.index("```json") + 7
                json_end = response_content.index("```", json_start)
                json_str = response_content[json_start:json_end].strip()
                result = json.loads(json_str)
                logger.debug("Successfully extracted JSON from markdown code block")
                return result
            except (ValueError, json.JSONDecodeError):
                pass
        
        # Try to find JSON object in the response with improved patterns
        import re
        
        # Try multiple JSON extraction patterns
        json_patterns = [
            r'\{[^{}]*\}',  # Simple single-level JSON
            r'\{.*?\}',     # Greedy JSON extraction  
            r'\{[\s\S]*\}', # Multi-line JSON with any characters
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response_content, re.DOTALL)
            for match in matches:
                try:
                    # Clean up common issues
                    cleaned_match = match.strip()
                    # Remove any trailing text after the closing brace
                    if '}' in cleaned_match:
                        end_pos = cleaned_match.rfind('}')
                        cleaned_match = cleaned_match[:end_pos + 1]
                    
                    result = json.loads(cleaned_match)
                    if expected_fields is None or any(field in result for field in expected_fields):
                        logger.debug("Successfully extracted JSON object from response")
                        return result
                except json.JSONDecodeError:
                    continue
        
        # Try to extract JSON from conversational responses 
        # Look for patterns like 'example_name: "something"' and convert to JSON
        if expected_fields and 'example_name' in expected_fields and 'summary' in expected_fields:
            try:
                # Extract example_name and summary from conversational text
                example_name_match = re.search(r'example_name[:\s]*["\']([^"\']+)["\']', response_content, re.IGNORECASE)
                summary_match = re.search(r'summary[:\s]*["\']([^"\']+)["\']', response_content, re.IGNORECASE)
                
                if example_name_match and summary_match:
                    result = {
                        "example_name": example_name_match.group(1),
                        "summary": summary_match.group(1)
                    }
                    logger.debug("Successfully extracted JSON from conversational response")
                    return result
            except Exception:
                pass
        
        # Last resort - try to clean up the response
        cleaned = response_content.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()
        
        try:
            result = json.loads(cleaned)
            logger.debug("Successfully parsed cleaned response")
            return result
        except json.JSONDecodeError:
            logger.error(
                f"Failed to parse JSON from {provider_name}. "
                f"Response preview: {response_content[:200]}..."
            )
            raise


def create_llm_completion(
    prompt: str,
    system_prompt: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    response_format: dict[str, str] | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """
    Create a completion using the configured LLM provider with proper error handling.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system prompt
        model: Optional model override (uses MODEL_CHOICE if not specified)
        provider: Optional provider override
        response_format: Optional response format (e.g., {"type": "json_object"})
        temperature: Temperature for sampling
        max_tokens: Maximum tokens in response
        
    Returns:
        The completion text from the LLM
        
    Raises:
        Exception: If LLM call fails
    """
    try:
        client = get_sync_llm_client(provider=provider)
        
        if model is None:
            model = _get_model_choice()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
            
        # Handle response format with robust Ollama support
        if response_format and response_format.get("type") == "json_object":
            # Detect if this is Ollama by checking base_url
            is_ollama = False
            if hasattr(client, "_base_url"):
                base_url_str = str(client._base_url).lower()
                is_ollama = "ollama" in base_url_str or ":11434" in base_url_str
            
            if is_ollama:
                # For Ollama, use prompt-based JSON instruction instead of response_format
                # Some Ollama versions may not fully support the response_format parameter
                json_instruction = (
                    "\n\nIMPORTANT: Respond with ONLY valid JSON in the exact format requested. "
                    "Do not include explanations, comments, or additional text."
                )
                messages[-1]["content"] += json_instruction
                
                # Set temperature to 0 for more consistent outputs
                kwargs["temperature"] = 0
                logger.debug("Using prompt-based JSON instruction for Ollama compatibility")
            else:
                # Use standard response_format for OpenAI/Google
                kwargs["response_format"] = response_format
        
        # Enhanced logging for provider-specific monitoring
        is_ollama_request = hasattr(client, "_base_url") and (
            "ollama" in str(client._base_url).lower() or ":11434" in str(client._base_url).lower()
        )
        is_gemini_request = hasattr(client, "_base_url") and (
            "googleapis.com" in str(client._base_url).lower()
        )
        
        if is_ollama_request:
            logger.debug(f"🦙 [OLLAMA] Creating completion with model: {model}")
            logger.debug(f"🦙 [OLLAMA] Base URL: {getattr(client, '_base_url', 'Unknown')}")
            logger.debug(f"🦙 [OLLAMA] Temperature: {kwargs.get('temperature', temperature)}")
            logger.debug(f"🦙 [OLLAMA] Max tokens: {kwargs.get('max_tokens', 'None')}")
            logger.debug(f"🦙 [OLLAMA] Response format: {kwargs.get('response_format', 'None')}")
            logger.debug(f"🦙 [OLLAMA] Message count: {len(messages)}")
            
            # Log prompt preview for JSON requests at info level (important for troubleshooting)
            if response_format and response_format.get("type") == "json_object":
                user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                logger.info(f"🦙 [OLLAMA] JSON Request - Prompt preview: {user_msg[:150]}...")
        elif is_gemini_request:
            logger.debug(f"🤖 [GEMINI] Creating completion with model: {model}")
            logger.debug(f"🤖 [GEMINI] Base URL: {getattr(client, '_base_url', 'Unknown')}")
            logger.debug(f"🤖 [GEMINI] Temperature: {kwargs.get('temperature', temperature)}")
            logger.debug(f"🤖 [GEMINI] Response format: {kwargs.get('response_format', 'None')}")
            
            # Log prompt preview for JSON requests
            if response_format and response_format.get("type") == "json_object":
                user_msg = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
                logger.info(f"🤖 [GEMINI] JSON Request - Prompt preview: {user_msg[:150]}...")
        else:
            logger.debug(f"Creating completion with model: {model}, provider: {provider or 'default'}")
            
        response = client.chat.completions.create(**kwargs)
        
        # Log response details for specific providers
        if is_ollama_request:
            content_preview = ""
            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.debug(f"🦙 [OLLAMA] Response received - Length: {len(content) if content else 0} chars")
                logger.debug(f"🦙 [OLLAMA] Response preview: {repr(content_preview)}")
                
                # Special logging for JSON responses
                if response_format and response_format.get("type") == "json_object":
                    try:
                        import json
                        json.loads(content)
                        logger.info(f"🦙 [OLLAMA] ✅ Valid JSON response received")
                    except json.JSONDecodeError:
                        logger.warning(f"🦙 [OLLAMA] ⚠️ Invalid JSON response - will attempt parsing")
            else:
                logger.error(f"🦙 [OLLAMA] ❌ Empty or invalid response structure")
        elif is_gemini_request:
            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    content_preview = content[:100] + "..." if len(content) > 100 else content
                logger.debug(f"🤖 [GEMINI] Response received - Length: {len(content) if content else 0} chars")
                logger.debug(f"🤖 [GEMINI] Response preview: {repr(content_preview)}")
                
                # Special logging for JSON responses
                if response_format and response_format.get("type") == "json_object":
                    try:
                        import json
                        json.loads(content)
                        logger.info(f"🤖 [GEMINI] ✅ Valid JSON response received")
                    except json.JSONDecodeError:
                        logger.warning(f"🤖 [GEMINI] ⚠️ Invalid JSON response - will attempt parsing")
            else:
                logger.error(f"🤖 [GEMINI] ❌ Empty or invalid response structure")
        
        if not response or not response.choices:
            raise ValueError("Empty response from LLM")
            
        content = response.choices[0].message.content
        if content is None or content.strip() == "":
            raise ValueError("LLM returned empty or None content")
            
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error creating LLM completion: {e}", exc_info=True)
        raise