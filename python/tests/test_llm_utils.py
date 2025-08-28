"""
Tests for LLM Utilities

Tests the unified LLM provider utilities that centralize OpenAI, Ollama, and Google Gemini support.
Focuses on core functionality: client creation, completion generation, and JSON parsing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from src.server.services.llm_utils import (
    _get_model_choice,
    create_llm_completion,
    get_sync_llm_client,
    parse_llm_json_response,
)


class TestModelChoice:
    """Test model choice selection logic"""

    @patch('src.server.services.llm_utils.credential_service')
    def test_model_choice_from_cache(self, mock_credential_service):
        """Test model choice retrieval from cache"""
        mock_credential_service._cache_initialized = True
        mock_credential_service._cache = {"MODEL_CHOICE": "gpt-4o"}
        
        result = _get_model_choice()
        assert result == "gpt-4o"

    @patch('src.server.services.llm_utils.credential_service')
    @patch('src.server.services.llm_utils.os.getenv')
    def test_model_choice_env_fallback(self, mock_getenv, mock_credential_service):
        """Test fallback to environment variable"""
        mock_credential_service._cache_initialized = False
        mock_getenv.return_value = "gpt-4o-mini"
        
        result = _get_model_choice()
        assert result == "gpt-4o-mini"


class TestLLMClient:
    """Test LLM client creation for different providers"""

    @patch('src.server.services.llm_utils.openai.OpenAI')
    @patch('src.server.services.llm_utils.os.getenv')
    def test_openai_client_creation(self, mock_getenv, mock_openai):
        """Test OpenAI client creation"""
        mock_getenv.return_value = "test-key"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = get_sync_llm_client(provider="openai")
        
        assert client == mock_client
        mock_openai.assert_called_once_with(api_key="test-key")

    @patch('src.server.services.llm_utils.openai.OpenAI')
    def test_ollama_client_creation(self, mock_openai):
        """Test Ollama client creation"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        with patch('src.server.services.llm_utils._get_cached_settings', return_value={"LLM_BASE_URL": "http://localhost:11434/v1"}):
            client = get_sync_llm_client(provider="ollama")
        
        assert client == mock_client
        mock_openai.assert_called_once_with(api_key="ollama", base_url="http://localhost:11434/v1")

    @patch('src.server.services.llm_utils.openai.OpenAI')
    @patch('src.server.services.llm_utils.os.getenv')
    def test_google_client_creation(self, mock_getenv, mock_openai):
        """Test Google Gemini client creation"""
        mock_getenv.return_value = "test-google-key"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        client = get_sync_llm_client(provider="google")
        
        assert client == mock_client
        mock_openai.assert_called_once_with(
            api_key="test-google-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )

    def test_unsupported_provider_error(self):
        """Test error with unsupported provider"""
        with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
            get_sync_llm_client(provider="invalid")


class TestJSONParsing:
    """Test JSON response parsing"""

    def test_valid_json_parsing(self):
        """Test parsing valid JSON response"""
        response = '{"name": "Test", "summary": "A test"}'
        result = parse_llm_json_response(response)
        
        assert result["name"] == "Test"
        assert result["summary"] == "A test"

    def test_markdown_json_extraction(self):
        """Test extracting JSON from markdown code blocks"""
        response = '''```json
        {"name": "Markdown Test", "summary": "JSON in markdown"}
        ```'''
        result = parse_llm_json_response(response)
        
        assert result["name"] == "Markdown Test"
        assert result["summary"] == "JSON in markdown"

    def test_empty_response_error(self):
        """Test error handling for empty response"""
        with pytest.raises(ValueError, match="Empty response from LLM"):
            parse_llm_json_response("")


class TestLLMCompletion:
    """Test LLM completion generation"""

    @patch('src.server.services.llm_utils.get_sync_llm_client')
    @patch('src.server.services.llm_utils._get_model_choice')
    def test_basic_completion(self, mock_get_model, mock_get_client):
        """Test basic completion generation"""
        mock_get_model.return_value = "gpt-4o"
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_client.chat.completions.create.return_value = mock_response
        
        result = create_llm_completion(prompt="Test prompt")
        
        assert result == "Test response"
        mock_client.chat.completions.create.assert_called_once()

    @patch('src.server.services.llm_utils.get_sync_llm_client')
    @patch('src.server.services.llm_utils._get_model_choice')
    def test_ollama_json_handling(self, mock_get_model, mock_get_client):
        """Test Ollama-specific JSON handling (uses prompt instructions instead of response_format)"""
        mock_get_model.return_value = "gpt-oss"
        mock_client = MagicMock()
        mock_client._base_url = "http://localhost:11434/v1"
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "success"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        result = create_llm_completion(
            prompt="Generate JSON",
            response_format={"type": "json_object"}
        )
        
        assert result == '{"result": "success"}'
        
        # Verify Ollama uses prompt-based JSON instructions
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["temperature"] == 0
        assert "IMPORTANT: Respond with ONLY valid JSON" in call_args["messages"][0]["content"]
        assert "response_format" not in call_args

    @patch('src.server.services.llm_utils.get_sync_llm_client')
    def test_empty_response_error(self, mock_get_client):
        """Test error handling for empty LLM response"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.chat.completions.create.return_value = mock_response
        
        with pytest.raises(ValueError, match="Empty response from LLM"):
            create_llm_completion(prompt="Test prompt")


class TestIntegration:
    """Integration test for the complete workflow"""

    @patch('src.server.services.llm_utils.os.getenv')
    @patch('src.server.services.llm_utils.openai.OpenAI')
    def test_end_to_end_workflow(self, mock_openai, mock_getenv):
        """Test complete workflow from client creation to JSON parsing"""
        mock_getenv.return_value = "test-key"
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"name": "Test", "summary": "Success"}'
        mock_client.chat.completions.create.return_value = mock_response
        
        with patch('src.server.services.llm_utils._get_model_choice', return_value="gpt-4o"):
            result = create_llm_completion(
                prompt="Analyze this",
                provider="openai",
                response_format={"type": "json_object"}
            )
        
        # Test the complete workflow
        assert result == '{"name": "Test", "summary": "Success"}'
        parsed = parse_llm_json_response(result)
        assert parsed["name"] == "Test"
        assert parsed["summary"] == "Success"