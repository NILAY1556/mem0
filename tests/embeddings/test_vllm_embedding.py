from unittest.mock import Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.vllm import VllmEmbedding


@pytest.fixture
def mock_vllm_client():
    with patch("mem0.embeddings.vllm.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client


def test_vllm_embedding_initialization():
    config = BaseEmbedderConfig(
        model="intfloat/e5-mistral-7b-instruct",
        vllm_base_url="http://localhost:8001/v1"
    )
    embedder = VllmEmbedding(config)
    
    assert embedder.config.model == "intfloat/e5-mistral-7b-instruct"
    assert embedder.config.embedding_dims == 4096


def test_vllm_embedding_default_config():
    embedder = VllmEmbedding()
    
    assert embedder.config.model == "intfloat/e5-mistral-7b-instruct"
    assert embedder.config.embedding_dims == 4096


def test_vllm_embedding_embed(mock_vllm_client):
    config = BaseEmbedderConfig(
        model="intfloat/e5-mistral-7b-instruct",
        vllm_base_url="http://localhost:8001/v1"
    )
    embedder = VllmEmbedding(config)
    
    # Mock the response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
    mock_vllm_client.embeddings.create.return_value = mock_response
    
    # Test embedding
    text = "Hello, world!"
    result = embedder.embed(text)
    
    # Verify the call
    mock_vllm_client.embeddings.create.assert_called_once_with(
        input=["Hello, world!"],
        model="intfloat/e5-mistral-7b-instruct"
    )
    
    # Verify the result
    assert result == [0.1, 0.2, 0.3, 0.4]


def test_vllm_embedding_text_preprocessing(mock_vllm_client):
    config = BaseEmbedderConfig(model="intfloat/e5-mistral-7b-instruct")
    embedder = VllmEmbedding(config)
    
    # Mock the response
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3, 0.4])]
    mock_vllm_client.embeddings.create.return_value = mock_response
    
    # Test with text containing newlines
    text = "Hello,\nworld!"
    embedder.embed(text)
    
    # Verify newlines are replaced with spaces
    mock_vllm_client.embeddings.create.assert_called_once_with(
        input=["Hello, world!"],
        model="intfloat/e5-mistral-7b-instruct"
    )
