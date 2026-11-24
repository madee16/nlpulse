"""Unit tests for M5 models: EmbeddingModel and SummarizationModel."""

import numpy as np
import pytest

from src.core.models.base import ModelMetadata
from src.core.models.nlp import EmbeddingModel, SummarizationModel


# -----------------------------------------------------------------------------
# Embedding Model Tests
# -----------------------------------------------------------------------------

def test_embedding_model_inference_vector_normalization():
    """Verify EmbeddingModel produces unit-normalized L2 dense embeddings."""
    meta = ModelMetadata(model_id="test-emb", name="Test Embedding", task="embedding")

    def mock_embedding_fn(texts: list):
        # Generate predictable vectors with non-unit norm, then unit normalize
        raw = np.array([[float(len(t)), float(i + 1)] for i, t in enumerate(texts)], dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / np.maximum(norms, 1e-9)

    model = EmbeddingModel(meta, pipeline_factory=lambda: mock_embedding_fn)
    model._load_weights("cpu")

    texts = ["FastAPI microservices", "PostgreSQL database indexing", "Docker containerization"]
    embs = model._execute_inference(texts)

    assert isinstance(embs, np.ndarray)
    assert embs.shape == (3, 2)
    # Verify L2 normalization
    norms = np.linalg.norm(embs, axis=1)
    for n in norms:
        assert pytest.approx(n, abs=1e-4) == 1.0

    model._unload_weights()
    assert model._model is None


def test_embedding_model_single_string_and_empty_list():
    """Verify EmbeddingModel handles single string input and empty list."""
    meta = ModelMetadata(model_id="test-emb", name="Test Embedding", task="embedding")

    def mock_fn(texts: list):
        return np.ones((len(texts), 384), dtype=np.float32)

    model = EmbeddingModel(meta, pipeline_factory=lambda: mock_fn)
    model._load_weights("cpu")

    # Single string input
    single_res = model._execute_inference("Single string input")
    assert single_res.shape == (1, 384)

    # Empty list input
    empty_res = model._execute_inference([])
    assert empty_res.shape == (0, 384)

    # Invalid input type
    with pytest.raises(ValueError):
        model._execute_inference(12345)


# -----------------------------------------------------------------------------
# Summarization Model Tests
# -----------------------------------------------------------------------------

def test_summarization_model_inference_success():
    """Verify SummarizationModel executes abstractive generation with length parameters."""
    meta = ModelMetadata(model_id="test-sum", name="Test Summarization", task="summarization")

    def mock_pipe(text: str, **kwargs):
        assert kwargs.get("max_length") == 100
        assert kwargs.get("min_length") == 20
        return [{"summary_text": "This is a concise summary of the input document."}]

    model = SummarizationModel(meta, pipeline_factory=lambda: mock_pipe)
    model._load_weights("cpu")

    result = model._execute_inference({
        "text": "Long document with many detailed paragraphs describing system design.",
        "max_length": 100,
        "min_length": 20,
    })

    assert result["summary_text"] == "This is a concise summary of the input document."

    model._unload_weights()
    assert model._pipeline is None


def test_summarization_model_validation_errors():
    """Verify SummarizationModel rejects invalid inputs."""
    meta = ModelMetadata(model_id="test-sum", name="Test Summarization", task="summarization")
    model = SummarizationModel(meta, pipeline_factory=lambda: lambda t, **kw: [])
    model._load_weights("cpu")

    # Non-dict
    with pytest.raises(ValueError):
        model._execute_inference("just text")

    # Empty text
    with pytest.raises(ValueError):
        model._execute_inference({"text": "   "})

    # min_length > max_length
    with pytest.raises(ValueError) as exc:
        model._execute_inference({"text": "Valid text", "min_length": 150, "max_length": 50})
    assert "cannot exceed" in str(exc.value)
