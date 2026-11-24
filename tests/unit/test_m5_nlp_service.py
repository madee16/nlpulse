"""Unit tests for M5 NLPService keyword extraction and text summarization."""

import uuid
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.config import get_settings
from src.core.models.base import ModelMetadata
from src.core.models.nlp import EmbeddingModel, SummarizationModel
from src.services.model_registry import ModelRegistry
from src.services.nlp_service import NLPService, NLPValidationError

settings = get_settings()


@pytest.fixture
def mock_session():
    """Mock async SQLAlchemy database session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def m5_test_registry():
    """Provide a mock model registry containing M5 test models."""
    reg = ModelRegistry()

    # Embedding mock
    emb_meta = ModelMetadata(
        model_id=settings.MODEL_EMBEDDING_ID,
        name="Test Embedding",
        task="embedding",
    )
    def mock_emb(texts: list):
        # Generate predictable distinct embeddings
        res = []
        for t in texts:
            if "database" in t.lower() or "sql" in t.lower():
                vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
            elif "fastapi" in t.lower() or "api" in t.lower():
                vec = np.array([0.1, 0.9, 0.0], dtype=np.float32)
            else:
                vec = np.array([0.3, 0.3, 0.3], dtype=np.float32)
            res.append(vec / np.linalg.norm(vec))
        return np.array(res, dtype=np.float32)

    reg.register(emb_meta, lambda: EmbeddingModel(emb_meta, pipeline_factory=lambda: mock_emb))

    # Summarization mock
    sum_meta = ModelMetadata(
        model_id=settings.MODEL_SUMMARIZATION_ID,
        name="Test Summarization",
        task="summarization",
    )
    def mock_sum(text: str, **kwargs):
        return [{"summary_text": f"Summary: {text[:40]}..."}]

    reg.register(sum_meta, lambda: SummarizationModel(sum_meta, pipeline_factory=lambda: mock_sum))

    return reg


# -----------------------------------------------------------------------------
# Keyword Extraction Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_keyword_extraction_ranking_and_metering(mock_session, m5_test_registry):
    """Verify keywords are extracted, ranked by cosine similarity, and usage is recorded."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()
    key_id = uuid.uuid4()

    text = "FastAPI provides asynchronous endpoints and high performance APIs for Python developers."
    result = await service.extract_keywords(
        text=text,
        developer_id=dev_id,
        api_key_id=key_id,
        max_keywords=3,
        ngram_range=(1, 2),
    )

    assert result["model_id"] == settings.MODEL_EMBEDDING_ID
    assert result["total_candidates"] >= 3
    keywords = result["keywords"]
    assert len(keywords) == 3
    # Scores must be sorted descending
    assert keywords[0]["score"] >= keywords[1]["score"] >= keywords[2]["score"]
    # Check top keywords are relevant
    top_words = [k["keyword"].lower() for k in keywords]
    assert any("fastapi" in w or "apis" in w for w in top_words)

    # Verify atomic usage metering recorded
    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_keyword_extraction_with_candidate_filter(mock_session, m5_test_registry):
    """Verify explicit candidates are filtered and evaluated against document."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()

    text = "PostgreSQL handles complex SQL queries and transactional databases."
    candidates = ["database", "fastapi", "sql queries", "react"]

    result = await service.extract_keywords(
        text=text,
        developer_id=dev_id,
        candidates=candidates,
        max_keywords=2,
    )

    keywords = result["keywords"]
    assert len(keywords) <= 2
    top_kw = keywords[0]["keyword"].lower()
    # "database" should rank top given document topic
    assert "database" in top_kw or "sql" in top_kw


@pytest.mark.asyncio
async def test_keyword_extraction_stopword_only_returns_empty(mock_session, m5_test_registry):
    """Verify input with only stopwords gracefully returns empty keywords."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()

    text = "To be or not to be that is the."
    result = await service.extract_keywords(text=text, developer_id=dev_id)
    assert result["keywords"] == []
    assert result["total_candidates"] == 0


@pytest.mark.asyncio
async def test_keyword_extraction_validation_errors(mock_session, m5_test_registry):
    """Verify invalid parameters raise NLPValidationError."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()

    # Empty text
    with pytest.raises(NLPValidationError):
        await service.extract_keywords(text="", developer_id=dev_id)

    # Invalid max_keywords
    with pytest.raises(NLPValidationError):
        await service.extract_keywords(text="Valid text", developer_id=dev_id, max_keywords=0)

    # Invalid ngram_range
    with pytest.raises(NLPValidationError):
        await service.extract_keywords(text="Valid text", developer_id=dev_id, ngram_range=(3, 1))


# -----------------------------------------------------------------------------
# Text Summarization Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_summarize_single_chunk_success(mock_session, m5_test_registry):
    """Verify single-pass abstractive summarization generates summary and meters usage."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()
    key_id = uuid.uuid4()

    text = "PostgreSQL is a powerful, open-source object-relational database system with over 35 years of active development."
    result = await service.summarize_text(
        text=text,
        developer_id=dev_id,
        api_key_id=key_id,
        min_length=10,
        max_length=60,
    )

    assert result["model_id"] == settings.MODEL_SUMMARIZATION_ID
    assert result["chunks_processed"] == 1
    assert "Summary:" in result["summary"]
    assert result["original_character_count"] == len(text)
    assert result["summary_character_count"] > 0
    assert result["latency_ms"] >= 0.0

    mock_session.add.assert_called_once()
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_summarize_long_document_multi_chunk(mock_session, m5_test_registry, monkeypatch):
    """Verify long documents trigger chunking and multi-pass consolidation."""
    # Force small chunk size for testing chunking pipeline
    monkeypatch.setattr(settings, "SUMMARIZATION_CHUNK_SIZE", 200)

    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()

    # Create a long text spanning multiple sentences
    paragraphs = [
        f"Section {i} explains comprehensive architectural principles of distributed ledger storage and replication."
        for i in range(10)
    ]
    long_text = " ".join(paragraphs)

    result = await service.summarize_text(
        text=long_text,
        developer_id=dev_id,
        min_length=15,
        max_length=80,
    )

    assert result["chunks_processed"] > 1
    assert "Summary:" in result["summary"]
    # Usage record cost units should match chunks_processed
    added_record = mock_session.add.call_args[0][0]
    assert added_record.cost_units == result["chunks_processed"]


@pytest.mark.asyncio
async def test_summarize_validation_errors(mock_session, m5_test_registry):
    """Verify summarization input validation catches empty text, invalid lengths, and temperatures."""
    service = NLPService(session=mock_session, registry=m5_test_registry)
    dev_id = uuid.uuid4()

    # Empty text
    with pytest.raises(NLPValidationError):
        await service.summarize_text(text="", developer_id=dev_id)

    # min_length > max_length
    with pytest.raises(NLPValidationError):
        await service.summarize_text(text="Valid text", developer_id=dev_id, min_length=100, max_length=50)

    # Invalid temperature range when do_sample=True
    with pytest.raises(NLPValidationError, match="temperature must be between 0.0"):
        await service.summarize_text(text="Valid text", developer_id=dev_id, do_sample=True, temperature=3.5)

    with pytest.raises(NLPValidationError, match="temperature must be between 0.0"):
        await service.summarize_text(text="Valid text", developer_id=dev_id, do_sample=True, temperature=0.0)

    # temperature specified without do_sample=True
    with pytest.raises(NLPValidationError, match="temperature can only be specified when do_sample=True"):
        await service.summarize_text(text="Valid text", developer_id=dev_id, do_sample=False, temperature=0.7)


@pytest.mark.asyncio
async def test_keyword_extraction_deterministic_tie_breaking(mock_session, m5_test_registry):
    """Verify identical similarity scores are deterministically tie-broken alphabetically."""
    reg = ModelRegistry()
    emb_meta = ModelMetadata(
        model_id=settings.MODEL_EMBEDDING_ID,
        name="Test Embedding",
        task="embedding",
    )
    # Return identical vectors for all texts to produce identical similarity scores
    def uniform_emb(texts: list):
        return np.ones((len(texts), 3), dtype=np.float32) / np.sqrt(3.0)

    reg.register(emb_meta, lambda: EmbeddingModel(emb_meta, pipeline_factory=lambda: uniform_emb))
    service = NLPService(session=mock_session, registry=reg)
    dev_id = uuid.uuid4()

    result = await service.extract_keywords(
        text="zebra alpha beta dolphin",
        developer_id=dev_id,
        candidates=["zebra", "alpha", "beta", "dolphin"],
        max_keywords=4,
    )
    kw_names = [k["keyword"] for k in result["keywords"]]
    # Should be sorted alphabetically because all scores are identical
    assert kw_names == sorted(kw_names)
