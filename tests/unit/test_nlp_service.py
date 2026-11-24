"""Unit tests for NLPService validation, orchestration, and usage metering."""

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models.base import ModelMetadata
from src.core.models.nlp import NERModel, SentimentModel, ZeroShotClassificationModel
from src.services.model_registry import ModelRegistry
from src.services.nlp_service import NLPService, NLPValidationError


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_registry() -> ModelRegistry:
    reg = ModelRegistry()

    # Register sentiment
    sm_meta = ModelMetadata(model_id="distilbert/distilbert-base-uncased-finetuned-sst-2-english", name="S", task="sentiment")
    sm = SentimentModel(sm_meta, pipeline_factory=lambda: lambda x: [{"label": "POSITIVE", "score": 0.99}])
    reg.register(sm_meta, lambda: sm)

    # Register NER
    ner_meta = ModelMetadata(model_id="dslim/bert-base-NER", name="N", task="ner")
    ner = NERModel(ner_meta, pipeline_factory=lambda: lambda x: [{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 0, "end": 5}])
    reg.register(ner_meta, lambda: ner)

    # Register Zero-shot
    zs_meta = ModelMetadata(model_id="valhalla/distilbart-mnli-12-1", name="Z", task="zero-shot")
    zs = ZeroShotClassificationModel(zs_meta, pipeline_factory=lambda: lambda text, candidate_labels, multi_label: {"labels": candidate_labels, "scores": [0.8] * len(candidate_labels)})
    reg.register(zs_meta, lambda: zs)

    return reg


@pytest.mark.asyncio
async def test_sentiment_service_success_and_metering(mock_session: AsyncMock, mock_registry: ModelRegistry):
    """Verify sentiment service executes model and accurately records usage."""
    service = NLPService(session=mock_session, registry=mock_registry)
    dev_id = uuid.uuid4()
    key_id = uuid.uuid4()

    res = await service.analyze_sentiment("Great product!", developer_id=dev_id, api_key_id=key_id)

    assert res["label"] == "POSITIVE"
    assert res["confidence"] == 0.99
    assert res["latency_ms"] >= 0.0
    assert "model_id" in res

    # Verify usage record was created
    assert mock_session.add.called
    assert mock_session.commit.called
    created_record = mock_session.add.call_args[0][0]
    assert created_record.developer_id == dev_id
    assert created_record.api_key_id == key_id
    assert created_record.status_code == 200
    assert created_record.endpoint == "/api/v1/nlp/sentiment"


@pytest.mark.asyncio
async def test_sentiment_service_rejects_empty_text(mock_session: AsyncMock, mock_registry: ModelRegistry):
    """Verify empty or whitespace-only text raises NLPValidationError without recording usage."""
    service = NLPService(session=mock_session, registry=mock_registry)
    dev_id = uuid.uuid4()

    with pytest.raises(NLPValidationError) as exc:
        await service.analyze_sentiment("   ", developer_id=dev_id)

    assert "empty" in str(exc.value.message).lower()
    # Usage must NOT be charged
    assert not mock_session.add.called


@pytest.mark.asyncio
async def test_ner_service_success_and_metering(mock_session: AsyncMock, mock_registry: ModelRegistry):
    """Verify NER service extracts entities and records usage."""
    service = NLPService(session=mock_session, registry=mock_registry)
    dev_id = uuid.uuid4()

    res = await service.extract_entities("Alice visited Paris.", developer_id=dev_id)

    assert res["entity_count"] == 1
    assert res["entities"][0]["text"] == "Alice"
    assert mock_session.add.called


@pytest.mark.asyncio
async def test_zero_shot_service_success_and_metering(mock_session: AsyncMock, mock_registry: ModelRegistry):
    """Verify zero-shot service validates labels, executes, and meters usage."""
    service = NLPService(session=mock_session, registry=mock_registry)
    dev_id = uuid.uuid4()

    res = await service.classify_zero_shot(
        text="The team won the championship game.",
        candidate_labels=["sports", "business", "sports"],  # contains duplicate
        multi_label=False,
        developer_id=dev_id,
    )

    # Duplicates deduplicated
    assert len(res["predictions"]) == 2
    assert mock_session.add.called


@pytest.mark.asyncio
async def test_zero_shot_service_rejects_empty_labels(mock_session: AsyncMock, mock_registry: ModelRegistry):
    """Verify zero-shot service rejects empty candidate labels."""
    service = NLPService(session=mock_session, registry=mock_registry)
    dev_id = uuid.uuid4()

    with pytest.raises(NLPValidationError):
        await service.classify_zero_shot("Some text", candidate_labels=[], multi_label=False, developer_id=dev_id)

    with pytest.raises(NLPValidationError):
        await service.classify_zero_shot("Some text", candidate_labels=["  "], multi_label=False, developer_id=dev_id)

    assert not mock_session.add.called
