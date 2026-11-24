"""Unit tests for M4 concrete NLP models: Sentiment, NER, and Zero-Shot Classification."""

import pytest

from src.core.models.base import ModelMetadata
from src.core.models.nlp import NERModel, SentimentModel, ZeroShotClassificationModel


# -----------------------------------------------------------------------------
# Sentiment Model Tests
# -----------------------------------------------------------------------------

def test_sentiment_model_inference_positive():
    """Verify sentiment analysis correctly formats top label, confidence, and distribution."""
    meta = ModelMetadata(model_id="test-sentiment", name="Test Sentiment", task="sentiment")

    def mock_pipeline(text: str):
        return [
            {"label": "POSITIVE", "score": 0.9854},
            {"label": "NEGATIVE", "score": 0.0146},
        ]

    model = SentimentModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference("I love building resilient software.")
    assert result["label"] == "POSITIVE"
    assert result["confidence"] == 0.9854
    assert result["scores"]["POSITIVE"] == 0.9854
    assert result["scores"]["NEGATIVE"] == 0.0146

    model._unload_weights()
    assert model._pipeline is None


def test_sentiment_model_inference_negative():
    """Verify negative sentiment analysis output."""
    meta = ModelMetadata(model_id="test-sentiment", name="Test Sentiment", task="sentiment")

    def mock_pipeline(text: str):
        return [
            {"label": "NEGATIVE", "score": 0.9512},
            {"label": "POSITIVE", "score": 0.0488},
        ]

    model = SentimentModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference("This latency is completely unacceptable.")
    assert result["label"] == "NEGATIVE"
    assert result["confidence"] == 0.9512
    assert result["scores"]["NEGATIVE"] == 0.9512


def test_sentiment_model_invalid_input_type():
    """Verify sentiment model rejects non-string input."""
    meta = ModelMetadata(model_id="test-sentiment", name="Test Sentiment", task="sentiment")
    model = SentimentModel(meta, pipeline_factory=lambda: lambda x: [])
    model._load_weights("cpu")

    with pytest.raises(ValueError) as exc:
        model._execute_inference(12345)
    assert "Expected string input" in str(exc.value)


# -----------------------------------------------------------------------------
# NER Model Tests
# -----------------------------------------------------------------------------

def test_ner_model_entity_extraction():
    """Verify NER extracts entity spans with character offsets and classifications."""
    meta = ModelMetadata(model_id="test-ner", name="Test NER", task="ner")

    def mock_pipeline(text: str):
        return [
            {"entity_group": "PER", "score": 0.9921, "word": "Sundar Pichai", "start": 0, "end": 13},
            {"entity_group": "ORG", "score": 0.9845, "word": "Google", "start": 27, "end": 33},
            {"entity_group": "LOC", "score": 0.9763, "word": "Mountain View", "start": 37, "end": 50},
        ]

    model = NERModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference("Sundar Pichai is the CEO of Google in Mountain View.")
    assert result["entity_count"] == 3
    entities = result["entities"]
    assert len(entities) == 3

    assert entities[0] == {
        "entity": "PER",
        "text": "Sundar Pichai",
        "confidence": 0.9921,
        "start_offset": 0,
        "end_offset": 13,
    }
    assert entities[1]["entity"] == "ORG"
    assert entities[2]["entity"] == "LOC"


def test_ner_model_entity_free_text():
    """Verify NER handles text with no named entities gracefully."""
    meta = ModelMetadata(model_id="test-ner", name="Test NER", task="ner")

    def mock_pipeline(text: str):
        return []

    model = NERModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference("The quick brown fox jumps over the lazy dog.")
    assert result["entity_count"] == 0
    assert result["entities"] == []


# -----------------------------------------------------------------------------
# Zero-Shot Classification Model Tests
# -----------------------------------------------------------------------------

def test_zero_shot_model_single_label():
    """Verify zero-shot classification in mutually exclusive single-label mode."""
    meta = ModelMetadata(model_id="test-zero-shot", name="Test Zero Shot", task="zero-shot")

    def mock_pipeline(text: str, candidate_labels: list, multi_label: bool):
        assert multi_label is False
        return {
            "sequence": text,
            "labels": ["finance", "technology", "sports"],
            "scores": [0.8251, 0.1245, 0.0504],
        }

    model = ZeroShotClassificationModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference({
        "text": "Quarterly earnings topped analyst expectations on cloud margins.",
        "candidate_labels": ["technology", "finance", "sports"],
        "multi_label": False,
    })

    assert result["top_label"] == "finance"
    assert result["top_score"] == 0.8251
    assert result["multi_label"] is False
    assert len(result["predictions"]) == 3
    assert result["predictions"][0]["label"] == "finance"
    assert result["predictions"][0]["score"] == 0.8251


def test_zero_shot_model_multi_label():
    """Verify zero-shot classification in independent multi-label mode."""
    meta = ModelMetadata(model_id="test-zero-shot", name="Test Zero Shot", task="zero-shot")

    def mock_pipeline(text: str, candidate_labels: list, multi_label: bool):
        assert multi_label is True
        return {
            "sequence": text,
            "labels": ["technology", "finance"],
            "scores": [0.9312, 0.8845],
        }

    model = ZeroShotClassificationModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference({
        "text": "Fintech startup raises series A to expand automated trading platform.",
        "candidate_labels": ["technology", "finance"],
        "multi_label": True,
    })

    assert result["multi_label"] is True
    assert result["top_label"] == "technology"
    assert len(result["predictions"]) == 2
    assert result["predictions"][0]["score"] == 0.9312
    assert result["predictions"][1]["score"] == 0.8845


def test_zero_shot_model_validation():
    """Verify zero-shot input validation for empty text and labels."""
    meta = ModelMetadata(model_id="test-zero-shot", name="Test Zero Shot", task="zero-shot")
    model = ZeroShotClassificationModel(meta, pipeline_factory=lambda: lambda *args, **kwargs: {})
    model._load_weights("cpu")

    with pytest.raises(ValueError):
        model._execute_inference("not a dict")

    with pytest.raises(ValueError):
        model._execute_inference({"text": "", "candidate_labels": ["news"]})

    with pytest.raises(ValueError):
        model._execute_inference({"text": "valid text", "candidate_labels": []})
