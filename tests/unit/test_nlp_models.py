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


def test_ner_model_adjacent_and_repeated_entities():
    """Verify NER does not merge distinct adjacent entities or drop repeated entities."""
    meta = ModelMetadata(model_id="test-ner", name="Test NER", task="ner")
    text = "Alice visited Bob in Paris and London. Later Alice returned to Paris."

    def mock_pipeline(inp: str):
        return [
            {"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 0, "end": 5},
            {"entity_group": "PER", "score": 0.98, "word": "Bob", "start": 14, "end": 17},
            {"entity_group": "LOC", "score": 0.97, "word": "Paris", "start": 21, "end": 26},
            {"entity_group": "LOC", "score": 0.96, "word": "London", "start": 31, "end": 37},
            {"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 45, "end": 50},
            {"entity_group": "LOC", "score": 0.97, "word": "Paris", "start": 63, "end": 68},
        ]

    model = NERModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference(text)
    entities = result["entities"]
    assert len(entities) == 6
    # Verify exact offset slicing matches entity text
    for ent in entities:
        assert text[ent["start_offset"]:ent["end_offset"]] == ent["text"]

    # Verify both occurrences of Alice and Paris are distinct with non-overlapping offsets
    alice_ents = [e for e in entities if e["text"] == "Alice"]
    assert len(alice_ents) == 2
    assert alice_ents[0]["start_offset"] == 0
    assert alice_ents[1]["start_offset"] == 45


def test_ner_model_subword_and_offset_boundary_alignment():
    """Verify subword token spans with whitespace boundary artifacts align cleanly with input text."""
    meta = ModelMetadata(model_id="test-ner", name="Test NER", task="ner")
    text = "We love HuggingFace tools."

    def mock_pipeline(inp: str):
        # Subwords aggregated into single span with clean character boundaries
        return [
            {"entity_group": "ORG", "score": 0.9542, "word": " HuggingFace ", "start": 8, "end": 19},
        ]

    model = NERModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference(text)
    assert len(result["entities"]) == 1
    ent = result["entities"][0]
    assert ent["text"] == "HuggingFace"
    assert ent["start_offset"] == 8
    assert ent["end_offset"] == 19
    assert text[ent["start_offset"]:ent["end_offset"]] == "HuggingFace"


def test_ner_model_inverted_or_out_of_bounds_offsets():
    """Verify NER guards against inverted or negative offsets safely."""
    meta = ModelMetadata(model_id="test-ner", name="Test NER", task="ner")
    text = "Sample sentence."

    def mock_pipeline(inp: str):
        return [
            {"entity_group": "MISC", "score": 0.85, "word": "Sample", "start": 6, "end": 0},
        ]

    model = NERModel(meta, pipeline_factory=lambda: mock_pipeline)
    model._load_weights("cpu")

    result = model._execute_inference(text)
    ent = result["entities"][0]
    assert ent["start_offset"] <= ent["end_offset"]
    assert ent["text"] == "Sample"


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


def test_zero_shot_model_preserves_pipeline_probabilities_no_double_transform():
    """Verify single-label softmax and multi-label independent sigmoid probabilities are preserved exactly."""
    meta = ModelMetadata(model_id="test-zero-shot", name="Test Zero Shot", task="zero-shot")

    # Single-label pipeline output (softmax sum = 1.0)
    single_label_pipeline = lambda t, candidate_labels, multi_label: {
        "sequence": t,
        "labels": ["urgent", "low", "medium"],
        "scores": [0.7000, 0.2000, 0.1000],
    }
    model_single = ZeroShotClassificationModel(meta, pipeline_factory=lambda: single_label_pipeline)
    model_single._load_weights("cpu")

    res_single = model_single._execute_inference({
        "text": "Server outage in production",
        "candidate_labels": ["urgent", "low", "medium"],
        "multi_label": False,
    })
    # Scores must remain exactly 0.7000, 0.2000, 0.1000 without second softmax
    assert res_single["top_label"] == "urgent"
    assert res_single["top_score"] == 0.7000
    assert sum(p["score"] for p in res_single["predictions"]) == 1.0

    # Multi-label pipeline output (independent sigmoid probabilities)
    multi_label_pipeline = lambda t, candidate_labels, multi_label: {
        "sequence": t,
        "labels": ["security", "infrastructure", "finance"],
        "scores": [0.9412, 0.8931, 0.0521],
    }
    model_multi = ZeroShotClassificationModel(meta, pipeline_factory=lambda: multi_label_pipeline)
    model_multi._load_weights("cpu")

    res_multi = model_multi._execute_inference({
        "text": "Critical zero-day vulnerability patched in kubernetes cluster",
        "candidate_labels": ["security", "infrastructure", "finance"],
        "multi_label": True,
    })
    # Must preserve independent sigmoid probabilities without distortion
    assert res_multi["predictions"][0] == {"label": "security", "score": 0.9412}
    assert res_multi["predictions"][1] == {"label": "infrastructure", "score": 0.8931}
    assert res_multi["predictions"][2] == {"label": "finance", "score": 0.0521}


def test_zero_shot_model_sorting_guarantee():
    """Verify predictions are sorted descending by score even if raw pipeline emitted unsorted scores."""
    meta = ModelMetadata(model_id="test-zero-shot", name="Test Zero Shot", task="zero-shot")

    # Unsorted mock pipeline output
    unsorted_pipeline = lambda t, candidate_labels, multi_label: {
        "sequence": t,
        "labels": ["biology", "quantum physics", "astronomy"],
        "scores": [0.1500, 0.8200, 0.4500],
    }
    model = ZeroShotClassificationModel(meta, pipeline_factory=lambda: unsorted_pipeline)
    model._load_weights("cpu")

    res = model._execute_inference({
        "text": "Superconducting qubits demonstrated quantum entanglement.",
        "candidate_labels": ["biology", "quantum physics", "astronomy"],
        "multi_label": False,
    })

    assert res["top_label"] == "quantum physics"
    assert res["top_score"] == 0.8200
    assert [p["label"] for p in res["predictions"]] == ["quantum physics", "astronomy", "biology"]
    assert [p["score"] for p in res["predictions"]] == [0.8200, 0.4500, 0.1500]
