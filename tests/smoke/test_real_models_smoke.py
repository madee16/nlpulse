"""Opt-in live model smoke tests for NLPulse M4 pipelines.

These tests download and execute actual Hugging Face model weights:
- distilbert/distilbert-base-uncased-finetuned-sst-2-english
- dslim/bert-base-NER
- valhalla/distilbart-mnli-12-1

They are strictly gated behind:
  pytest -m smoke
  RUN_REAL_MODEL_SMOKE=1
"""

import os
import pytest
from src.core.models.nlp import SentimentModel, NERModel, ZeroShotClassificationModel


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        os.getenv("RUN_REAL_MODEL_SMOKE") != "1",
        reason="Skipping live model smoke tests: RUN_REAL_MODEL_SMOKE=1 not set",
    ),
]


@pytest.mark.asyncio
async def test_live_sentiment_pipeline_smoke():
    model = SentimentModel()
    await model.load("cpu")
    try:
        result = await model.predict("NLPulse provides incredible low-latency NLP APIs!")
        assert result["label"] == "POSITIVE"
        assert result["confidence"] > 0.90
        assert "scores" in result
    finally:
        await model.unload()


@pytest.mark.asyncio
async def test_live_ner_pipeline_smoke():
    model = NERModel()
    await model.load("cpu")
    try:
        text = "Sundar Pichai leads Google in Mountain View, California."
        result = await model.predict(text)
        entities = result["entities"]
        assert len(entities) > 0
        entity_texts = [e["text"] for e in entities]
        assert any("Sundar Pichai" in t for t in entity_texts)
        assert any("Google" in t for t in entity_texts)
        assert any("Mountain View" in t or "California" in t for t in entity_texts)
        for e in entities:
            assert 0 <= e["start_offset"] <= e["end_offset"] <= len(text)
            assert text[e["start_offset"]:e["end_offset"]] == e["text"]
    finally:
        await model.unload()


@pytest.mark.asyncio
async def test_live_zero_shot_pipeline_smoke():
    model = ZeroShotClassificationModel()
    await model.load("cpu")
    try:
        result = await model.predict({
            "text": "SpaceX launched another Falcon 9 rocket carrying Starlink satellites into orbit.",
            "candidate_labels": ["aerospace", "sports", "cooking"],
            "multi_label": False,
        })
        assert result["top_label"] == "aerospace"
        assert result["predictions"][0]["label"] == "aerospace"
        assert result["predictions"][0]["score"] > 0.80
    finally:
        await model.unload()
