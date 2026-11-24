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
    await model.load()
    try:
        result = await model.predict("NLPulse provides incredible low-latency NLP APIs!")
        assert result["sentiment"] == "POSITIVE"
        assert result["confidence"] > 0.90
        assert "latency_ms" in result
    finally:
        await model.unload()


@pytest.mark.asyncio
async def test_live_ner_pipeline_smoke():
    model = NERModel()
    await model.load()
    try:
        result = await model.predict("Sundar Pichai leads Google in Mountain View, California.")
        entities = result["entities"]
        entity_texts = [e["text"] for e in entities]
        assert any("Sundar Pichai" in text for text in entity_texts)
        assert any("Google" in text for text in entity_texts)
        assert any("Mountain View" in text or "California" in text for text in entity_texts)
    finally:
        await model.unload()


@pytest.mark.asyncio
async def test_live_zero_shot_pipeline_smoke():
    model = ZeroShotClassificationModel()
    await model.load()
    try:
        result = await model.predict(
            text="SpaceX launched another Falcon 9 rocket carrying Starlink satellites into orbit.",
            candidate_labels=["aerospace", "sports", "cooking"],
            multi_label=False,
        )
        assert result["predicted_label"] == "aerospace"
        assert result["scores"][0]["label"] == "aerospace"
        assert result["scores"][0]["score"] > 0.80
    finally:
        await model.unload()
