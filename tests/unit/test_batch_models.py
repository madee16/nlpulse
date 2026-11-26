"""Unit tests for M7 model-level batched inference and M3 lifecycle parity."""

import asyncio
import time
from typing import Any, List
import pytest

from src.core.exceptions import ModelError, ModelInUseError, ModelInferenceError
from src.core.models.base import BaseModel, ModelMetadata, ModelState
from src.core.models.nlp import NERModel, SentimentModel, ZeroShotClassificationModel


class SlowBatchModel(BaseModel):
    """Test model verifying predict_batch shares M3 thread retention and timeout protection."""

    def __init__(self, metadata: ModelMetadata, delay: float = 0.0) -> None:
        super().__init__(metadata)
        self.delay = delay

    def _load_weights(self, device: str) -> None:
        pass

    def _unload_weights(self) -> None:
        pass

    def _execute_inference(self, inputs: Any, **kwargs: Any) -> Any:
        time.sleep(self.delay)
        return {"value": inputs}


@pytest.mark.asyncio
async def test_base_model_predict_batch_lifecycle_and_timeout():
    """Verify predict_batch preserves M3 active_inferences tracking and timeout safety."""
    meta = ModelMetadata(
        model_id="batch-base-test",
        name="Batch Base Test",
        task="test",
        max_concurrency=1,
        timeout_seconds=0.05,
    )
    model = SlowBatchModel(meta, delay=0.1)

    with pytest.raises(ModelError) as not_ready:
        await model.predict_batch(["a", "b"])
    assert not_ready.value.status_code == 503

    await model.load("cpu")

    # Timeout while worker thread continues running (2 items * 0.1s = 0.2s total)
    with pytest.raises(ModelInferenceError) as exc_info:
        await model.predict_batch(["a", "b"])
    assert "timed out" in str(exc_info.value.reason).lower()
    assert model.active_inferences == 1
    assert model.is_busy() is True

    with pytest.raises(ModelInUseError):
        await model.unload()

    # Wait for background worker thread to finish both items
    await asyncio.sleep(0.25)
    assert model.active_inferences == 0
    assert model.is_busy() is False
    await model.unload()
    assert model.state == ModelState.UNLOADED


@pytest.mark.asyncio
async def test_sentiment_model_predict_batch_uses_single_batched_pipeline_call():
    """Verify SentimentModel.predict_batch invokes HF-style pipeline once with full text list."""
    calls: List[Any] = []

    class FakeHFSentimentPipeline:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            if isinstance(inputs, list):
                return [
                    [{"label": "POSITIVE", "score": 0.95}, {"label": "NEGATIVE", "score": 0.05}]
                    if "good" in t
                    else [{"label": "NEGATIVE", "score": 0.91}, {"label": "POSITIVE", "score": 0.09}]
                    for t in inputs
                ]
            return [{"label": "POSITIVE", "score": 0.95}, {"label": "NEGATIVE", "score": 0.05}]

    meta = ModelMetadata(model_id="sent-batch", name="Sentiment Batch", task="sentiment")
    model = SentimentModel(meta, pipeline_factory=lambda: FakeHFSentimentPipeline())
    await model.load("cpu")

    results = await model.predict_batch(["good job", "bad luck", "good day"], batch_size=8)
    assert len(calls) == 1
    assert calls[0][0] == ["good job", "bad luck", "good day"]
    assert calls[0][1]["batch_size"] == 8
    assert calls[0][1]["truncation"] is True
    assert [r["label"] for r in results] == ["POSITIVE", "NEGATIVE", "POSITIVE"]
    assert model.total_inferences == 1


@pytest.mark.asyncio
async def test_sentiment_model_predict_batch_legacy_single_mock_fallback():
    """Verify SentimentModel.predict_batch remains compatible with legacy single-string mocks."""
    def legacy_mock(text: str):
        if "great" in text.lower():
            return [{"label": "POSITIVE", "score": 0.99}, {"label": "NEGATIVE", "score": 0.01}]
        return [{"label": "NEGATIVE", "score": 0.97}, {"label": "POSITIVE", "score": 0.03}]

    meta = ModelMetadata(model_id="sent-legacy", name="Sentiment Legacy", task="sentiment")
    model = SentimentModel(meta, pipeline_factory=lambda: legacy_mock)
    await model.load("cpu")

    results = await model.predict_batch(["great work", "terrible bug"])
    assert [r["label"] for r in results] == ["POSITIVE", "NEGATIVE"]


@pytest.mark.asyncio
async def test_ner_model_predict_batch_uses_single_batched_pipeline_call():
    """Verify NERModel.predict_batch passes list of texts in one batched pipeline call."""
    calls: List[Any] = []

    class FakeHFNERPipeline:
        tokenizer = object()
        model = object()

        def __call__(self, inputs, **kwargs):
            calls.append((inputs, kwargs))
            if isinstance(inputs, list):
                out = []
                for t in inputs:
                    if "Alice" in t:
                        out.append([{"entity_group": "PER", "score": 0.99, "word": "Alice", "start": 0, "end": 5}])
                    else:
                        out.append([])
                return out
            return []

    meta = ModelMetadata(model_id="ner-batch", name="NER Batch", task="ner")
    model = NERModel(meta, pipeline_factory=lambda: FakeHFNERPipeline())
    await model.load("cpu")

    results = await model.predict_batch(["Alice arrived.", "Nothing here."], batch_size=4)
    assert len(calls) == 1
    assert calls[0][1]["batch_size"] == 4
    assert results[0]["entity_count"] == 1
    assert results[0]["entities"][0]["text"] == "Alice"
    assert results[1]["entity_count"] == 0
    assert results[1]["entities"] == []


@pytest.mark.asyncio
async def test_zero_shot_model_predict_batch_uses_single_batched_pipeline_call():
    """Verify ZeroShotClassificationModel.predict_batch runs batched NLI classification."""
    calls: List[Any] = []

    class FakeHFZeroShotPipeline:
        tokenizer = object()
        model = object()

        def __call__(self, sequences, candidate_labels, multi_label=False, **kwargs):
            calls.append((sequences, candidate_labels, multi_label, kwargs))
            if isinstance(sequences, list):
                return [
                    {
                        "sequence": seq,
                        "labels": [candidate_labels[1], candidate_labels[0]],
                        "scores": [0.85, 0.15] if idx % 2 == 0 else [0.35, 0.65],
                    }
                    for idx, seq in enumerate(sequences)
                ]
            return {"sequence": sequences, "labels": candidate_labels, "scores": [0.8, 0.2]}

    meta = ModelMetadata(model_id="zs-batch", name="ZeroShot Batch", task="zero-shot")
    model = ZeroShotClassificationModel(meta, pipeline_factory=lambda: FakeHFZeroShotPipeline())
    await model.load("cpu")

    results = await model.predict_batch(
        {
            "texts": ["Text one", "Text two"],
            "candidate_labels": ["sports", "tech"],
            "multi_label": False,
        },
        batch_size=8,
    )
    assert len(calls) == 1
    assert calls[0][0] == ["Text one", "Text two"]
    assert calls[0][3]["batch_size"] == 8
    assert results[0]["top_label"] == "tech"
    assert results[1]["top_label"] == "sports"


@pytest.mark.asyncio
async def test_embedding_model_pytorch_tensor_batching_offline():
    """Verify EmbeddingModel executes batched PyTorch masked mean-pooling and L2 normalization."""
    import numpy as np
    import torch
    from src.core.models.nlp import EmbeddingModel

    class FakeTokenizer:
        model_max_length = 256

        def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            n = len(texts)
            return {
                "input_ids": torch.ones((n, 4), dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 0]] * n, dtype=torch.long),
            }

    class FakeEncoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask):
            batch_size, seq_len = input_ids.shape
            hidden = torch.arange(1, batch_size * seq_len * 6 + 1, dtype=torch.float32).reshape(batch_size, seq_len, 6)
            return (hidden,)

    meta = ModelMetadata(model_id="emb-tensor-test", name="Emb Tensor", task="embedding")
    model = EmbeddingModel(meta)
    model._tokenizer = FakeTokenizer()
    enc = FakeEncoder()
    enc.eval()
    model._model = enc
    model._device_str = "cpu"
    model._state = ModelState.READY

    vectors = await model.predict(["first doc", "second candidate", "third phrase"])
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (3, 6)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)

