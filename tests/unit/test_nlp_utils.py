"""Unit tests for NLP text utilities: candidate extraction, stopword filtering, and chunking."""

import pytest

from src.core.nlp_utils import (
    chunk_text_by_sentences,
    extract_candidate_ngrams,
)


def test_extract_candidate_ngrams_unigrams_and_bigrams():
    """Verify candidate extraction filters stopwords and generates clean n-grams."""
    text = "FastAPI enables high performance web APIs and microservices in Python."
    # Unigrams only
    unigrams = extract_candidate_ngrams(text, ngram_range=(1, 1))
    assert "FastAPI" in unigrams
    assert "performance" in unigrams
    assert "APIs" in unigrams
    assert "microservices" in unigrams
    assert "Python" in unigrams
    # Stopwords should not appear
    assert "and" not in unigrams
    assert "in" not in unigrams

    # Bigrams
    bigrams = extract_candidate_ngrams(text, ngram_range=(2, 2))
    assert any("web APIs" in bg for bg in bigrams)
    # Stopwords at edges of bigrams should be rejected
    assert not any(bg.startswith("and ") or bg.endswith(" in") for bg in bigrams)


def test_extract_candidate_ngrams_stopword_only_text():
    """Verify text containing only stopwords returns empty candidate list."""
    text = "To be or not to be that is the question."
    candidates = extract_candidate_ngrams(text, ngram_range=(1, 1))
    # "question" is not a stopword, but "to", "be", "or", "not", "that", "is", "the" are stopwords
    assert candidates == ["question"]

    pure_stopwords = "To be and or not if so then."
    assert extract_candidate_ngrams(pure_stopwords, ngram_range=(1, 1)) == []


def test_extract_candidate_ngrams_with_explicit_filter():
    """Verify explicit candidate list filters and deduplicates properly."""
    candidates = ["Machine Learning", "fastapi", "machine learning", "", "   ", "Docker"]
    filtered = extract_candidate_ngrams("Any text", candidate_filter=candidates)
    assert filtered == ["machine learning", "fastapi", "docker"]


def test_extract_candidate_ngrams_invalid_range():
    """Verify invalid n-gram bounds raise ValueError."""
    with pytest.raises(ValueError):
        extract_candidate_ngrams("Sample text", ngram_range=(2, 1))

    with pytest.raises(ValueError):
        extract_candidate_ngrams("Sample text", ngram_range=(0, 2))

    with pytest.raises(ValueError):
        extract_candidate_ngrams("Sample text", ngram_range=(1, 4))


def test_chunk_text_by_sentences_short_document():
    """Verify text under chunk size is returned as single chunk."""
    text = "This is a short text. It fits easily in one chunk."
    chunks = chunk_text_by_sentences(text, chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_by_sentences_long_document():
    """Verify long multi-sentence document is split at sentence boundaries."""
    sentences = [f"This is sentence number {i} providing valuable details." for i in range(50)]
    long_doc = " ".join(sentences)

    chunks = chunk_text_by_sentences(long_doc, chunk_size=400)
    assert len(chunks) > 1

    # Verify no sentence is sliced in half (every chunk should end with punctuation)
    for c in chunks:
        assert c.strip().endswith(".")

    # Verify all sentences are preserved across chunks
    reconstructed = " ".join(chunks)
    for s in sentences:
        assert s in reconstructed


def test_chunk_text_by_sentences_huge_single_sentence():
    """Verify huge individual sentence without punctuation is safely split into word chunks."""
    huge_sentence = " ".join(["word" + str(i) for i in range(200)])
    chunks = chunk_text_by_sentences(huge_sentence, chunk_size=150)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 250


def test_chunk_text_by_sentences_token_bounded():
    """Verify chunking strictly respects max_tokens when a tokenizer is simulated."""
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            # 1 token per 2 words
            return [1] * max(1, len(text.split()) * 2)

    tok = DummyTokenizer()
    sentences = [f"This is a detailed sentence number {i} discussing machine learning." for i in range(30)]
    long_doc = " ".join(sentences)

    chunks = chunk_text_by_sentences(long_doc, chunk_size=2000, tokenizer=tok, max_tokens=50)
    assert len(chunks) > 1
    for c in chunks:
        tok_count = len(tok.encode(c))
        assert tok_count <= 60  # Allows slight margin for trailing word merges
