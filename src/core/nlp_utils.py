"""NLP text processing utilities: n-gram candidate extraction, stopwords, and document chunking."""

from __future__ import annotations

import re
from typing import List, Optional, Set, Tuple

# Comprehensive standard English stopwords list (zero external NLTK dependency required)
ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd",
    "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
    "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
})

_WORD_PATTERN = re.compile(r"(?u)\b[a-zA-Z0-9_-]+\b")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def extract_candidate_ngrams(
    text: str,
    ngram_range: Tuple[int, int] = (1, 1),
    candidate_filter: Optional[List[str]] = None,
    max_candidates: int = 100,
) -> List[str]:
    """Extract candidate keyword n-grams from text, filtering stopwords and duplicates.

    Args:
        text: Input raw text string.
        ngram_range: Tuple of (min_n, max_n). Defaults to (1, 1) for single keywords.
        candidate_filter: Optional list of explicit candidate keywords to evaluate.
        max_candidates: Maximum candidate phrases to retain for embedding evaluation.

    Returns:
        List of unique candidate n-gram strings preserving first-occurrence order.
    """
    if candidate_filter is not None:
        # User provided explicit candidates to score against document
        seen: Set[str] = set()
        clean_candidates: List[str] = []
        for c in candidate_filter:
            if isinstance(c, str) and c.strip():
                clean = c.strip().lower()
                if clean not in seen:
                    seen.add(clean)
                    clean_candidates.append(clean)
        return clean_candidates[:max_candidates]

    min_n, max_n = ngram_range
    if min_n < 1 or max_n < min_n or max_n > 3:
        raise ValueError(f"Invalid ngram_range ({min_n}, {max_n}). Must satisfy 1 <= min_n <= max_n <= 3.")

    # Tokenize words while preserving case for proper nouns, but lowercase for stopword filtering
    tokens = _WORD_PATTERN.findall(text)
    if not tokens:
        return []

    candidates: List[str] = []
    seen_candidates: Set[str] = set()

    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            ngram_tokens = tokens[i : i + n]
            first_word = ngram_tokens[0].lower()
            last_word = ngram_tokens[-1].lower()

            # Rule 1: First and last word cannot be stopwords
            if first_word in ENGLISH_STOPWORDS or last_word in ENGLISH_STOPWORDS:
                continue

            # Rule 2: Minimum token length of 2 characters and contains at least 1 letter
            if any(len(tok) < 2 or not any(char.isalpha() for char in tok) for tok in ngram_tokens):
                continue

            phrase = " ".join(ngram_tokens)
            norm_phrase = phrase.lower()

            if norm_phrase not in seen_candidates:
                seen_candidates.add(norm_phrase)
                candidates.append(phrase)

            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    return candidates


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = 2500,
    min_chunk_size: int = 200,
) -> List[str]:
    """Split text into natural sentence-bounded chunks for long-document summarization.

    Args:
        text: Input document text.
        chunk_size: Maximum character length target per chunk.
        min_chunk_size: Minimum characters to avoid trailing tiny fragments.

    Returns:
        List of text chunks preserving document ordering without truncated words.
    """
    if len(text) <= chunk_size:
        return [text]

    sentences = _SENTENCE_SPLIT_PATTERN.split(text.strip())
    if not sentences:
        return [text]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(sent)

        # If single sentence exceeds chunk_size, split by word
        if sent_len > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            words = sent.split()
            word_chunk: List[str] = []
            word_len = 0
            for w in words:
                if word_len + len(w) + 1 > chunk_size and word_chunk:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [w]
                    word_len = len(w)
                else:
                    word_chunk.append(w)
                    word_len += len(w) + 1
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        if current_length + sent_len + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = sent_len
        else:
            current_chunk.append(sent)
            current_length += sent_len + 1

    if current_chunk:
        # Merge with previous chunk if trailing chunk is too small
        last_text = " ".join(current_chunk)
        if chunks and len(last_text) < min_chunk_size and (len(chunks[-1]) + len(last_text) + 1 <= chunk_size * 1.2):
            chunks[-1] = chunks[-1] + " " + last_text
        else:
            chunks.append(last_text)

    return chunks
