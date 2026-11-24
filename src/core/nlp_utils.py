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


def count_tokens_or_estimate(text: str, tokenizer: Optional[Any] = None) -> int:
    """Accurately count tokens using a provided tokenizer, or conservatively estimate count.

    Conservative estimation uses 1 token ~= 3.5 characters or word count * 1.33.
    """
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Conservative fallback: max of word-based heuristic and character-based heuristic
    words = text.split()
    word_est = int(len(words) * 1.35)
    char_est = int(len(text) / 3.5)
    return max(word_est, char_est, 1)


def chunk_text_by_sentences(
    text: str,
    chunk_size: int = 1800,
    min_chunk_size: int = 200,
    tokenizer: Optional[Any] = None,
    max_tokens: int = 600,
) -> List[str]:
    """Split text into natural sentence-bounded chunks for long-document summarization.

    Enforces both character boundaries and tokenizer/token limits (default 600 tokens)
    to strictly guarantee that model input limits (1024 tokens for distilbart-cnn-12-6)
    are respected without silent truncation.

    Args:
        text: Input document text.
        chunk_size: Maximum character length target per chunk.
        min_chunk_size: Minimum characters to avoid trailing tiny fragments.
        tokenizer: Optional Hugging Face tokenizer to compute exact token lengths.
        max_tokens: Maximum token limit per chunk (leaves buffer below 1024 for special tokens).

    Returns:
        List of text chunks preserving document ordering without truncated words.
    """
    if not text or not text.strip():
        return []

    clean_text = text.strip()
    initial_tokens = count_tokens_or_estimate(clean_text, tokenizer)
    if len(clean_text) <= chunk_size and initial_tokens <= max_tokens:
        return [clean_text]

    sentences = _SENTENCE_SPLIT_PATTERN.split(clean_text)
    if not sentences:
        return [clean_text]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_length = 0
    current_tokens = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_len = len(sent)
        sent_tokens = count_tokens_or_estimate(sent, tokenizer)

        # If a single sentence exceeds chunk_size or max_tokens, split by word
        if sent_len > chunk_size or sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
                current_tokens = 0

            words = sent.split()
            word_chunk: List[str] = []
            word_len = 0
            word_toks = 0
            for w in words:
                w_tok = count_tokens_or_estimate(w, tokenizer)
                if (word_len + len(w) + 1 > chunk_size or word_toks + w_tok > max_tokens) and word_chunk:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = [w]
                    word_len = len(w)
                    word_toks = w_tok
                else:
                    word_chunk.append(w)
                    word_len += len(w) + 1
                    word_toks += w_tok
            if word_chunk:
                chunks.append(" ".join(word_chunk))
            continue

        if (current_length + sent_len + 1 > chunk_size or current_tokens + sent_tokens > max_tokens) and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_length = sent_len
            current_tokens = sent_tokens
        else:
            current_chunk.append(sent)
            current_length += sent_len + 1
            current_tokens += sent_tokens

    if current_chunk:
        last_text = " ".join(current_chunk)
        last_tokens = count_tokens_or_estimate(last_text, tokenizer)
        if chunks and len(last_text) < min_chunk_size and (len(chunks[-1]) + len(last_text) + 1 <= chunk_size * 1.1) and (count_tokens_or_estimate(chunks[-1] + " " + last_text, tokenizer) <= max_tokens):
            chunks[-1] = chunks[-1] + " " + last_text
        else:
            chunks.append(last_text)

    return chunks
