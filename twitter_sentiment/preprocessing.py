"""Tweet preprocessing utilities."""
from __future__ import annotations

import re
from typing import Iterable, List

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


class TweetPreprocessor:
    """Clean tweet text for sentiment analysis."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    MENTION_HASHTAG_PATTERN = re.compile(r"[@#]\w+")
    EXTRA_SPACES_PATTERN = re.compile(r"\s{2,}")

    def __init__(self, remove_stopwords: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words("english")) if remove_stopwords else set()

    def clean_text(self, text: str) -> str:
        text = self.URL_PATTERN.sub("", text)
        text = self.MENTION_HASHTAG_PATTERN.sub("", text)
        text = re.sub(r"[^A-Za-z0-9'\s]", " ", text)
        tokens = [t.lower() for t in text.split() if t]
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        cleaned = " ".join(tokens)
        cleaned = self.EXTRA_SPACES_PATTERN.sub(" ", cleaned).strip()
        return cleaned

    def batch_clean(self, texts: Iterable[str]) -> List[str]:
        return [self.clean_text(text) for text in texts]
