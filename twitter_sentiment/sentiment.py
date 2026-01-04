"""Sentiment analysis powered by Hugging Face models."""
from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.pipelines import pipeline

from .config import AnalysisConfig


class SentimentAnalyzer:
    """Run sentiment predictions using a transformer model."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        )
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
            truncation=True,
            max_length=self.config.max_length,
            return_all_scores=True,
        )

    def predict(self, texts: Iterable[str]) -> List[Dict[str, float]]:
        """Return probability distributions for sentiment labels."""
        predictions = self.pipeline(list(texts))
        results: List[Dict[str, float]] = []
        for pred in predictions:
            scores = {entry["label"].lower(): entry["score"] for entry in pred}
            label = max(pred, key=lambda x: x["score"]).get("label", "neutral")
            results.append({"label": label.upper(), **scores})
        return results
