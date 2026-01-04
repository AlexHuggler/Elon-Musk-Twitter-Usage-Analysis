"""Plotting utilities for sentiment analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import AnalysisConfig


class SentimentVisualizer:
    """Visualize sentiment distributions."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()

    def sentiment_distribution(self, df: pd.DataFrame, username: str) -> Path:
        plt.figure(figsize=(8, 5))
        order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]
        sns.countplot(data=df, x="sentiment", order=order, palette="viridis")
        plt.title(f"Sentiment Distribution for @{username}")
        plt.xlabel("Sentiment")
        plt.ylabel("Tweet Count")
        plt.tight_layout()

        output_path = self.config.figures_dir / f"{username}_sentiment_distribution.png"
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return output_path

    def sentiment_time_series(self, df: pd.DataFrame, username: str) -> Path:
        df = df.copy()
        df["date_only"] = df["date"].dt.date
        summary = df.groupby(["date_only", "sentiment"]).size().reset_index(name="count")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=summary, x="date_only", y="count", hue="sentiment", marker="o")
        plt.title(f"Daily Sentiment Trend for @{username}")
        plt.xlabel("Date")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()

        output_path = self.config.figures_dir / f"{username}_sentiment_trend.png"
        self.config.figures_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        return output_path
