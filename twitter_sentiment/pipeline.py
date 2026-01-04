"""High-level pipeline for Twitter sentiment analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .config import AnalysisConfig
from .preprocessing import TweetPreprocessor
from .scraper import TweetScraper
from .sentiment import SentimentAnalyzer
from .visualization import SentimentVisualizer


def analyze_user(
    username: str, config: AnalysisConfig | None = None, tweets_limit: int | None = None
) -> Tuple[pd.DataFrame, Path, Path]:
    """Run scraping, sentiment analysis, and visualization for a user."""
    cfg = config or AnalysisConfig()
    cfg.prepare_directories()
    limit = tweets_limit or cfg.tweets_limit

    scraper = TweetScraper(cfg)
    preprocessor = TweetPreprocessor()
    analyzer = SentimentAnalyzer(cfg)
    visualizer = SentimentVisualizer(cfg)

    tweets = scraper.fetch_user_tweets(username, limit)
    if not tweets:
        raise ValueError(f"No tweets found for @{username}.")

    df = pd.DataFrame(tweets)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["clean_text"] = preprocessor.batch_clean(df["text"].fillna(""))

    predictions = analyzer.predict(df["clean_text"])
    prediction_df = pd.DataFrame(predictions)
    df = pd.concat([df, prediction_df], axis=1)
    df.rename(columns={"label": "sentiment"}, inplace=True)

    csv_path = cfg.output_dir / f"{username}_tweets.csv"
    df.to_csv(csv_path, index=False)

    dist_plot = visualizer.sentiment_distribution(df, username)
    trend_plot = visualizer.sentiment_time_series(df, username)

    return df, dist_plot, trend_plot
