"""Tweet scraping utilities using Nitter."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from ntscraper import Nitter

from .config import AnalysisConfig


class TweetScraper:
    """Scrape tweets from X using Nitter instances."""

    def __init__(self, config: AnalysisConfig | None = None) -> None:
        self.config = config or AnalysisConfig()
        self.client = Nitter(
            host=self.config.nitter_instance, timeout=self.config.request_timeout
        )

    def fetch_user_tweets(self, username: str, limit: int | None = None) -> List[Dict[str, Any]]:
        """Return a list of tweet dictionaries for a user."""
        number = limit or self.config.tweets_limit
        result = self.client.get_tweets(username, mode="user", number=number)
        tweets = []
        for item in result.get("tweets", []):
            tweet_data = {
                "id": item.get("id"),
                "date": self._parse_date(item.get("date")),
                "text": item.get("text", ""),
                "replies": item.get("stats", {}).get("replies", 0),
                "retweets": item.get("stats", {}).get("retweets", 0),
                "quotes": item.get("stats", {}).get("quotes", 0),
                "likes": item.get("stats", {}).get("likes", 0),
                "link": item.get("link"),
            }
            tweets.append(tweet_data)
        return tweets

    @staticmethod
    def _parse_date(value: Any) -> dt.datetime | None:
        """Convert a Nitter date value into a timezone-aware datetime."""
        if isinstance(value, (int, float)):
            return dt.datetime.fromtimestamp(value, tz=dt.timezone.utc)
        if isinstance(value, str):
            try:
                return dt.datetime.fromisoformat(value)
            except ValueError:
                return None
        return None
