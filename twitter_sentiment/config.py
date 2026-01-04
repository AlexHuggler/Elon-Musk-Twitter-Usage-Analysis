"""Configuration for Twitter sentiment analysis."""
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


@dataclass
class AnalysisConfig:
    """Settings for the sentiment analysis pipeline."""

    tweets_limit: int = 200
    model_name: str = DEFAULT_MODEL
    max_length: int = 256
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    figures_dir: Path = field(default_factory=lambda: Path("figures"))
    request_timeout: int = 30
    nitter_instance: str | None = None

    def prepare_directories(self) -> None:
        """Ensure output directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
