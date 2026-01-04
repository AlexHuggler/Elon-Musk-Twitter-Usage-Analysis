# Twitter Sentiment Analysis (Modernized)

This project provides a modular, class-based pipeline for scraping recent tweets from any X (formerly Twitter) account, cleaning the text, and running sentiment analysis with the latest **cardiffnlp/twitter-roberta-base-sentiment-latest** model. Visual reports and CSV exports are generated automatically for the requested handle.

## Features

- **Pluggable scraper:** Uses [`ntscraper`](https://pypi.org/project/ntscraper/) (Nitter) to fetch recent tweets, reactions, and timestamps.
- **State-of-the-art NLP:** Hugging Face transformer fine-tuned for modern Twitter/X sentiment.
- **Parametric analysis:** `analyze_user("somehandle")` works for any account and names output files after the handle (e.g., `somehandle_tweets.csv`).
- **Visualization:** Matplotlib/Seaborn charts with user-specific titles for quick reporting.
- **Configurable:** Adjust tweet limits, model name, and output directories via `AnalysisConfig`.

## Project Structure

```
.
├── twitter_sentiment/
│   ├── __init__.py
│   ├── config.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── scraper.py
│   ├── sentiment.py
│   └── visualization.py
├── requirements.txt
├── Twitter_Sentiment_Analysis_Modern.ipynb
└── ... (legacy files)
```

## Quickstart

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run an analysis for any user**

   ```python
   from twitter_sentiment import analyze_user, AnalysisConfig

   df, dist_plot, trend_plot = analyze_user(
       "BillGates",  # Twitter handle without the @ symbol
       config=AnalysisConfig(tweets_limit=150)
   )
   print(df.head())
   print(f"Saved data to: {dist_plot}")
   ```

   Outputs:

   - CSV: `outputs/<handle>_tweets.csv`
   - Figures: `figures/<handle>_sentiment_distribution.png`, `figures/<handle>_sentiment_trend.png`

## Google Colab

Use the provided notebook `Twitter_Sentiment_Analysis_Modern.ipynb` for a Colab-ready walkthrough that installs requirements, scrapes tweets, and visualizes sentiment for any handle.

## Notes

- The scraper relies on public Nitter instances; availability may vary. Configure a preferred instance via `AnalysisConfig(nitter_instance="https://nitter.net")` if needed.
- Hugging Face models download on first use; running on GPU (if available) will accelerate inference.
