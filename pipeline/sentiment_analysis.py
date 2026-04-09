# =============================
# sentiment_analysis.py
# =============================

import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def apply_vader_sentiment(symbol):
    # --- Paths ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    news_file = f"{symbol.lower()}_news_data.csv"
    stock_file = f"{symbol.lower()}_stock_data.csv"
    output_file = f"{symbol.lower()}_final_dataset.csv"

    news_path = os.path.join(data_dir, news_file)
    stock_path = os.path.join(data_dir, stock_file)
    output_path = os.path.join(data_dir, output_file)

    if not os.path.exists(news_path) or not os.path.exists(stock_path):
        print(f"❌ Sentiment Error: Missing files in {data_dir}")
        print(f"Looking for: {news_file} AND {stock_file}")
        return

    print(f"✨ Analyzing sentiment for {symbol.upper()}...")
    df_news = pd.read_csv(news_path, parse_dates=['Date'])
    df_stock = pd.read_csv(stock_path, parse_dates=['Date'])

    vader_analyzer = SentimentIntensityAnalyzer()
    df_news['vader_score'] = df_news['Headline'].apply(lambda x: vader_analyzer.polarity_scores(str(x))['compound'])
    daily_sentiment = df_news.groupby('Date')[['vader_score']].mean().reset_index()

    df_stock.columns = [c.strip().capitalize() for c in df_stock.columns]
    final_df = pd.merge(df_stock, daily_sentiment, on='Date', how='left').fillna(0)
    final_df['vader_lag1'] = final_df['vader_score'].shift(1)
    final_df['vader_lag2'] = final_df['vader_score'].shift(2)
    final_df.fillna(0, inplace=True)

    final_df.to_csv(output_path, index=False)
    print(f"✅ SUCCESS: Dataset saved at: {output_path}")

if __name__ == "__main__":
    symbol = input("Enter symbol for sentiment analysis (e.g., GP): ").strip()
    apply_vader_sentiment(symbol)