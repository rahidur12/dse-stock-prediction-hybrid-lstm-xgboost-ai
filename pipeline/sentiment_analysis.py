# =============================
# sentiment_analysis.py
# =============================

import pandas as pd
import numpy as np
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def apply_vader_sentiment(symbol):
    # --- 1. SETUP PATHS ---
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data")
    
    sym = symbol.lower()
    
    # Dynamic file naming based on the symbol provided
    news_file = f"{sym}_news_data.csv"
    stock_file = f"{sym}_stock_data.csv"
    output_file = f"{sym}_final_dataset.csv"

    news_path = os.path.join(data_dir, news_file)
    stock_path = os.path.join(data_dir, stock_file)
    output_path = os.path.join(data_dir, output_file)

    # Verification before processing
    if not os.path.exists(news_path) or not os.path.exists(stock_path):
        print(f"❌ Sentiment Error: Missing files in {data_dir}")
        print(f"Looking for: {news_file} AND {stock_file}")
        return

    # --- 2. INITIALIZE VADER ---
    vader_analyzer = SentimentIntensityAnalyzer()
    
    # --- 3. LOAD DATA ---
    print(f"✨ Analyzing sentiment for {symbol.upper()}...")
    df_news = pd.read_csv(news_path, parse_dates=['Date'])
    df_stock = pd.read_csv(stock_path, parse_dates=['Date'])

    # Calculate VADER scores
    # Compound score ranges from -1 (Extremely Negative) to +1 (Extremely Positive)
    df_news['vader_score'] = df_news['Headline'].apply(
        lambda x: vader_analyzer.polarity_scores(str(x))['compound']
    )
    
    # Aggregate scores: If a day has 5 news articles, we take the average mood
    daily_sentiment = df_news.groupby('Date')[['vader_score']].mean().reset_index()

    # --- 4. MERGE (The "Left Join" Strategy) ---
    # Ensure column names are clean
    df_stock.columns = [c.strip().capitalize() for c in df_stock.columns]
    
    # Merge news into the stock timeline. 
    # Days with no news get a 0.0 (Neutral) sentiment.
    final_df = pd.merge(df_stock, daily_sentiment, on='Date', how='left').fillna(0)

    # --- 5. FEATURE ENGINEERING ---
    # Lagged features: Helps the LSTM understand that news from 24-48 hours ago
    # might still be affecting the stock price today.
    final_df['vader_lag1'] = final_df['vader_score'].shift(1)
    final_df['vader_lag2'] = final_df['vader_score'].shift(2)
    final_df.fillna(0, inplace=True)

    # Save final output
    final_df.to_csv(output_path, index=False)
    print(f"✅ SUCCESS: Dataset saved at: {output_path}")

if __name__ == "__main__":
    # Now you can call it for any company symbol
    user_symbol = input("Enter symbol for sentiment analysis (e.g., bracbank): ").strip()
    apply_vader_sentiment(user_symbol)