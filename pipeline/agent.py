# =============================
# agent.py
# =============================

import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import stock_data_scrapper_cleaner as sd
import news_headline_scrapper_cleaner as nh
import sentiment_analysis as sa

# Symbol → Full Name mapping
COMPANY_NAME_MAP = {
    "GP": "Grameenphone",
    "BRACBANK": "BRAC Bank",
    "BEXIMCO": "Beximco",
    "BATASHOE": "Bata Shoe"
}

def main():
    parser = argparse.ArgumentParser(description="DSE Stock & Sentiment Pipeline Agent")
    parser.add_argument("--company", type=str, required=True, help="Stock Symbol (e.g., GP, BRACBANK)")
    args = parser.parse_args()

    symbol = args.company.upper()
    company_full_name = COMPANY_NAME_MAP.get(symbol, symbol)
    
    print(f"\n🚀 STARTING PIPELINE FOR: {symbol}")
    print("="*40)

    # Step 1: Stock Data
    sd.download_and_fill_stock_data(symbol)

    # Step 2: News Scraper
    print(f"🔍 Searching news for: {company_full_name}")
    nh.get_cleaned_bd_news(symbol, company_full_name)

    # Step 3: Sentiment Analysis
    sa.apply_vader_sentiment(symbol)

    print(f"\n✅ PIPELINE COMPLETE: {symbol}")
    print(f"📂 Output files are ready in the 'data' folder.")

if __name__ == "__main__":
    main()