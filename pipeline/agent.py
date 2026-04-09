# =============================
# agent.py
# =============================

import argparse
import sys
import os

# Ensure the script can find its modules
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import your scripts
import stock_data_scrapper_cleaner as sd
import news_headline_scrapper_cleaner as nh
import sentiment_analysis as sa

def main():
    parser = argparse.ArgumentParser(description="DSE Stock & Sentiment Pipeline Agent")
    # You can now run this from terminal like: python agent.py --company GP
    parser.add_argument("--company", type=str, required=True, help="Stock Symbol (e.g., GP, BRACBANK)")
    args = parser.parse_args()
    
    symbol = args.company.upper()
    print(f"\n🚀 STARTING PIPELINE FOR: {symbol}")
    print("="*40)

    # ---------------------------------------------------------
    # Step 1: Download & Fill Stock Prices
    # ---------------------------------------------------------
    # This creates: data/gp_stock_data.csv
    sd.download_and_fill_stock_data(symbol)
    
    # ---------------------------------------------------------
    # Step 2: Fetch & Clean News Headlines
    # ---------------------------------------------------------
    # Mapping symbols to full names helps Google News find better results.
    # You can expand this dictionary as you add more companies to your thesis.
    name_map = {
        "GP": "Grameenphone",
        "BRACBANK": "BRAC Bank",
        "BEXIMCO": "Beximco",
        "BATASHOE": "Bata Shoe"
    }
    
    search_term = name_map.get(symbol, symbol)
    print(f"🔍 Searching news for: {search_term}")
    nh.get_cleaned_bd_news([search_term])
    
    # ---------------------------------------------------------
    # Step 3: Run VADER Sentiment Analysis
    # ---------------------------------------------------------
    # This takes the news file and produces the final sentiment CSV
    sa.apply_vader_sentiment(symbol)

    print(f"\n✅ PIPELINE COMPLETE: {symbol}")
    print(f"📂 Output files are ready in the 'data' folder.")

if __name__ == "__main__":
    main()