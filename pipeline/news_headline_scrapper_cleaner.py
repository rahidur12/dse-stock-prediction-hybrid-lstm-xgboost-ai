# =============================
# news_headline_scrapper_cleaner.py
# =============================

import os
import pandas as pd
import re
import time
import random
import nltk
#from pygooglenews import GoogleNews
from gnews import GNews
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import dateparser
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Setup NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def is_relevant_headline(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    text = re.sub(r'http\S+|[^a-z\s]', '', text.lower())
    words = [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text) if w not in stop_words and len(w) > 2]
    return len(words) >= 3

def get_cleaned_bd_news(symbol, company_full_name):
    # project data path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "data")
    if not os.path.exists(data_dir): os.makedirs(data_dir)

    gn = GoogleNews(lang='en', country='BD')
    all_articles = []
    seen_urls = set()
    
    # --- Time chunking: 2 years weekly ---
    today = datetime.now()
    start_date = today - relativedelta(years=2)
    current = start_date
    date_chunks = []
    while current < today:
        next_week = current + timedelta(weeks=1)
        date_chunks.append((current, min(next_week, today)))
        current = next_week

    print(f"📊 Scraping 2 years of news for: {company_full_name}...")

    for chunk_start, chunk_end in tqdm(date_chunks, desc="Fetching Weeks"):
        from_date = chunk_start.strftime('%Y-%m-%d')
        to_date = chunk_end.strftime('%Y-%m-%d')
        query = f'"{company_full_name}" AND (Bangladesh OR DSE OR Stock)'
        
        try:
            search = gn.search(query, from_=from_date, to_=to_date)
            if search and 'entries' in search:
                for entry in search['entries']:
                    if entry.link not in seen_urls:
                        seen_urls.add(entry.link)
                        pub_date = dateparser.parse(entry.published)
                        if pub_date:
                            all_articles.append({
                                'Date': pub_date.strftime('%Y-%m-%d'),
                                'Headline': entry.title,
                                'URL': entry.link
                            })
            time.sleep(random.uniform(0.1, 0.2))
        except:
            continue

    if all_articles:
        df = pd.DataFrame(all_articles)
        df = df[df['Headline'].str.contains(company_full_name, case=False, na=False)]
        df = df.drop_duplicates(subset=['Headline'])
        print("🧹 Applying NLP cleaning...")
        df = df[df['Headline'].apply(is_relevant_headline)].reset_index(drop=True)

        # --- Save file using symbol ---
        output_file = os.path.join(data_dir, f"{symbol.lower()}_news_data.csv")
        df.to_csv(output_file, index=False)
        print(f"✅ PROCESS COMPLETE: Found {len(df)} relevant articles. Saved as {output_file}")
        return df
    
    print("❌ No articles found.")
    return None

if __name__ == "__main__":
    company_name = input("Enter the company full name to scrape: ")
    symbol = input("Enter the company symbol: ")
    get_cleaned_bd_news(symbol, company_name)