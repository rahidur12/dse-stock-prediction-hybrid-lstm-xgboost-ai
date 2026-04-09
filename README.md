# DSE Stock Prediction using Hybrid LSTM, XGBoost & AI Research

**Sentiment-aware financial forecasting for the Dhaka Stock Exchange**\
Hybrid LSTM + XGBoost · VADER Sentiment · AI Research Agent · Streamlit UI

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Pipeline Overview](#pipeline-overview)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [End-to-End Application](#end-to-end-application)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Author & Contact](#author--contact)

---

## 📌 Project Overview

This project implements an **agentic, end-to-end financial forecasting system** for the **Dhaka Stock Exchange (DSE)**, combining:

- 📊 Market data (OHLCV)
- 📰 Real-time news sentiment
- 🤖 Hybrid machine learning models
- 🧠 Independent AI research agent

The system is designed to simulate a **real-world quantitative research pipeline**, where data is continuously collected, processed, modeled, and deployed into an interactive dashboard.

### 🔍 Core Capabilities

1. Stock Price Prediction using a hybrid LSTM + XGBoost architecture
2. Sentiment-aware modeling using financial news headlines
3. Automated data pipeline (agent-based)
4. AI-powered independent market research
5. Live visualization and prediction dashboard

---

## 🚀 Key Features

### 🤖 Agentic Pipeline

A CLI-based orchestration system automates:

- Stock data collection from DSE
- News scraping from Google News
- NLP cleaning and filtering
- Sentiment scoring and feature engineering

### 🧠 Sentiment Intelligence

- VADER sentiment analysis for financial news
- Daily sentiment aggregation
- Lag features (T-1, T-2)

### 📈 Hybrid ML Forecasting

- LSTM for temporal pattern learning
- XGBoost for feature-based refinement
- Supports both sentiment and non-sentiment models

### 📊 Live Market Integration

- Real-time DSE trade data
- Dynamic price synchronization

### 🧠 AI Research Agent

- Independent AI-generated market analysis
- Extracted numerical price targets

---

## ⚙️ Pipeline Overview

### 1. Stock Data Collection

- Fetches 2 years of OHLCV data
- Fills missing dates using forward-fill

### 2. News Scraping & Cleaning

- Google News scraping (Bangladesh region)
- NLP filtering and deduplication

### 3. Sentiment Feature Engineering

- VADER compound scores
- Daily aggregation
- Lag features for temporal impact

### 4. Dataset Construction

Final dataset includes:

- OHLCV features
- Sentiment scores
- Lagged sentiment features

---

## 🤖 Models Used

### LSTM

- Captures time-series dependencies
- 350 units + Dropout + Dense layer

### XGBoost

- Handles non-linear feature interactions
- Improves prediction stability

### Hybrid Model

- LSTM generates trend
- XGBoost refines output

---

## 📈 Evaluation Metrics

- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Correlation Score

---

## 🖥️ End-to-End Application

A Streamlit dashboard provides:

- Live candlestick charts
- Real-time price tracking
- Hybrid ML predictions
- AI-generated market research

---

## 📂 Project Structure

```
Agentic Stock Predictor/
│
├── app.py                          ← Streamlit dashboard
├── requirements.txt
│
├── pipeline/
│   ├── agent.py                    ← CLI orchestrator (run this first)
│   ├── stock_data_scrapper_cleaner.py
│   ├── news_headline_scrapper_cleaner.py
│   └── sentiment_analysis.py
│
├── data/                           ← All CSVs land here (auto-created)
│   ├── gp_stock_model_ready.csv
│   ├── gp_cleaned_news.csv
│   ├── grameenphone_final_sentiment_data.csv
│   ├── brac_stock_model_ready.csv
│   ├── brac_cleaned_news.csv
│   └── bracbank_final_sentiment_data.csv
│
├── gp_predictor/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── model_evaluation.py
│
├── brac_bank_predictor/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── model_evaluation.py
│
├── interference/
│   ├── predict_gp.py
│   └── predict_brac_bank.py
│
└── models/                         ← Saved artefacts (auto-created)
    ├── gp_lstm_model.keras
    ├── gp_xgb_model.pkl
    ├── gp_scaler_x.pkl
    ├── gp_scaler_y.pkl
    ├── bracbank_lstm_model.keras
    ├── bracbank_xgb_model.pkl
    ├── bracbank_scaler_x.pkl
    └── bracbank_scaler_y.pkl
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline (fetches data, scrapes news, builds sentiment CSV)
```bash
# For Grameenphone
python pipeline/agent.py --company gp

# For BRAC Bank
python pipeline/agent.py --company bracbank
```

### 3. Train the models
```bash
python gp_predictor/train.py
python brac_bank_predictor/train.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

---

## 🏗️ Architecture

```
Google News  ──►  news_headline_scrapper_cleaner.py
                        │
DSE / bdshare ──►  stock_data_scrapper_cleaner.py
                        │
                  sentiment_analysis.py  (VADER + DSE lexicon)
                        │
                  data/  ←── merged CSVs
                        │
            ┌───────────┴───────────┐
     gp_predictor/           brac_bank_predictor/
       train.py                   train.py
    (LSTM + XGBoost)           (LSTM + XGBoost)
            │                       │
       models/*.keras           models/*.keras
       models/*.pkl             models/*.pkl
            └───────────┬───────────┘
                  interference/
             predict_gp.py / predict_brac_bank.py
                        │
                     app.py  ←──  AI Research Agent
                  (Streamlit)       (live sentiment)
```

---

## 👤 Author

**Rahidur Rahman**\
Bachelor of Science in Computer Science & Engineering\
East Delta University

📧 [rahidurrahman12@gmail.com](mailto\:rahidurrahman12@gmail.com) 💻 [https://github.com/rahidur12](https://github.com/rahidur12)

---

⭐ If you find this project useful, feel free to star the repository!

