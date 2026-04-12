import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import re
import importlib
from dotenv import load_dotenv
from datetime import datetime
from bdshare import get_current_trade_data

# -----------------------------
# CONFIG & SECRETS
# -----------------------------
st.set_page_config(page_title="DSE Agentic Predictor", layout="wide")
load_dotenv()

# Priority: Streamlit Cloud Secrets -> Local .env
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .ai-card { background-color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #00c8ff; margin-bottom: 20px; color: white; }
    .ai-header { font-size: 24px; font-weight: bold; color: white; margin-bottom: 10px; }
    .ai-sub { font-size: 14px; color: #aaa; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UTILITIES
# -----------------------------
def to_float(val):
    if val is None: return 0.0
    if isinstance(val, (list, np.ndarray)):
        return to_float(val[0]) if len(val) > 0 else 0.0
    try:
        if isinstance(val, str): val = val.replace(',', '')
        return float(val)
    except: return 0.0

# -----------------------------
# DYNAMIC PREDICTION ROUTER
# -----------------------------
def run_ticker_prediction(ticker):
    """Dynamically imports the correct prediction script for the selected ticker."""
    try:
        # Map tickers to their specific interference scripts
        module_map = {
            "GP": "interference.predict_gp",
            "BRACBANK": "interference.predict_brac_bank"
        }
        
        module_path = module_map.get(ticker.upper())
        if not module_path:
            return f"No model script found for {ticker}", 0.0
            
        # Dynamically import/reload the module
        module = importlib.import_module(module_path)
        return module.get_prediction(ticker.lower())
    except Exception as e:
        return f"Prediction Route Error: {str(e)}", 0.0

# -----------------------------
# INDEPENDENT AI RESEARCH
# -----------------------------
def get_ai_independent_forecast(symbol, price, sentiment_avg, weekly_trend):
    if not API_KEY:
        return "AI Research Offline: Add OPENROUTER_API_KEY to Streamlit Secrets."
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        prompt = f"Senior DSE Analyst: Forecast {symbol} for tomorrow. Current Price: {price} BDT, Sentiment: {sentiment_avg:.2f}, Trend: {weekly_trend}. Output MUST include 'AI_TARGET: XXX.X BDT'."
        data = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4 
        }
        response = requests.post(url, headers=headers, json=data, timeout=15)
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI Error: {str(e)}"

# -----------------------------
# LIVE DSE DATA SCRAPER
# -----------------------------
def fetch_live_dse(symbol):
    try:
        df_live = get_current_trade_data()
        row = df_live[df_live['symbol'].str.upper() == symbol.upper().strip()]
        if not row.empty:
            res = row.iloc[0]
            return {
                "Open": to_float(res.get('open', 0)) or to_float(res.get('ycp', 0)),
                "High": to_float(res.get('high', 0)),
                "Low": to_float(res.get('low', 0)),
                "Close": to_float(res.get('ltp', 0)),
                "Volume": to_float(res.get('volume', 0)),
                "Time": datetime.now().strftime("%I:%M %p")
            }
    except: return None

# -----------------------------
# UI LOGIC
# -----------------------------
st.title("📈 DSE Agentic Stock Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controls")
    symbol = st.selectbox("Select Ticker", ["GP", "BRACBANK"]) # Removed Beximco if models aren't ready

    if st.button("⚡ Sync Live Price", type="primary", use_container_width=True):
        live_data = fetch_live_dse(symbol)
        if live_data:
            st.session_state[f"{symbol}_live"] = live_data
            st.success(f"Synced {symbol} at {live_data['Time']}")

    if st.button("🔮 Predict Tomorrow", use_container_width=True):
        with st.spinner(f"Running Hybrid Model for {symbol}..."):
            # Use the dynamic router to prevent 0.00 BDT errors
            raw_pred, csv_close = run_ticker_prediction(symbol)
            
            if isinstance(raw_pred, str):
                st.error(raw_pred) # Show the actual path/loading error
                pred_value = 0.0
            else:
                pred_value = to_float(raw_pred)
            
            live = st.session_state.get(f"{symbol}_live")
            current_close = to_float(live["Close"] if live else csv_close)
            
            st.metric("Last Close", f"{current_close:.2f} BDT")
            if pred_value > 0:
                st.metric("Model Prediction", f"{pred_value:.2f} BDT", delta=f"{pred_value - current_close:.2f} BDT")

with col2:
    st.subheader("Price History")
    data_path = f"data/{symbol.lower()}_final_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        live = st.session_state.get(f"{symbol}_live", {})
        d_close = to_float(live.get("Close", df['Close'].iloc[-1]))
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Open", f"{to_float(live.get('Open', df['Open'].iloc[-1])):.2f}")
        m2.metric("High", f"{to_float(live.get('High', df['High'].iloc[-1])):.2f}")
        m3.metric("Low", f"{to_float(live.get('Low', df['Low'].iloc[-1])):.2f}")
        m4.metric("Close", f"{d_close:.2f}")
        m5.metric("Volume", f"{int(to_float(live.get('Volume', df['Volume'].iloc[-1]))):,}")

        chart_df = df.tail(30).copy()
        fig = go.Figure(data=[go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'])])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# AI RESEARCH SECTION
# -----------------------------
st.markdown("---")
if st.button("🤖 Get AI Independent Research & Prediction", use_container_width=True):
    live = st.session_state.get(f"{symbol}_live", {})
    price = to_float(live.get("Close", 0))
    
    if price > 0:
        with st.spinner("Analyzing Market Sentiment..."):
            hist_df = pd.read_csv(f"data/{symbol.lower()}_final_dataset.csv")
            avg_sent = hist_df['vader_score'].tail(7).mean() if 'vader_score' in hist_df.columns else 0.0
            weekly_trend = "Bullish" if hist_df['Close'].iloc[-1] > hist_df['Close'].iloc[-7] else "Bearish"
            
            ai_raw_text = get_ai_independent_forecast(symbol, price, avg_sent, weekly_trend)
            st.markdown(f'<div class="ai-card"><div class="ai-header">🧠 AI Research</div>{ai_raw_text}</div>', unsafe_allow_html=True)
            
            ai_match = re.search(r"AI_TARGET:\s*([\d\.]+)", ai_raw_text)
            if ai_match:
                ai_val = float(ai_match.group(1))
                st.metric("AI Target", f"{ai_val:.2f} BDT", delta=f"{ai_val - price:.2f} BDT")
    else:
        st.error("Please sync live price first.")