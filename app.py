import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import re
import importlib
from bs4 import BeautifulSoup # Required for fallback scraping
from dotenv import load_dotenv
from datetime import datetime
from bdshare import get_current_trade_data

# -----------------------------
# DYNAMIC PREDICTION LOADER
# -----------------------------
def get_prediction_dynamically(symbol, live_data=None):
    try:
        module_map = {
            "GP": "interference.predict_gp",
            "BRACBANK": "interference.predict_brac_bank" 
        }
        module_name = module_map.get(symbol.upper(), "interference.predict_gp")
        module = importlib.import_module(module_name)
        importlib.reload(module)
        return module.get_prediction(symbol.lower(), live_entry=live_data)
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# -----------------------------
# CONFIG & UTILS
# -----------------------------
st.set_page_config(page_title="DSE Agentic Predictor", layout="wide")
load_dotenv()
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

def to_float(val):
    if val is None or str(val).strip() == "--" or str(val).strip() == "": return 0.0
    try:
        if isinstance(val, (list, np.ndarray)): val = val[0]
        val = str(val).replace(',', '').strip()
        return float(val)
    except: return 0.0

# -----------------------------
# ROBUST LIVE FETCH (Library + Fallback Scraper)
# -----------------------------
def fetch_live_dse(symbol):
    search_sym = symbol.upper().strip()
    try:
        # Method 1: Try bdshare library
        df_live = get_current_trade_data()
        if df_live is not None and not df_live.empty:
            # Flexible filtering for symbol names
            row = df_live[df_live['symbol'].str.upper().str.replace(" ", "") == search_sym.replace(" ", "")]
            if not row.empty:
                res = row.iloc[0]
                return {
                    "Open": to_float(res.get('open')),
                    "High": to_float(res.get('high')),
                    "Low": to_float(res.get('low')),
                    "Close": to_float(res.get('ltp')), # LTP is the real-time Close
                    "Volume": int(to_float(res.get('volume'))),
                    "Time": datetime.now().strftime("%I:%M %p")
                }

        # Method 2: Manual Fallback Scraper (Direct from DSE)
        headers = {'User-Agent': 'Mozilla/5.0'}
        url = "https://www.dsebd.org/latest_share_price_scroll_l.php"
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Scrape the table directly
        for tr in soup.find_all('tr'):
            tds = tr.find_all('td')
            if len(tds) > 8 and tds[1].text.strip().upper() == search_sym:
                return {
                    "Open": to_float(tds[4].text),
                    "High": to_float(tds[5].text),
                    "Low": to_float(tds[6].text),
                    "Close": to_float(tds[3].text), # LTP/Close
                    "Volume": int(to_float(tds[9].text)),
                    "Time": datetime.now().strftime("%I:%M %p")
                }
    except Exception as e:
        st.error(f"Sync failed: {str(e)}")
    return None

# -----------------------------
# CORE LOGIC & UI
# -----------------------------
st.title("📈 DSE Agentic Stock Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])
symbol = col1.selectbox("Select Ticker", ["GP", "BRACBANK"])

if col1.button("⚡ Sync Live Price", type="primary", use_container_width=True):
    live_data = fetch_live_dse(symbol)
    if live_data:
        st.session_state[f"{symbol}_live"] = live_data
        st.success(f"Fetched {symbol} from DSE at {live_data['Time']}")
    else:
        st.error("Could not fetch live data. Market might be closed or DSE site is down.")

if col1.button("🔮 Predict Tomorrow", use_container_width=True):
    with st.spinner("Calculating..."):
        live = st.session_state.get(f"{symbol}_live")
        raw_pred, csv_close = get_prediction_dynamically(symbol, live_data=live)
        
        pred_value = to_float(raw_pred) if not isinstance(raw_pred, str) else 0.0
        current_close = to_float(live["Close"] if live else csv_close)
        
        diff = pred_value - current_close
        col1.metric("Current Price Used", f"{current_close:.2f} BDT")
        col1.metric("Model Prediction", f"{pred_value:.2f} BDT", 
                  delta=f"{diff:.2f} BDT" if pred_value > 0 else None)

# --- Historical Chart & AI Research follows same pattern as your original ---
# [Keep the rest of your original logic for Price History and AI Research]

# 3. Price History
with col2:
    st.subheader("Price History")
    data_path = f"data/{symbol.lower()}_final_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        live = st.session_state.get(f"{symbol}_live", {})
        
        disp_time = f"Last Synced: {live.get('Time')}" if live else f"Last Updated: {df['Date'].iloc[-1]}"
        st.markdown(f"**📅 {disp_time}**")

        # Metrics display using live override
        vals = [to_float(live.get(k, df[k].iloc[-1])) for k in ['Open', 'High', 'Low', 'Close', 'Volume']]
        m1, m2, m3, m4, m5 = st.columns(5)
        for m, label, val in zip([m1,m2,m3,m4,m5], ["Open","High","Low","Close","Volume"], vals):
            m.metric(label, f"{val:.2f}" if label != "Volume" else f"{int(val):,}")

        # Candlestick
        chart_df = df.tail(30).copy()
        if live:
            chart_df = pd.concat([chart_df, pd.DataFrame([{'Date': 'Live', 'Open': vals[0], 'High': vals[1], 'Low': vals[2], 'Close': vals[3]}])], ignore_index=True)
        fig = go.Figure(data=[go.Candlestick(x=chart_df['Date'], open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'])])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0,r=0,t=10,b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# 4. AI Research
st.markdown("---")
if st.button("🤖 Get AI Independent Research & Prediction", use_container_width=True):
    live = st.session_state.get(f"{symbol}_live", {})
    df = pd.read_csv(data_path) if os.path.exists(data_path) else None
    price = to_float(live.get("Close", df['Close'].iloc[-1] if df is not None else 0))
    
    if price > 0:
        with st.spinner("AI Analysis..."):
            avg_sent = df['vader_score'].tail(7).mean() if df is not None and 'vader_score' in df.columns else 0.0
            trend = "Bullish" if df is not None and df['Close'].iloc[-1] > df['Close'].iloc[-7] else "Neutral"
            
            ai_text = get_ai_independent_forecast(symbol, price, avg_sent, trend)
            # ML Model Prediction with Live Data
            raw_model_pred, _ = get_prediction_dynamically(symbol, live_data=live)
            model_pred = to_float(raw_model_pred)
            
            st.markdown(f'<div class="ai-card"><div class="ai-header">🧠 AI Research</div>{ai_text}</div>', unsafe_allow_html=True)
            
            l, r = st.columns(2)
            l.metric("🤖 ML Model Target", f"{model_pred:.2f} BDT", delta=f"{model_pred-price:.2f} BDT")
            
            ai_match = re.search(r"AI_TARGET:\s*([\d,.]+)", ai_text)
            if ai_match:
                ai_val = float(ai_match.group(1).replace(',', ''))
                r.metric("🧠 AI Outlook Target", f"{ai_val:.2f} BDT", delta=f"{ai_val-price:.2f} BDT")