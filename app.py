# =============================
# interference/app.py
# =============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import re
import importlib
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from datetime import datetime

# -----------------------------
# DYNAMIC PREDICTION LOADER
# -----------------------------
def get_prediction_dynamically(symbol, live_data=None):
    """
    Loads prediction module and passes live_data to override CSV values.
    """
    try:
        module_map = {
            "GP": "interference.predict_gp",
            "BRACBANK": "interference.predict_brac_bank"
        }

        module_name = module_map.get(symbol.upper(), "interference.predict_gp")
        module = importlib.import_module(module_name)
        importlib.reload(module)

        # We pass live_data here so the model uses the latest DSE price
        return module.get_prediction(symbol.lower(), live_entry=live_data)
    except Exception as e:
        return f"Error: {str(e)}", 0.0

# -----------------------------
# CONFIG & UI ASSETS
# -----------------------------
st.set_page_config(page_title="DSE Agentic Predictor", layout="wide")
load_dotenv()
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")

st.markdown("""
<style>
    .ai-card { background-color: #111; padding: 20px; border-radius: 10px; border-left: 5px solid #00c8ff; margin-bottom: 20px; color: white; }
    .ai-header { font-size: 24px; font-weight: bold; color: white; margin-bottom: 10px; }
    .ai-sub { font-size: 14px; color: #aaa; margin-bottom: 15px; }
</style>
""", unsafe_allow_html=True)

def to_float(val):
    if val is None: return 0.0
    try:
        if isinstance(val, (list, np.ndarray)): val = val[0]
        if isinstance(val, str): val = val.replace(',', '').strip()
        return float(val)
    except: return 0.0

# -----------------------------
# DSE SCRAPER (replaces bdshare)
# -----------------------------
def fetch_live_dse(symbol):
    """
    Scrapes live OHLCV data from dsebd.org displayCompany page.
    Falls back gracefully on any error.
    """
    url = f"https://www.dsebd.org/displayCompany.php?name={symbol.upper()}"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # ---- Strategy 1: Look for the company info table that holds OHLCV ----
        # dsebd displayCompany page has a <table> with rows like:
        #   Last Trading Price | Open | High | Low | Close | Volume | YCP ...
        # We parse ALL tables and look for the one containing these keywords.
        data = {}
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True).lower()
                    value = cells[1].get_text(strip=True)
                    # Map DSE label variants → standard keys
                    if any(k in label for k in ["last trading price", "last trade price", "ltp", "closing price", "close"]):
                        data["Close"] = value
                    elif "open" in label:
                        data["Open"] = value
                    elif "high" in label:
                        data["High"] = value
                    elif "low" in label:
                        data["Low"] = value
                    elif "volume" in label:
                        data["Volume"] = value
                    elif any(k in label for k in ["yesterday", "ycp", "previous"]):
                        data["YCP"] = value

        # ---- Strategy 2: Parse the prominent price badge / header span ----
        # Some versions of the page embed the LTP in a <span class="..."> or <b>
        if "Close" not in data:
            for tag in soup.find_all(["span", "b", "strong", "td"]):
                txt = tag.get_text(strip=True).replace(",", "")
                # A plausible BDT price for DSE is between 1 and 10000
                try:
                    val = float(txt)
                    if 1.0 < val < 10000.0:
                        data["Close"] = str(val)
                        break
                except ValueError:
                    continue

        # ---- Build result dict ----
        close  = to_float(data.get("Close", 0))
        open_  = to_float(data.get("Open", 0))
        high   = to_float(data.get("High", 0))
        low    = to_float(data.get("Low", 0))
        volume = to_float(data.get("Volume", 0))
        ycp    = to_float(data.get("YCP", 0))

        # If open is missing fall back to ycp
        if open_ == 0:
            open_ = ycp if ycp > 0 else close

        if close > 0:
            return {
                "Open":   open_,
                "High":   high,
                "Low":    low,
                "Close":  close,
                "Volume": volume,
                "Time":   datetime.now().strftime("%I:%M %p"),
                "Source": "dsebd.org"
            }
        else:
            return None

    except Exception as e:
        st.warning(f"DSE scrape failed for {symbol}: {e}")
        return None

# -----------------------------
# CORE FUNCTIONS
# -----------------------------
def get_ai_independent_forecast(symbol, price, sentiment_avg, weekly_trend):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        prompt = f"""
        Act as a Senior Market Researcher for the Dhaka Stock Exchange.
        Analyze the following data for {symbol}:
        - Current Price: {price} BDT
        - 7-Day Avg Sentiment: {sentiment_avg}
        - Weekly Trend: {weekly_trend}

        Provide a concise market outlook.
        CRITICAL: End with 'AI_TARGET: ' followed by your predicted numerical price.
        """
        data = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
        response = requests.post(url, headers=headers, json=data)
        result = response.json()
        return result["choices"][0]["message"]["content"] if "choices" in result else "AI Research Offline."
    except Exception as e:
        return f"AI Error: {str(e)}"

# -----------------------------
# UI MAIN
# -----------------------------
st.title("📈 DSE Agentic Stock Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])
symbol = col1.selectbox("Select Ticker", ["GP", "BRACBANK"])

data_path = f"data/{symbol.lower()}_final_dataset.csv"

# 1. Sync Logic
if col1.button("⚡ Sync Live Price", type="primary", use_container_width=True):
    with st.spinner(f"Fetching live data from dsebd.org for {symbol}..."):
        live_data = fetch_live_dse(symbol)
    if live_data:
        st.session_state[f"{symbol}_live"] = live_data
        st.success(
            f"✅ Synced {symbol} @ {live_data['Close']:.2f} BDT "
            f"(O:{live_data['Open']:.2f} H:{live_data['High']:.2f} "
            f"L:{live_data['Low']:.2f}) — {live_data['Time']}"
        )
    else:
        st.error(
            "⚠️ Could not fetch live data from dsebd.org. "
            "The site may be down or blocking requests. "
            "Prediction will fall back to latest CSV values."
        )

# 2. Prediction Logic
if col1.button("🔮 Predict Tomorrow", use_container_width=True):
    with st.spinner("Calculating..."):
        live = st.session_state.get(f"{symbol}_live")
        # PASS LIVE DATA TO THE ML MODEL
        raw_pred, csv_close = get_prediction_dynamically(symbol, live_data=live)

        pred_value    = to_float(raw_pred) if not isinstance(raw_pred, str) else 0.0
        current_close = to_float(live["Close"] if live else csv_close)

        diff = pred_value - current_close
        col1.metric("Last Price Used", f"{current_close:.2f} BDT")
        col1.metric(
            "Model Prediction",
            f"{pred_value:.2f} BDT",
            delta=f"{diff:.2f} BDT" if pred_value > 0 else None
        )

# 3. Price History
with col2:
    st.subheader("Price History")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        live = st.session_state.get(f"{symbol}_live", {})

        disp_time = (
            f"Last Synced: {live.get('Time')} (dsebd.org)"
            if live else
            f"Last Updated: {df['Date'].iloc[-1]} (CSV)"
        )
        st.markdown(f"**📅 {disp_time}**")

        # Metrics display using live override
        vals = [to_float(live.get(k, df[k].iloc[-1])) for k in ['Open', 'High', 'Low', 'Close', 'Volume']]
        m1, m2, m3, m4, m5 = st.columns(5)
        for m, label, val in zip([m1, m2, m3, m4, m5], ["Open", "High", "Low", "Close", "Volume"], vals):
            m.metric(label, f"{val:.2f}" if label != "Volume" else f"{int(val):,}")

        # Candlestick
        chart_df = df.tail(30).copy()
        if live:
            chart_df = pd.concat(
                [
                    chart_df,
                    pd.DataFrame([{
                        'Date':  'Live',
                        'Open':  vals[0],
                        'High':  vals[1],
                        'Low':   vals[2],
                        'Close': vals[3]
                    }])
                ],
                ignore_index=True
            )
        fig = go.Figure(data=[go.Candlestick(
            x=chart_df['Date'],
            open=chart_df['Open'],
            high=chart_df['High'],
            low=chart_df['Low'],
            close=chart_df['Close']
        )])
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Dataset not found at: {data_path}")

# 4. AI Research
st.markdown("---")
if st.button("🤖 Get AI Independent Research & Prediction", use_container_width=True):
    live = st.session_state.get(f"{symbol}_live", {})
    df   = pd.read_csv(data_path) if os.path.exists(data_path) else None
    price = to_float(live.get("Close", df['Close'].iloc[-1] if df is not None else 0))

    if price > 0:
        with st.spinner("AI Analysis..."):
            avg_sent = df['vader_score'].tail(7).mean() if df is not None and 'vader_score' in df.columns else 0.0
            trend    = "Bullish" if df is not None and df['Close'].iloc[-1] > df['Close'].iloc[-7] else "Neutral"

            ai_text = get_ai_independent_forecast(symbol, price, avg_sent, trend)
            # ML Model Prediction with Live Data
            raw_model_pred, _ = get_prediction_dynamically(symbol, live_data=live)
            model_pred = to_float(raw_model_pred)

            st.markdown(
                f'<div class="ai-card"><div class="ai-header">🧠 AI Research</div>{ai_text}</div>',
                unsafe_allow_html=True
            )

            l, r = st.columns(2)
            l.metric("🤖 ML Model Target", f"{model_pred:.2f} BDT", delta=f"{model_pred - price:.2f} BDT")

            ai_match = re.search(r"AI_TARGET:\s*([\d,.]+)", ai_text)
            if ai_match:
                ai_val = float(ai_match.group(1).replace(',', ''))
                r.metric("🧠 AI Outlook Target", f"{ai_val:.2f} BDT", delta=f"{ai_val - price:.2f} BDT")
    else:
        st.warning("Could not determine current price. Please sync live data first.")