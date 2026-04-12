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
# FIX: uses importlib so both GP and BRACBANK route correctly
# -----------------------------
def get_prediction_dynamically(symbol, live_data=None):
    try:
        module_map = {
            "GP":       "interference.predict_gp",
            "BRACBANK": "interference.predict_brac_bank",
        }
        module_name = module_map.get(symbol.upper(), "interference.predict_gp")
        module = importlib.import_module(module_name)
        importlib.reload(module)
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
# DSE LIVE SCRAPER — MULTI-SOURCE FALLBACK CHAIN
# FIX: bdshare has no 'open' column — use ycp as Open approximation
# FIX: try multiple sources if bdshare/dsebd blocks the server IP
# -----------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Referer": "https://www.dsebd.org/",
}

def _parse_share_price_table(html: str, symbol: str) -> dict:
    soup  = BeautifulSoup(html, "html.parser")
    table = soup.find("table", {"class": lambda c: c and "shares-table" in c})
    if not table:
        table = soup.find("table")
    if not table:
        return {}
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 11:
            continue
        if cols[1].get_text(strip=True).upper() == symbol.upper():
            ltp    = to_float(cols[2].get_text(strip=True))
            high   = to_float(cols[3].get_text(strip=True))
            low    = to_float(cols[4].get_text(strip=True))
            closep = to_float(cols[5].get_text(strip=True))
            ycp    = to_float(cols[6].get_text(strip=True))
            volume = to_float(cols[10].get_text(strip=True))
            close  = ltp if ltp > 0 else closep
            return {"Open": ycp, "High": high, "Low": low,
                    "Close": close, "Volume": volume, "YCP": ycp}
    return {}

def fetch_live_dse(symbol: str) -> dict | None:
    session = requests.Session()
    session.headers.update(_HEADERS)

    # Source 1: bdshare library
    try:
        from bdshare import get_current_trade_data
        df_live = get_current_trade_data(symbol.upper())
        if not df_live.empty:
            row  = df_live.iloc[0]
            ltp  = to_float(row.get("ltp",  0))
            ycp  = to_float(row.get("ycp",  0))
            # FIX: bdshare has no 'open' column → use ycp as Open
            live_close = ltp if ltp > 0 else to_float(row.get("close", 0))
            if live_close > 0:
                return {
                    "Open":   ycp,
                    "High":   to_float(row.get("high",   0)),
                    "Low":    to_float(row.get("low",    0)),
                    "Close":  live_close,
                    "Volume": to_float(row.get("volume", 0)),
                    "Time":   datetime.now().strftime("%I:%M %p"),
                    "Source": "bdshare"
                }
    except Exception:
        pass

    # Source 2: direct HTML scrape of dsebd.org (primary + alt domain)
    for url in [
        "https://www.dsebd.org/latest_share_price_scroll_l.php",
        "https://dsebd.org/latest_share_price_scroll_l.php",
        "https://dsebd.com.bd/latest_share_price_scroll_l.php",
    ]:
        try:
            resp = session.get(url, timeout=12)
            if resp.status_code == 200 and len(resp.content) > 500:
                data = _parse_share_price_table(resp.text, symbol)
                if data.get("Close", 0) > 0:
                    data["Time"]   = datetime.now().strftime("%I:%M %p")
                    data["Source"] = url
                    return data
        except Exception:
            continue

    # Source 3: plain-text quotes.txt
    for url in [
        "https://dsebd.org/datafile/quotes.txt",
        "http://dsebd.org/datafile/quotes.txt",
    ]:
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and symbol.upper() in resp.text.upper():
                for line in resp.text.splitlines():
                    parts = line.split()
                    if parts and parts[0].upper() == symbol.upper() and len(parts) >= 6:
                        ltp  = to_float(parts[1])
                        ycp  = to_float(parts[5]) if len(parts) > 5 else 0.0
                        if ltp > 0:
                            return {
                                "Open":   ycp,
                                "High":   to_float(parts[2]),
                                "Low":    to_float(parts[3]),
                                "Close":  ltp,
                                "Volume": to_float(parts[6]) if len(parts) > 6 else 0.0,
                                "Time":   datetime.now().strftime("%I:%M %p"),
                                "Source": "quotes.txt"
                            }
        except Exception:
            continue

    return None  # All sources failed

# -----------------------------
# AI RESEARCH FUNCTION
# (unchanged from old working version)
# -----------------------------
def get_ai_independent_forecast(symbol, price, sentiment_avg, weekly_trend):
    try:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        prompt = f"""
        Act as a Senior Market Researcher for the Dhaka Stock Exchange.

        STRICT MISSION: Provide an independent numerical forecast for {symbol} tomorrow.
        DO NOT guess what other models say. Use these inputs:
        - Current Price (LTP): {price} BDT
        - 7-Day Sentiment: {sentiment_avg:.2f}
        - Recent Trend: {weekly_trend}

        OUTPUT REQUIREMENTS:
        1. Professional market analysis.
        2. 'Reasoning:' section.
        3. 'Numerical:' section (MUST include 'AI_TARGET: XXX.X BDT').
        """
        data = {
            "model": "mistralai/mixtral-8x7b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
        response = requests.post(url, headers=headers, json=data)
        result   = response.json()
        return result["choices"][0]["message"]["content"] if "choices" in result else "AI Research Offline."
    except Exception as e:
        return f"AI Error: {str(e)}"

# -----------------------------
# UI MAIN
# -----------------------------
st.title("📈 DSE Agentic Stock Predictor")
st.markdown("---")

# FIX: auto-sync on ticker change (restored from old code)
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = None

col1, col2 = st.columns([1, 2])
symbol    = col1.selectbox("Select Ticker", ["GP", "BRACBANK"])
data_path = f"data/{symbol.lower()}_final_dataset.csv"

# Auto-sync when ticker changes (restores old behavior)
if st.session_state.last_ticker != symbol:
    st.session_state.last_ticker = symbol
    auto = fetch_live_dse(symbol)
    if auto:
        st.session_state[f"{symbol}_live"] = auto

with col1:
    st.subheader("Controls")

    # 1. Manual Sync Button
    if st.button("⚡ Sync Live Price", type="primary", use_container_width=True):
        with st.spinner(f"Fetching live data for {symbol}..."):
            live_data = fetch_live_dse(symbol)
        if live_data:
            st.session_state[f"{symbol}_live"] = live_data
            st.success(
                f"✅ {symbol} @ {live_data['Close']:.2f} BDT "
                f"(H:{live_data['High']:.2f} / L:{live_data['Low']:.2f}) "
                f"— {live_data['Time']} via {live_data.get('Source','DSE')}"
            )
        else:
            st.warning(
                "⚠️ All DSE sources failed (IP block likely). "
                "Enter price manually below."
            )

    # 2. Manual price entry (guaranteed fallback)
    with st.expander("✏️ Enter Price Manually (if sync fails)"):
        m_close  = st.number_input("Close / LTP (BDT) *", min_value=0.0, step=0.10, format="%.2f", key=f"mc_{symbol}")
        m_open   = st.number_input("Open (BDT)",          min_value=0.0, step=0.10, format="%.2f", key=f"mo_{symbol}")
        m_high   = st.number_input("High (BDT)",          min_value=0.0, step=0.10, format="%.2f", key=f"mh_{symbol}")
        m_low    = st.number_input("Low (BDT)",           min_value=0.0, step=0.10, format="%.2f", key=f"ml_{symbol}")
        m_volume = st.number_input("Volume",              min_value=0.0, step=1.0,  format="%.0f", key=f"mv_{symbol}")
        if st.button("💾 Use Manual Prices", use_container_width=True):
            if m_close > 0:
                st.session_state[f"{symbol}_live"] = {
                    "Open":   m_open   if m_open  > 0 else m_close,
                    "High":   m_high   if m_high  > 0 else m_close,
                    "Low":    m_low    if m_low   > 0 else m_close,
                    "Close":  m_close,
                    "Volume": m_volume,
                    "Time":   datetime.now().strftime("%I:%M %p"),
                    "Source": "Manual Entry"
                }
                st.success(f"✅ Manual price saved: {m_close:.2f} BDT")
            else:
                st.error("Close / LTP must be > 0")

    # Show active price source
    live = st.session_state.get(f"{symbol}_live")
    if live:
        st.info(f"📌 Active: **{live['Close']:.2f} BDT** ({live.get('Source','DSE')})")

    # 3. Predict Button
    # FIX: passes live_data to prediction so model uses current price, not CSV
    if st.button("🔮 Predict Tomorrow", use_container_width=True):
        with st.spinner("Calculating predictions..."):
            live = st.session_state.get(f"{symbol}_live")
            raw_pred, csv_close = get_prediction_dynamically(symbol, live_data=live)

            pred_value    = to_float(raw_pred) if not isinstance(raw_pred, str) else 0.0
            current_close = to_float(live["Close"] if live else csv_close)
            diff          = pred_value - current_close

            st.metric("Last Close Used", f"{current_close:.2f} BDT")
            st.metric(
                "Model Prediction",
                f"{pred_value:.2f} BDT",
                delta=f"{diff:.2f} BDT" if pred_value > 0 else None
            )
            if isinstance(raw_pred, str) and raw_pred.startswith("Error"):
                st.error(raw_pred)

# 4. Price History Chart
with col2:
    st.subheader("Price History")
    df   = None
    live = st.session_state.get(f"{symbol}_live", {})

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        if live:
            disp_time = f"Last Synced: {live.get('Time')} ({live.get('Source','DSE')})"
        else:
            disp_time = f"Last Updated: {df['Date'].iloc[-1]} (CSV)"
        st.caption(f"📅 {disp_time}")

        vals = [to_float(live.get(k, df[k].iloc[-1])) for k in ['Open', 'High', 'Low', 'Close', 'Volume']]
        m1, m2, m3, m4, m5 = st.columns(5)
        for m, label, val in zip([m1,m2,m3,m4,m5], ["Open","High","Low","Close","Volume"], vals):
            m.metric(label, f"{val:.2f}" if label != "Volume" else f"{int(val):,}")

        chart_df = df.tail(30).copy()
        if live:
            chart_df = pd.concat([
                chart_df,
                pd.DataFrame([{"Date": "Live", "Open": vals[0], "High": vals[1],
                                "Low": vals[2], "Close": vals[3]}])
            ], ignore_index=True)

        fig = go.Figure(data=[go.Candlestick(
            x=chart_df['Date'], open=chart_df['Open'],
            high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close']
        )])
        fig.update_layout(
            template="plotly_dark", height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Historical data CSV not found for this ticker.")

# 5. AI Research Section (restored full old structure)
st.markdown("---")
if st.button("🤖 Get AI Independent Research & Prediction", use_container_width=True):
    live  = st.session_state.get(f"{symbol}_live", {})
    price = to_float(live.get("Close", 0))

    if price > 0:
        with st.spinner("AI performing independent research..."):
            if df is not None:
                avg_sent     = df['vader_score'].tail(7).mean() if 'vader_score' in df.columns else 0.0
                weekly_trend = "Bullish" if df['Close'].iloc[-1] > df['Close'].iloc[-7] else "Bearish"
            else:
                avg_sent, weekly_trend = 0.0, "Neutral"

            ai_raw_text    = get_ai_independent_forecast(symbol, price, avg_sent, weekly_trend)
            raw_model_pred, _ = get_prediction_dynamically(symbol, live_data=live)
            model_pred     = to_float(raw_model_pred)

            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-header">🧠 AI Independent Research</div>
                <div class="ai-sub">Advanced market analysis persona.</div>
                <div style="font-size:15px;line-height:1.6;">{ai_raw_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("📊 Comparative Forecast Summary")
            left, right = st.columns(2)

            with left:
                st.markdown("### 🤖 Your ML Model")
                m_diff = model_pred - price
                st.metric("Model Target", f"{model_pred:.2f} BDT",
                          delta=f"{m_diff:.2f} BDT" if model_pred > 0 else None)

            with right:
                st.markdown("### 🧠 AI Independent Outlook")
                ai_match = re.search(r"AI_TARGET:\s*([\d,.]+)", ai_raw_text)
                if ai_match:
                    ai_val = float(ai_match.group(1).replace(",", ""))
                    a_diff = ai_val - price
                    st.metric("AI Target", f"{ai_val:.2f} BDT", delta=f"{a_diff:.2f} BDT")
                else:
                    st.write(f"News Sentiment: **{avg_sent:.2f}**")
                    st.warning("Could not parse numerical AI target from text.")
    else:
        st.error("Please sync live data first (or enter price manually).")