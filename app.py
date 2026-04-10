import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import requests
import re
from dotenv import load_dotenv
from datetime import datetime
from bdshare import get_current_trade_data

# Note: Ensure you have updated your prediction logic to use the right libraries
from interference.predict_gp import get_prediction

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="DSE Agentic Predictor", layout="wide")
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .ai-card {
        background-color: #111;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00c8ff;
        margin-bottom: 20px;
        color: white;
    }
    .ai-header {
        font-size: 24px;
        font-weight: bold;
        color: white;
        margin-bottom: 10px;
    }
    .ai-sub {
        font-size: 14px;
        color: #aaa;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# INDEPENDENT AI RESEARCH FUNCTION
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
        result = response.json()
        return result["choices"][0]["message"]["content"] if "choices" in result else "AI Research Offline."
    except Exception as e:
        return f"AI Error: {str(e)}"

# -----------------------------
# LIVE DSE DATA SCRAPER
# -----------------------------
def fetch_live_dse(symbol):
    try:
        df_live = get_current_trade_data()
        # More robust symbol matching
        search_sym = symbol.upper().strip()
        row = df_live[df_live['symbol'].str.upper() == search_sym]
        
        if not row.empty:
            res = row.iloc[0]
            ltp = float(str(res.get('ltp', 0)).replace(',', ''))
            ycp = float(str(res.get('ycp', 0)).replace(',', ''))
            raw_open = float(str(res.get('open', 0)).replace(',', ''))
            
            return {
                "Open": raw_open if raw_open > 0 else ycp,
                "High": float(str(res.get('high', 0)).replace(',', '')),
                "Low": float(str(res.get('low', 0)).replace(',', '')),
                "Close": ltp,
                "Volume": float(str(res.get('volume', 0)).replace(',', '')),
                "Time": datetime.now().strftime("%I:%M %p")
            }
    except Exception as e:
        st.error(f"Live Fetch Error: {e}")
        return None
    return None

# -----------------------------
# UI LOGIC
# -----------------------------
st.title("📈 DSE Agentic Stock Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 2])

if "last_ticker" not in st.session_state: st.session_state.last_ticker = None

with col1:
    st.subheader("Controls")
    symbol = st.selectbox("Select Ticker", ["GP", "BRACBANK"])

    if st.session_state.last_ticker != symbol:
        st.session_state.last_ticker = symbol
        auto_data = fetch_live_dse(symbol)
        if auto_data: st.session_state[f"{symbol}_live"] = auto_data

    if st.button("⚡ Sync Live Price", type="primary", use_container_width=True):
        live_data = fetch_live_dse(symbol)
        if live_data:
            st.session_state[f"{symbol}_live"] = live_data
            st.success(f"Synced at {live_data['Time']}")

    if st.button("🔮 Predict Tomorrow", use_container_width=True):
        fresh = fetch_live_dse(symbol)
        if fresh: st.session_state[f"{symbol}_live"] = fresh
        
        # Get raw outputs
        pred_raw, csv_close_raw = get_prediction(symbol)
        live = st.session_state.get(f"{symbol}_live")
        
        # --- TYPE SAFETY FIX ---
        # 1. Flatten pred_value (it's often array([[x]]) )
        if isinstance(pred_raw, (np.ndarray, list)):
            pred_value = float(np.array(pred_raw).flatten()[0])
        else:
            pred_value = float(pred_raw)
            
        # 2. Flatten current_close
        current_close = float(live["Close"] if live else csv_close_raw)
        
        # 3. Calculate difference safely
        diff = pred_value - current_close
        
        st.metric("Last Close", f"{current_close:.2f} BDT")
        st.metric("Model Prediction", f"{pred_value:.2f} BDT", delta=f"{diff:.2f} BDT")

with col2:
    st.subheader("Price History")
    data_path = f"data/{symbol.lower()}_final_dataset.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        live = st.session_state.get(f"{symbol}_live", {})
        
        disp_open = float(live.get("Open", df['Open'].iloc[-1]))
        disp_high = float(live.get("High", df['High'].iloc[-1]))
        disp_low = float(live.get("Low", df['Low'].iloc[-1]))
        disp_close = float(live.get("Close", df['Close'].iloc[-1]))
        disp_vol = float(live.get("Volume", df['Volume'].iloc[-1]))
        disp_time = live.get("Time", "Manual Entry / Last Sync")
        
        st.caption(f"📅 Data Freshness: {disp_time}")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Open", f"{disp_open:.2f}")
        m2.metric("High", f"{disp_high:.2f}")
        m3.metric("Low", f"{disp_low:.2f}")
        m4.metric("Close", f"{disp_close:.2f}")
        m5.metric("Volume", f"{int(disp_vol):,}")

        chart_df = df.tail(30).copy()
        if live:
            live_row = pd.DataFrame([{
                'Date': 'Live', 
                'Open': disp_open, 
                'High': disp_high, 
                'Low': disp_low, 
                'Close': disp_close
            }])
            chart_df = pd.concat([chart_df, live_row], ignore_index=True)

        fig = go.Figure(data=[go.Candlestick(
            x=chart_df['Date'], 
            open=chart_df['Open'], 
            high=chart_df['High'], 
            low=chart_df['Low'], 
            close=chart_df['Close']
        )])
        fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# INDEPENDENT AI RESEARCH SECTION
# -----------------------------
st.markdown("---")
if st.button("🤖 Get AI Independent Research & Prediction", use_container_width=True):
    live = st.session_state.get(f"{symbol}_live", {})
    price = live.get("Close", 0)
    
    if price > 0:
        with st.spinner("AI performing independent research..."):
            # Ensure df is defined from the History section
            if 'df' in locals():
                avg_sent = df['vader_score'].tail(7).mean()
                weekly_trend = "Bullish" if df['Close'].iloc[-1] > df['Close'].iloc[-7] else "Bearish"
            else:
                avg_sent = 0.0
                weekly_trend = "Unknown"

            ai_raw_text = get_ai_independent_forecast(symbol, price, avg_sent, weekly_trend)
            
            # Predict value handling for comparative section
            pred_raw, _ = get_prediction(symbol)
            model_pred = float(np.array(pred_raw).flatten()[0]) if isinstance(pred_raw, (np.ndarray, list)) else float(pred_raw)
            
            # Display Stylish AI Card
            st.markdown(f"""
            <div class="ai-card">
                <div class="ai-header">🧠 AI Independent Research</div>
                <div class="ai-sub">Advanced Market Analysis Persona.</div>
                <div style="font-size: 15px; line-height: 1.6;">
                    {ai_raw_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("📊 Comparative Forecast Summary")
            left, right = st.columns(2)
            
            with left:
                st.markdown("### 🤖 Your ML Model")
                m_diff = model_pred - price
                st.metric("Model Target", f"{model_pred:.2f} BDT", delta=f"{m_diff:.2f} BDT")
                
            with right:
                st.markdown("### 🧠 AI Independent Outlook")
                ai_match = re.search(r"AI_TARGET:\s*([\d\.]+)", ai_raw_text)
                if ai_match:
                    ai_val = float(ai_match.group(1))
                    a_diff = ai_val - price
                    st.metric("AI Target", f"{ai_val:.2f} BDT", delta=f"{a_diff:.2f} BDT")
                else:
                    st.info("Analysis complete. Check 'Numerical' section in text above for specific targets.")
                
    else:
        st.error("Please sync live data first to provide a baseline for the AI.")