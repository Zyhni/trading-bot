"""
frontend_realtime_tradingview.py

Frontend Streamlit:
- Menampilkan TradingView widget (chart interaktif)
- Mengambil historical bars panjang dari TradingView via tvDatafeed (preferred)
- Fallback ke yfinance jika tvDatafeed tidak tersedia
- Menghitung indikator (sma/ema/rsi/atr/..) untuk analisis AI
- Menyimpan image chart internal untuk AI Vision
- Auto-load saat app start / saat symbol berubah (tanpa tombol Load)
"""

import os
import random
import string
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.io as pio

from src.indicators import add_all_indicators
from src.ai_analyzer_ferdev import get_ferdev_analyzer
from economic_calendar import EconCalendarLoader
from news import NewsLoader


# ==========================
# OPTIONAL IMPORT: tvDatafeed
# ==========================
HAS_TV = False
TvDatafeed = None
Interval = None

try:
    from tvDatafeed import TvDatafeed, Interval  # <--- FIX
    HAS_TV = True
except Exception:
    HAS_TV = False


# ==========================
# OPTIONAL IMPORT: yfinance
# ==========================
try:
    import yfinance as yf
except Exception:
    yf = None


# ==========================
# CONFIG
# ==========================
class Config:
    TIMEFRAMES = ["M15", "M30", "H1"]
    N_BARS = 5000
    FERDEV_API_KEY = "key_17ToTo"


# ==========================
# TRADINGVIEW WIDGET HTML
# ==========================
def create_tradingview_widget_html(tv_symbol: str, tv_interval: str = "60", width="100%", height=720):
    uid = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    html = f"""
    <div class="tradingview-widget-container">
      <div id="tv-widget-{uid}"></div>
    </div>

    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
      "width": "{width}",
      "height": {height},
      "symbol": "{tv_symbol}",
      "interval": "{tv_interval}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#0e1117",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tv-widget-{uid}"
    }});
    </script>
    """
    return html


# ==========================
# DATA LOADER
# ==========================
class TradingViewDataLoader:
    """
    Load data dari tvDatafeed (TradingView unofficial).
    Jika tvDatafeed tidak tersedia -> fallback yfinance.
    """

    TV_CANDIDATES = {
        "XAUUSD": [("XAUUSD", "OANDA"), ("XAUUSD", "FX_IDC"), ("GC1!", "COMEX"), ("GLD", "AMEX")],
        "EURUSD": [("EURUSD", "OANDA"), ("EURUSD", "FX_IDC")],
        "GBPUSD": [("GBPUSD", "OANDA"), ("GBPUSD", "FX_IDC")],
        "USDJPY": [("USDJPY", "OANDA"), ("USDJPY", "FX_IDC")],
    }

    YF_CANDIDATES = {
        "XAUUSD": ["XAUUSD=X", "GC=F", "GLD"],
        "EURUSD": ["EURUSD=X"],
        "GBPUSD": ["GBPUSD=X"],
        "USDJPY": ["USDJPY=X", "JPY=X"],
    }

    def __init__(self, symbol="XAUUSD", tv_username=None, tv_password=None):
        self.symbol = symbol
        self.tv_username = tv_username
        self.tv_password = tv_password

    def _tv_interval_map(self):
        """
        Interval map hanya dibuat kalau tvDatafeed tersedia.
        Ini FIX utama untuk error Interval not defined.
        """
        if not HAS_TV or Interval is None:
            return {}

        return {
            "M15": (Interval.in_15_minute, "15"),
            "M30": (Interval.in_30_minute, "30"),
            "H1":  (Interval.in_1_hour, "60"),
        }

    def _try_tv_get(self, sym: str, exch: str, interval_enum, n_bars: int):
        if not HAS_TV:
            return None

        try:
            if self.tv_username and self.tv_password:
                tv = TvDatafeed(self.tv_username, self.tv_password)
            else:
                tv = TvDatafeed()

            df = tv.get_hist(symbol=sym, exchange=exch, interval=interval_enum, n_bars=n_bars)

            if df is None or df.empty:
                return None

            df = df.reset_index()

            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "time"})
            elif "time" not in df.columns:
                df = df.rename(columns={df.columns[0]: "time"})

            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

            if "volume" in df.columns and "tick_volume" not in df.columns:
                df = df.rename(columns={"volume": "tick_volume"})

            return df

        except Exception:
            return None

    def _try_yf_get(self, ticker: str, interval: str, period: str):
        if yf is None:
            return None

        try:
            df = yf.download(tickers=ticker, interval=interval, period=period, progress=False)

            if df is None or df.empty:
                return None

            df = df.reset_index()

            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "time"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "time"})

            rename_map = {}
            for col in df.columns:
                if col.lower() == "open":
                    rename_map[col] = "open"
                elif col.lower() == "high":
                    rename_map[col] = "high"
                elif col.lower() == "low":
                    rename_map[col] = "low"
                elif col.lower() == "close":
                    rename_map[col] = "close"
                elif col.lower() == "volume":
                    rename_map[col] = "tick_volume"

            df = df.rename(columns=rename_map)
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)

            return df

        except Exception:
            return None

    def load_all_timeframes(self, n_bars: int = Config.N_BARS):
        frames = {}
        used_source = None

        # ==========================
        # 1) TRY tvDatafeed
        # ==========================
        if HAS_TV:
            interval_map = self._tv_interval_map()
            candidates = self.TV_CANDIDATES.get(self.symbol, [(self.symbol, "OANDA")])

            for sym_tv, exch in candidates:
                ok = True
                temp = {}

                for tf in Config.TIMEFRAMES:
                    interval_enum, _ = interval_map.get(tf, (None, None))
                    if interval_enum is None:
                        ok = False
                        break

                    df = self._try_tv_get(sym_tv, exch, interval_enum, n_bars)

                    if df is None or df.empty:
                        ok = False
                        break

                    temp[tf] = df

                if ok:
                    frames = temp
                    used_source = f"tv:{sym_tv}@{exch}"
                    break

        # ==========================
        # 2) FALLBACK yfinance
        # ==========================
        if not frames:
            yf_cands = self.YF_CANDIDATES.get(self.symbol, [self.symbol])

            interval_map = {"M15": "15m", "M30": "30m", "H1": "60m"}

            for cand in yf_cands:
                ok = True
                temp = {}

                for tf in Config.TIMEFRAMES:
                    interval = interval_map.get(tf, "60m")
                    period = "60d" if interval in ["15m", "30m"] else "730d"

                    df = self._try_yf_get(cand, interval, period)

                    if df is None or df.empty:
                        ok = False
                        break

                    temp[tf] = df

                if ok:
                    frames = temp
                    used_source = f"yf:{cand}"
                    break

        # ==========================
        # 3) STILL FAIL -> ERROR
        # ==========================
        if not frames:
            raise Exception(
                f"No intraday data available for symbol {self.symbol}. "
                f"tvDatafeed={HAS_TV}, yfinance={'OK' if yf else 'MISSING'}"
            )

        # ==========================
        # INDICATORS
        # ==========================
        for tf, df in frames.items():
            try:
                frames[tf] = add_all_indicators(df)
            except Exception:
                frames[tf] = df

        # ==========================
        # tick_info
        # ==========================
        latest_close = None
        for tf in ["H1", "M30", "M15"]:
            if tf in frames and not frames[tf].empty:
                latest_close = frames[tf].iloc[-1].get("close")
                if latest_close is not None:
                    break

        if latest_close is None:
            latest_close = 0.0

        tick_info = {
            "time": datetime.now(),
            "bid": float(latest_close),
            "ask": float(latest_close),
            "last": float(latest_close),
            "volume": float(frames["M15"].iloc[-1].get("tick_volume", 0)) if "M15" in frames else 0,
        }

        return {
            "data": frames,
            "tick_info": tick_info,
            "spread_point": None,
            "timestamp": datetime.now(),
            "source": used_source,
        }

    def tradingview_widget_symbol(self):
        mapping = {
            "XAUUSD": "OANDA:XAUUSD",
            "EURUSD": "OANDA:EURUSD",
            "GBPUSD": "OANDA:GBPUSD",
            "USDJPY": "OANDA:USDJPY",
        }
        return mapping.get(self.symbol, f"OANDA:{self.symbol}")


# ==========================
# AI ANALYZER WRAPPER
# ==========================
class AIChartAnalyzerFerdev:
    def __init__(self, api_key=None):
        self.analyzer = get_ferdev_analyzer(api_key or Config.FERDEV_API_KEY)

    def analyze_chart(self, chart_image_path: str, market_context: str = ""):
        try:
            return self.analyzer.analyze_chart(
                image_path=chart_image_path,
                market_context=market_context
            )
        except Exception as e:
            return {
                "success": False,
                "analysis": f"Error: {e}",
                "recommendation": "HOLD",
                "confidence": 0
            }


# ==========================
# BUILD MARKET CONTEXT
# ==========================
def build_market_context(symbol: str, market_data: dict) -> str:
    lines = []
    tick = market_data.get("tick_info", {})

    lines.append(f"SYMBOL: {symbol}")
    lines.append(f"TIME: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if tick:
        lines.append(f"PRICE: Bid={tick.get('bid',0):.2f}, Ask={tick.get('ask',0):.2f}")

    lines.append("")
    lines.append("TECHNICAL SUMMARY:")

    for tf, df in market_data["data"].items():
        if df is None or df.empty:
            continue

        last = df.iloc[-1]

        close = last.get("close", None)
        rsi = last.get("rsi_14", None)
        atr = last.get("atr_14", None)
        sma = last.get("sma_20", None)
        ema = last.get("ema_20", None)

        lines.append(
            f"{tf}: close={close:.2f} | RSI14={rsi:.2f} | ATR14={atr:.4f} | SMA20={sma:.2f} | EMA20={ema:.2f}"
        )

    # ==========================
    # NEWS
    # ==========================
    lines.append("")
    lines.append("LATEST NEWS:")

    try:
        news_loader = NewsLoader()
        news_items = news_loader.fetch_normalized_news(
            query=f"{symbol} OR gold OR USD OR forex OR FED OR inflation",
            page_size=5
        )

        if news_items:
            for n in news_items[:5]:
                title = n.get("headline", "No title")
                src = n.get("source", "unknown")
                ts = n.get("timestamp", "")
                summary = n.get("summary", "")

                lines.append(f"- {title} ({src}) [{ts}]")
                if summary:
                    lines.append(f"  Summary: {summary}")

        else:
            lines.append("- No major news found.")

    except Exception as e:
        lines.append(f"- News loader error: {e}")

    # ==========================
    # ECONOMIC CALENDAR
    # ==========================
    lines.append("")
    lines.append("ECONOMIC CALENDAR:")

    try:
        econ_loader = EconCalendarLoader(output_type="df")
        econ_df = econ_loader.fetch()

        if econ_df is not None and not econ_df.empty:
            econ_df = econ_df.head(8)

            for _, row in econ_df.iterrows():
                event = row.get("event", "")
                country = row.get("country", "")
                impact = row.get("impact", "")
                actual = row.get("actual", "")
                forecast = row.get("forecast", "")
                previous = row.get("previous", "")
                date = row.get("date", "")

                lines.append(
                    f"- {date} | {country} | {impact} | {event} | actual={actual}, forecast={forecast}, prev={previous}"
                )
        else:
            lines.append("- No economic events found.")

    except Exception as e:
        lines.append(f"- Econ calendar loader error: {e}")

    return "\n".join(lines)

def build_debug_context_with_result(symbol: str, market_data: dict, ai_result: dict | None) -> str:
    ctx = build_market_context(symbol, market_data)

    if ai_result:
        rec = ai_result.get("recommendation", "HOLD")
        conf = ai_result.get("confidence", 0)
        analysis = ai_result.get("analysis", "")

        final_block = f"""
=========================
FINAL AI RESULT
=========================
RECOMMENDATION: {rec}
CONFIDENCE: {conf}%

SHORT ANALYSIS:
{analysis[:1200]}
"""
    else:
        final_block = """
=========================
FINAL AI RESULT
=========================
RECOMMENDATION: NONE (AI not run yet)
CONFIDENCE: 0%
"""

    return ctx + "\n" + final_block
# ==========================
# STREAMLIT APP
# ==========================
def main():
    st.set_page_config(page_title="Bot Trading AI Dashboard", layout="wide")

    st.markdown(
        "<style>.main-header{font-size:2.2rem;color:#FFD700;text-align:center;margin-bottom:1rem}</style>",
        unsafe_allow_html=True
    )
    st.markdown('<h1 class="main-header">📊 Bot Trading AI Analysis</h1>', unsafe_allow_html=True)

    # ==========================
    # SESSION STATE DEFAULTS
    # ==========================
    if "market_data" not in st.session_state:
        st.session_state.market_data = None
    if "chart_image_path" not in st.session_state:
        st.session_state.chart_image_path = None
    if "last_symbol" not in st.session_state:
        st.session_state.last_symbol = None
    if "ai_analysis" not in st.session_state:
        st.session_state.ai_analysis = None

    # ==========================
    # SIDEBAR
    # ==========================
    with st.sidebar:
        st.header("⚙ Settings")

        symbol = st.selectbox("Symbol", ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"], index=0)
        chart_type = st.radio("Chart Type", ["Single Timeframe"], index=0)

        if chart_type == "Single Timeframe":
            selected_tf = st.selectbox("Timeframe", Config.TIMEFRAMES, index=2)
        else:
            selected_tf = "Multi"

        st.markdown("---")

    # ==========================
    # AUTO LOAD DATA
    # ==========================
    need_load = False
    if st.session_state.market_data is None:
        need_load = True
    if st.session_state.last_symbol != symbol:
        need_load = True

    if need_load:
        st.session_state.ai_analysis = None
        st.session_state.chart_image_path = None

        loader = TradingViewDataLoader(
            symbol=symbol,
        )

        with st.spinner(f"Loading data {symbol} ..."):
            try:
                md = loader.load_all_timeframes(n_bars=Config.N_BARS)
                st.session_state.market_data = md
                st.session_state.last_symbol = symbol
                st.success(f"Data loaded successfully")
            except Exception as e:
                st.session_state.market_data = None
                st.error(f"Error loading data: {e}")

    # ==========================
    # TABS
    # ==========================
    tab1, tab2, tab3 = st.tabs(["📉 TradingView Chart", "📊 Market Data", "🤖 AI Analysis"])

    # ==========================
    # TAB 1: TradingView Chart
    # ==========================
    with tab1:
        if not st.session_state.market_data:
            st.warning("Market data belum tersedia.")
        else:
            tv_symbol = TradingViewDataLoader(symbol).tradingview_widget_symbol()

            if selected_tf == "H1":
                tv_interval = "60"
            elif selected_tf == "M30":
                tv_interval = "30"
            else:
                tv_interval = "15"

            st.markdown("### TradingView Chart")
            tv_html = create_tradingview_widget_html(tv_symbol, tv_interval, height=720)
            components.html(tv_html, height=720, scrolling=False)

            tick = st.session_state.market_data.get("tick_info", {})

            c1, c2, c3 = st.columns(3)
            c1.metric("Bid", f"{tick.get('bid',0):.2f}")
            c2.metric("Ask", f"{tick.get('ask',0):.2f}")
            c3.metric("Source", st.session_state.market_data.get("source", "-"))

    # ==========================
    # TAB 2: Market Data
    # ==========================
    with tab2:
        if not st.session_state.market_data:
            st.warning("Market data belum tersedia.")
        else:
            st.subheader("Latest Technical Indicators")

            data = st.session_state.market_data["data"]
            rows = []

            for tf in Config.TIMEFRAMES:
                df = data.get(tf)
                if df is None or df.empty:
                    continue

                last = df.iloc[-1]

                rows.append({
                    "TF": tf,
                    "Time": str(last.get("time")),
                    "Close": round(float(last.get("close", 0)), 4),
                    "SMA20": round(float(last.get("sma_20", 0)), 4) if pd.notna(last.get("sma_20", None)) else None,
                    "EMA20": round(float(last.get("ema_20", 0)), 4) if pd.notna(last.get("ema_20", None)) else None,
                    "RSI14": round(float(last.get("rsi_14", 0)), 2) if pd.notna(last.get("rsi_14", None)) else None,
                    "ATR14": round(float(last.get("atr_14", 0)), 4) if pd.notna(last.get("atr_14", None)) else None,
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("---")
            st.subheader("News Preview")

            try:
                news_loader = NewsLoader()
                news_items = news_loader.fetch_normalized_news(query=f"{symbol} OR gold OR USD", page_size=5)

                if news_items:
                    for item in news_items[:5]:
                        st.markdown(f"**{item.get('headline','No title')}**")
                        st.caption(f"{item.get('source','-')} | {item.get('timestamp','-')}")
                        if item.get("summary"):
                            st.write(item["summary"])
                        st.divider()
                else:
                    st.info("No news available.")

            except Exception as e:
                st.warning(f"News error: {e}")

            st.markdown("---")
            st.subheader("Economic Calendar Preview")

            try:
                econ_loader = EconCalendarLoader(output_type="df")
                econ_df = econ_loader.fetch()

                if econ_df is not None and not econ_df.empty:
                    st.dataframe(econ_df.head(10), use_container_width=True)
                else:
                    st.info("No economic calendar data.")

            except Exception as e:
                st.warning(f"Econ calendar error: {e}")

    # ==========================
    # TAB 3: AI Analysis
    # ==========================
    with tab3:
        st.subheader("AI Analysis")

        if not st.session_state.market_data:
            st.warning("No market data.")
        else:
            df_for_img = st.session_state.market_data["data"].get("H1")

            img_path = None
            if df_for_img is not None and not df_for_img.empty:
                try:
                    fig = go.Figure(data=[
                        go.Candlestick(
                            x=df_for_img["time"],
                            open=df_for_img["open"],
                            high=df_for_img["high"],
                            low=df_for_img["low"],
                            close=df_for_img["close"],
                        )
                    ])

                    if "sma_20" in df_for_img.columns:
                        fig.add_trace(go.Scatter(
                            x=df_for_img["time"], y=df_for_img["sma_20"],
                            name="SMA20", line=dict(width=1)
                        ))

                    if "ema_20" in df_for_img.columns:
                        fig.add_trace(go.Scatter(
                            x=df_for_img["time"], y=df_for_img["ema_20"],
                            name="EMA20", line=dict(width=1, dash="dash")
                        ))

                    fig.update_layout(template="plotly_dark", height=720)
                    st.plotly_chart(fig, use_container_width=True)
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    pio.write_image(fig, tmp.name, width=1600, height=900, scale=2)
                    img_path = tmp.name
                    st.session_state.chart_image_path = img_path

                except Exception as e:
                    st.warning(f"Failed generate chart image: {e}")

            if st.session_state.ai_analysis is None:
                with st.spinner("AI analyzing..."):
                    analyzer = AIChartAnalyzerFerdev()

                    market_context = build_market_context(symbol, st.session_state.market_data)

                    result = analyzer.analyze_chart(
                        chart_image_path=img_path or "",
                        market_context=market_context
                    )

                    st.session_state.ai_analysis = result

            if st.session_state.ai_analysis:
                res = st.session_state.ai_analysis
                st.markdown("### Result")

                st.write(res.get("analysis", res.get("raw_response", "No response")))
                st.markdown(f"**Recommendation:** `{res.get('recommendation','HOLD')}`")
                st.markdown(f"**Confidence:** `{res.get('confidence',0)}%`")

                if "market_context_used" in res:
                    st.markdown("### Market Context Sent to AI")
                    st.code(res["market_context_used"])

            with st.expander("DEBUG: Market Context + FINAL RESULT"):
                st.code(
                    build_debug_context_with_result(
                        symbol,
                        st.session_state.market_data,
                        st.session_state.ai_analysis
                    )
                )


    st.sidebar.markdown("---")

if __name__ == "__main__":
    main()
