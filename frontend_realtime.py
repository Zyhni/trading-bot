# frontend_realtime_ferdev.py
import os
import base64
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import tempfile

from src.mt5_client import (
    initialize_mt5,
    ensure_symbol,
    get_rates,
    get_tick_info,
    get_spread_point,
    shutdown_mt5,
    DEFAULT_SYMBOL,
)
from src.indicators import add_all_indicators
from economic_calendar import EconCalendarLoader
from news import NewsLoader
from src.ai_analyzer_ferdev import get_ferdev_analyzer


# === CONFIGURATION ===
class Config:
    TIMEFRAMES = ["M15", "M30", "H1"]
    CHART_PERIODS = {
        "M15": 100,
        "M30": 80,
        "H1": 60
    }
    FIBONACCI_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # API Configuration
    FERDEV_API_KEY = "key_17ToTo"


# === CHART UTILITIES ===
def add_technical_indicators(fig, df, include_sma=True, include_ema=True, include_rsi=True):
    """Tambahkan indikator teknikal ke chart"""
    # Tambahkan SMA 20
    if include_sma and 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=1.5)
        ))
    
    # Tambahkan EMA 20
    if include_ema and 'ema_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['ema_20'],
            mode='lines',
            name='EMA 20',
            line=dict(color='cyan', width=1.5, dash='dash')
        ))
    
    # Tambahkan RSI subplot
    if include_rsi and 'rsi_14' in df.columns:
        # Buat subplot untuk RSI
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['rsi_14'],
            mode='lines',
            name='RSI 14',
            line=dict(color='purple', width=1.5),
            yaxis="y2"
        ))
        
        # Tambahkan garis overbought (70) dan oversold (30)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, yref="y2")
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, yref="y2")
    
    return fig


def create_multi_timeframe_chart(df_m15, df_m30, df_h1):
    """Buat chart dengan 3 timeframe"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('H1 Chart', 'M30 Chart', 'M15 Chart'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # H1 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_h1['time'],
            open=df_h1['open'],
            high=df_h1['high'],
            low=df_h1['low'],
            close=df_h1['close'],
            name='H1'
        ),
        row=1, col=1
    )
    
    # M30 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_m30['time'],
            open=df_m30['open'],
            high=df_m30['high'],
            low=df_m30['low'],
            close=df_m30['close'],
            name='M30'
        ),
        row=2, col=1
    )
    
    # M15 Chart
    fig.add_trace(
        go.Candlestick(
            x=df_m15['time'],
            open=df_m15['open'],
            high=df_m15['high'],
            low=df_m15['low'],
            close=df_m15['close'],
            name='M15'
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Multi Timeframe XAUUSD Analysis',
        height=900,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        template="plotly_dark"
    )
    
    # Update axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=3, col=1)
    
    return fig


def save_chart_as_image(fig, timeframe="H1"):
    """Simpan chart sebagai gambar untuk analisis AI"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        # Untuk analisis AI, kita butuh resolusi tinggi
        pio.write_image(fig, tmp.name, width=1600, height=900, scale=2)
        return tmp.name


# === DATA LOADER ===
class MT5DataLoader:
    def __init__(self, symbol=DEFAULT_SYMBOL):
        self.symbol = symbol
    
    def load_all_timeframes(self):
        """Load data untuk semua timeframe"""
        try:
            initialize_mt5()
            ensure_symbol(self.symbol)
            
            data = {}
            for tf in Config.TIMEFRAMES:
                count = Config.CHART_PERIODS.get(tf, 100)
                df = get_rates(self.symbol, tf, count)
                df = add_all_indicators(df)
                data[tf] = df
            
            tick_info = get_tick_info(self.symbol)
            spread_point = get_spread_point(self.symbol)
            
            shutdown_mt5()
            
            return {
                'data': data,
                'tick_info': tick_info,
                'spread_point': spread_point,
                'timestamp': datetime.now()
            }
        except Exception as e:
            shutdown_mt5()
            raise e


# === AI ANALYZER WITH FERDEV ===
class AIChartAnalyzerFerdev:
    def __init__(self, api_key=None):
        self.analyzer = get_ferdev_analyzer(api_key or Config.FERDEV_API_KEY)
    
    def analyze_chart(self, chart_image_path, market_context=""):
        """Analisis chart menggunakan Ferdev AI Vision"""
        try:
            result = self.analyzer.analyze_chart(
                image_path=chart_image_path,
                market_context=market_context
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "analysis": f"Error in analysis: {str(e)}",
                "recommendation": "HOLD",
                "confidence": 0,
                "levels": {},
                "risk_management": {}
            }
    
    def format_analysis_for_display(self, analysis_result):
        """Format hasil analisis untuk ditampilkan di Streamlit"""
        if not analysis_result.get("success", False):
            return "❌ **Gagal melakukan analisis AI.**\n\nCoba lagi nanti atau gunakan analisis manual."
        
        output = []
        
        # Header
        output.append("## 🤖 **ANALISIS AI VISION**")
        output.append(f"*Timestamp: {analysis_result.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}*")
        output.append("---")
        
        # Analisis utama
        if analysis_result.get("analysis"):
            output.append("### 📊 **ANALISIS UTAMA**")
            output.append(analysis_result["analysis"])
            output.append("")
        
        # Rekomendasi dengan confidence
        recommendation = analysis_result.get("recommendation", "HOLD")
        confidence = analysis_result.get("confidence", 0)
        
        output.append(f"### 🎯 **REKOMENDASI: {recommendation}**")
        
        # Progress bar untuk confidence
        confidence_color = "🟢" if confidence >= 70 else "🟡" if confidence >= 50 else "🔴"
        output.append(f"{confidence_color} **Confidence Level: {confidence}%**")
        
        # Risk management
        risk_mgmt = analysis_result.get("risk_management", {})
        if any(risk_mgmt.values()):
            output.append("### 🛡️ **MANAJEMEN RISIKO**")
            
            if risk_mgmt.get("entry") and risk_mgmt["entry"] != "N/A":
                output.append(f"- **Entry Zone:** {risk_mgmt['entry']}")
            
            if risk_mgmt.get("stop_loss") and risk_mgmt["stop_loss"] != "N/A":
                output.append(f"- **Stop Loss:** {risk_mgmt['stop_loss']}")
            
            if risk_mgmt.get("take_profit") and risk_mgmt["take_profit"] != "N/A":
                output.append(f"- **Take Profit:** {risk_mgmt['take_profit']}")
            
            if risk_mgmt.get("risk_reward") and risk_mgmt["risk_reward"] != "N/A":
                output.append(f"- **Risk/Reward:** {risk_mgmt['risk_reward']}")
        
        return "\n".join(output)


# === STREAMLIT APP ===
def main():
    st.set_page_config(
        page_title="XAUUSD AI Trading System (Ferdev Vision)",
        layout="wide",
        page_icon="🏆"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFA500;
        margin-top: 1rem;
    }
    .ai-response {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FFD700;
        margin: 10px 0;
    }
    .metric-box {
        background-color: #2D2D2D;
        padding: 15px;
        border-radius: 8px;
        margin: 5px;
    }
    .buy-signal {
        background-color: rgba(0, 255, 0, 0.1);
        border-left: 5px solid #00FF00;
        padding: 15px;
        border-radius: 5px;
    }
    .sell-signal {
        background-color: rgba(255, 0, 0, 0.1);
        border-left: 5px solid #FF0000;
        padding: 15px;
        border-radius: 5px;
    }
    .hold-signal {
        background-color: rgba(255, 255, 0, 0.1);
        border-left: 5px solid #FFFF00;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🏆 XAUUSD AI Trading System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">Powered by Ferdev Vision AI</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'market_data' not in st.session_state:
        st.session_state.market_data = None
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    if 'chart_image_path' not in st.session_state:
        st.session_state.chart_image_path = None
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ **Settings**")
        
        # Symbol selection
        symbol = st.selectbox(
            "Symbol",
            ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"],
            index=0
        )
        
        # Chart type
        chart_type = st.radio(
            "Chart Type",
            ["Single Timeframe", "Multi Timeframe"],
            index=0
        )
        
        if chart_type == "Single Timeframe":
            selected_tf = st.selectbox(
                "Timeframe",
                Config.TIMEFRAMES,
                index=2  # Default H1
            )
        else:
            selected_tf = "Multi"
        
        # Analysis options
        st.subheader("📈 **Analysis Options**")
        
        col1, col2 = st.columns(2)
        with col1:
            include_sma = st.checkbox("SMA", value=True)
            include_ema = st.checkbox("EMA", value=True)
        with col2:
            include_rsi = st.checkbox("RSI", value=True)
            include_fibo = st.checkbox("Fibonacci", value=False)
        
        # Data sources
        st.subheader("📊 **Data Sources**")
        include_news = st.checkbox("Include News", value=True)
        include_econ = st.checkbox("Include Economic Calendar", value=True)
        
        st.markdown("---")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 **Load Market Data**", type="primary", use_container_width=True):
                with st.spinner("Loading market data..."):
                    try:
                        loader = MT5DataLoader(symbol)
                        st.session_state.market_data = loader.load_all_timeframes()
                        st.success("✅ Data loaded successfully!")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
        
        with col2:
            if st.button("🗑️ **Clear Analysis**", use_container_width=True):
                st.session_state.ai_analysis = None
                st.session_state.chart_image_path = None
                st.rerun()
        
        st.markdown("---")
        st.caption(f"**AI Engine:** Ferdev Vision API")
        st.caption(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📈 **Chart Analysis**", "📊 **Market Data**", "🤖 **AI Analysis**"])
    
    with tab1:
        if st.session_state.market_data:
            data = st.session_state.market_data['data']
            tick_info = st.session_state.market_data['tick_info']
            
            # Display current price
            if tick_info:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💰 **BID**", f"{tick_info['bid']:.2f}")
                with col2:
                    st.metric("💱 **ASK**", f"{tick_info['ask']:.2f}")
                with col3:
                    st.metric("📏 **SPREAD**", f"{st.session_state.market_data['spread_point']} pips")
                with col4:
                    st.metric("🕒 **TIME**", datetime.now().strftime("%H:%M:%S"))
            
            # Create chart based on selection
            if chart_type == "Single Timeframe" and selected_tf in data:
                df = data[selected_tf]
                
                # Create candlestick chart
                fig = go.Figure(data=[
                    go.Candlestick(
                        x=df['time'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=selected_tf,
                        increasing_line_color='#00FF00',
                        decreasing_line_color='#FF0000'
                    )
                ])
                
                # Add technical indicators
                fig = add_technical_indicators(
                    fig, df, 
                    include_sma=include_sma,
                    include_ema=include_ema,
                    include_rsi=include_rsi
                )
                
                # Add Fibonacci if selected
                if include_fibo and len(df) > 20:
                    # Simple Fibonacci levels
                    recent_high = df['high'].tail(20).max()
                    recent_low = df['low'].tail(20).min()
                    
                    for level in Config.FIBONACCI_LEVELS:
                        fib_price = recent_low + (recent_high - recent_low) * level
                        fig.add_hline(
                            y=fib_price,
                            line_dash="dash",
                            line_color="purple",
                            opacity=0.4,
                            annotation_text=f"Fibo {level*100:.1f}%",
                            annotation_position="right"
                        )
                
                # Update layout
                fig.update_layout(
                    title=f'📊 {symbol} {selected_tf} Chart',
                    xaxis_title="Time",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=600,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                # Add secondary y-axis for RSI if included
                if include_rsi and 'rsi_14' in df.columns:
                    fig.update_layout(
                        yaxis2=dict(
                            title="RSI",
                            overlaying="y",
                            side="right",
                            range=[0, 100]
                        )
                    )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save chart for AI analysis
                chart_path = save_chart_as_image(fig, selected_tf)
                st.session_state.chart_image_path = chart_path
                
                # Show chart info
                with st.expander("📋 **Chart Information**"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Candles", len(df))
                        st.metric("Time Range", f"{df['time'].iloc[0].strftime('%H:%M')} - {df['time'].iloc[-1].strftime('%H:%M')}")
                    with col2:
                        st.metric("Open", f"{df['open'].iloc[-1]:.2f}")
                        st.metric("Close", f"{df['close'].iloc[-1]:.2f}")
                
            elif chart_type == "Multi Timeframe":
                # Create multi-timeframe chart
                if all(tf in data for tf in Config.TIMEFRAMES):
                    fig = create_multi_timeframe_chart(
                        data['M15'],
                        data['M30'],
                        data['H1']
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Save chart for AI analysis
                    chart_path = save_chart_as_image(fig, "Multi")
                    st.session_state.chart_image_path = chart_path
    
    with tab2:
        if st.session_state.market_data:
            data = st.session_state.market_data['data']
            
            # Market overview
            st.subheader("📈 **Market Overview**")
            
            # Create metrics for each timeframe
            timeframe_metrics = []
            for tf in Config.TIMEFRAMES:
                if tf in data:
                    df = data[tf]
                    latest = df.iloc[-1]
                    prev = df.iloc[-2] if len(df) > 1 else latest
                    
                    change = ((latest['close'] - prev['close']) / prev['close']) * 100
                    
                    timeframe_metrics.append({
                        'Timeframe': tf,
                        'Price': latest['close'],
                        'Change %': change,
                        'High': latest['high'],
                        'Low': latest['low'],
                        'Volume': latest.get('tick_volume', 0)
                    })
            
            if timeframe_metrics:
                metrics_df = pd.DataFrame(timeframe_metrics)
                st.dataframe(metrics_df, use_container_width=True)
            
            # Technical indicators table
            st.subheader("📊 **Technical Indicators**")
            
            tech_data = []
            for tf in Config.TIMEFRAMES:
                if tf in data:
                    df = data[tf]
                    latest = df.iloc[-1]
                    
                    tech_data.append({
                        'TF': tf,
                        'SMA 20': latest.get('sma_20', 'N/A'),
                        'EMA 20': latest.get('ema_20', 'N/A'),
                        'RSI 14': latest.get('rsi_14', 'N/A'),
                        'ATR 14': latest.get('atr_14', 'N/A'),
                        'Momentum': latest.get('mom_10', 'N/A'),
                        'Volatility': latest.get('vol_20', 'N/A')
                    })
            
            if tech_data:
                tech_df = pd.DataFrame(tech_data)
                st.dataframe(tech_df, use_container_width=True)
            
            # Load and display news
            if include_news:
                st.subheader("📰 **Latest Financial News**")
                try:
                    news_loader = NewsLoader()
                    news = news_loader.fetch_normalized_news(
                        query="gold OR XAUUSD OR forex OR trading",
                        page_size=4
                    )
                    
                    for item in news:
                        with st.container():
                            st.markdown(f"**📌 {item['headline']}**")
                            st.caption(f"⏰ {item['timestamp']} | 📰 {item['source']}")
                            if item.get('summary'):
                                st.write(item['summary'])
                            st.divider()
                except Exception as e:
                    st.warning(f"⚠️ Could not load news: {e}")
            
            # Load and display economic calendar
            if include_econ:
                st.subheader("📅 **Economic Calendar**")
                try:
                    econ_loader = EconCalendarLoader(output_type="df")
                    econ_df = econ_loader.fetch()
                    
                    if not econ_df.empty:
                        # Filter for today and tomorrow
                        today = datetime.now().date()
                        tomorrow = today + timedelta(days=1)
                        
                        econ_recent = econ_df[
                            (econ_df['date'] >= today) & 
                            (econ_df['date'] <= tomorrow) &
                            (econ_df['importance'] >= 2)
                        ]
                        
                        if not econ_recent.empty:
                            # Format for display
                            display_df = econ_recent[['datetime', 'country', 'event', 'actual', 'forecast', 'importance']].copy()
                            display_df['datetime'] = display_df['datetime'].dt.strftime('%H:%M')
                            display_df['importance'] = display_df['importance'].apply(lambda x: '🔥' * x)
                            
                            st.dataframe(
                                display_df.sort_values('datetime'),
                                use_container_width=True,
                                column_config={
                                    "datetime": "Time",
                                    "country": "Country",
                                    "event": "Event",
                                    "actual": "Actual",
                                    "forecast": "Forecast",
                                    "importance": "Impact"
                                }
                            )
                        else:
                            st.info("ℹ️ No important economic events today or tomorrow")
                    else:
                        st.info("ℹ️ No economic calendar data available")
                except Exception as e:
                    st.warning(f"⚠️ Could not load economic calendar: {e}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🤖 **AI Chart Analysis (Ferdev Vision)**</h2>', unsafe_allow_html=True)
        
        if not st.session_state.market_data:
            st.warning("⚠️ Please load market data first from the sidebar")
            return
        
        if not st.session_state.chart_image_path:
            st.warning("⚠️ Please generate a chart first in the Chart Analysis tab")
            return
        
        # Display chart image
        st.image(st.session_state.chart_image_path, 
                caption="📸 Chart for AI Analysis", 
                use_column_width=True)
        
        # Analysis button
        if st.button("🧠 **Start AI Analysis**", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing the chart and market data..."):
                try:
                    # Prepare market context
                    market_context_lines = []
                    
                    # Current price
                    if st.session_state.market_data.get('tick_info'):
                        tick = st.session_state.market_data['tick_info']
                        market_context_lines.append(f"Current Price: Bid={tick['bid']:.2f}, Ask={tick['ask']:.2f}")
                    
                    # Technical summary
                    tech_summary = []
                    for tf in Config.TIMEFRAMES:
                        if tf in st.session_state.market_data['data']:
                            df = st.session_state.market_data['data'][tf]
                            latest = df.iloc[-1]
                            
                            tech_summary.append(
                                f"{tf}: Close={latest['close']:.2f}, "
                                f"RSI={latest.get('rsi_14', 'N/A'):.2f}, "
                                f"ATR={latest.get('atr_14', 'N/A'):.3f}"
                            )
                    
                    if tech_summary:
                        market_context_lines.append("Technical Indicators:")
                        market_context_lines.extend(tech_summary)
                    
                    market_context = "\n".join(market_context_lines)
                    
                    # Run AI analysis
                    analyzer = AIChartAnalyzerFerdev()
                    result = analyzer.analyze_chart(
                        chart_image_path=st.session_state.chart_image_path,
                        market_context=market_context
                    )
                    
                    st.session_state.ai_analysis = result
                    
                    # Show success message
                    if result.get("success"):
                        st.success("✅ AI analysis completed successfully!")
                    else:
                        st.warning("⚠️ AI analysis completed with warnings")
                    
                except Exception as e:
                    st.error(f"❌ AI Analysis Error: {e}")
        
        # Display AI analysis results
        if st.session_state.ai_analysis:
            analysis = st.session_state.ai_analysis
            
            # Format and display analysis
            formatted_output = analyzer.format_analysis_for_display(analysis)
            st.markdown(formatted_output)
            
            # Recommendation with visual indicator
            recommendation = analysis.get("recommendation", "HOLD")
            confidence = analysis.get("confidence", 0)
            
            st.subheader("🎯 **Trading Signal**")
            
            # Visual signal box
            if recommendation == "BUY":
                st.markdown('<div class="buy-signal">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"### 🟢 **BUY SIGNAL**")
                    st.markdown(f"**Confidence:** {confidence}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggested action
                st.info("**Suggested Action:** Consider opening LONG position with proper risk management")
                
            elif recommendation == "SELL":
                st.markdown('<div class="sell-signal">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"### 🔴 **SELL SIGNAL**")
                    st.markdown(f"**Confidence:** {confidence}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggested action
                st.info("**Suggested Action:** Consider opening SHORT position with proper risk management")
                
            else:
                st.markdown('<div class="hold-signal">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"### 🟡 **HOLD / NEUTRAL**")
                    st.markdown(f"**Confidence:** {confidence}%")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Suggested action
                st.info("**Suggested Action:** Wait for clearer signal or confirmation")
            
            # Risk management summary
            risk_mgmt = analysis.get("risk_management", {})
            if any(risk_mgmt.values()):
                st.subheader("🛡️ **Risk Management Summary**")
                
                cols = st.columns(4)
                metrics = [
                    ("Entry Zone", risk_mgmt.get("entry", "N/A")),
                    ("Stop Loss", risk_mgmt.get("stop_loss", "N/A")),
                    ("Take Profit", risk_mgmt.get("take_profit", "N/A")),
                    ("Risk/Reward", risk_mgmt.get("risk_reward", "N/A"))
                ]
                
                for idx, (label, value) in enumerate(metrics):
                    with cols[idx]:
                        st.metric(label, value)
            
            # Raw analysis (collapsible)
            with st.expander("📄 **View Raw AI Analysis**"):
                if analysis.get("raw_response"):
                    st.text_area("Raw AI Response", 
                                analysis["raw_response"], 
                                height=300)
                else:
                    st.write("No raw response available")


if __name__ == "__main__":
    main()