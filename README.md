# 🏆 Trading Bot AI System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![MetaTrader5](https://img.shields.io/badge/MetaTrader5-00BFFF?style=for-the-badge&logo=metatrader&logoColor=white)
![Ollama](https://img.shields.io/badge/GPT-7C3AED?style=for-the-badge&logo=GPT&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

Sistem Trading AI untuk XAUUSD yang menggabungkan analisis teknikal, fundamental, berita, dan kecerdasan buatan untuk memberikan rekomendasi trading yang akurat.

## ✨ Fitur Utama

### 🤖 Analisis AI Multi-Modal
- **Visual Chart Analysis**: Analisis gambar chart dengan model LLava 13B
- **Multi-Timeframe Analysis**: Analisis simultan M15, M30, dan H1
- **Fundamental Analysis**: Integrasi kalender ekonomi dan berita real-time
- **Teknikal Analysis**: 20+ indikator teknikal otomatis

### 📈 Data Real-Time
- **Live MT5 Connection**: Data langsung dari broker MetaTrader 5
- **Economic Calendar**: Event ekonomi penting dari TradingEconomics
- **News Aggregator**: Berita terkini dari NewsAPI
- **Technical Indicators**: SMA, EMA, RSI, ATR, Fibonacci, Support/Resistance

### 🎯 Trading Intelligence
- **AI-Powered Signals**: Rekomendasi BUY/SELL/HOLD berbasis AI
- **Risk Management**: Auto-calculate Stop Loss & Take Profit
- **Multi-Timeframe Confirmation**: Konfirmasi sinyal dari 3 timeframe
- **Confidence Scoring**: Skor kepercayaan untuk setiap rekomendasi

## 🚀 Quick Start

### Prerequisites
1. Python 3.9+
2. MetaTrader 5
3. GPT
4. Broker Account

### Installation
1. Clone repository
```bash
git clone https://github.com/zyhni/trading-bot.git
```
```bash
cd trading-bot
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run Program
```bash
streamlit run frontend_realtime.py
```
