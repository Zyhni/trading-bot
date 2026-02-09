# src/ai_analyzer_ferdev.py
import requests
import base64
import logging
from datetime import datetime
from urllib.parse import quote
from typing import Optional, Dict, Any
import tempfile
import os

class FerdevVisionAnalyzer:
    """Menggunakan API Ferdev Vision AI (gratis)"""
    
    def __init__(self, api_key: str = "key_17ToTo"):
        self.api_base = "https://api.ferdev.my.id/ai/vision"
        self.api_key = api_key
        
    def analyze_chart(self, 
                     image_path: str, 
                     market_context: str = "",
                     additional_prompt: str = "",
                     timeout: int = 60) -> Dict[str, Any]:
        """
        Analisis chart trading dengan API Ferdev Vision
        
        Args:
            image_path: Path ke file gambar chart
            market_context: Konteks market saat ini
            additional_prompt: Prompt tambahan
            timeout: Timeout dalam detik
            
        Returns:
            Dict dengan hasil analisis
        """
        try:
            # Step 1: Upload gambar ke CDN (Catbox)
            image_url = self._upload_to_cdn(image_path)
            
            if not image_url:
                # Fallback ke gambar default dari Ferdev
                image_url = "https://cdn.ferdev.my.id/file/a0hdb.png"
                logging.warning("Gagal upload gambar, menggunakan gambar default")
            
            # Step 2: Buat prompt untuk analisis
            prompt = self._create_analysis_prompt(market_context, additional_prompt)
            
            # Step 3: Panggil API Ferdev Vision
            result = self._call_vision_api(image_url, prompt, timeout)
            
            # Step 4: Parse hasil
            return self._parse_api_response(result)
            
        except Exception as e:
            logging.error(f"Error in Ferdev Vision analysis: {e}")
            return {
                "success": False,
                "analysis": f"Error: {str(e)}",
                "recommendation": "HOLD",
                "confidence": 0,
                "levels": {}
            }
    
    def _upload_to_cdn(self, image_path: str) -> Optional[str]:
        """Upload gambar ke CDN Catbox.moe"""
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            files = {'fileToUpload': ('chart.png', image_bytes, 'image/png')}
            data = {'reqtype': 'fileupload'}
            
            response = requests.post(
                'https://catbox.moe/user/api.php',
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                url = response.text.strip()
                if url.startswith('http'):
                    return url
            
        except Exception as e:
            logging.error(f"CDN upload error: {e}")
        
        return None
    
    def _create_analysis_prompt(self, market_context: str, additional_prompt: str) -> str:
        """Buat prompt untuk analisis trading"""
        base_prompt = """ANALISIS CHART TRADING XAUUSD (GOLD) DENGAN DETAIL:

ANDA ADALAH ANALIS TRADING PROFESIONAL. ANALISIS CHART INI DAN BERIKAN:

1. TREND ANALYSIS (30%):
   - Identifikasi trend utama (Bullish/Bearish/Sideways)
   - Konfirmasi dengan multiple timeframe (jika ada)
   - Strength of trend (Strong/Medium/Weak)

2. PATTERN RECOGNITION (25%):
   - Pola candlestick yang terlihat (Doji, Engulfing, Hammer, etc)
   - Chart pattern (Head & Shoulders, Double Top/Bottom, Triangle, etc)
   - Breakout atau breakdown pattern

3. SUPPORT & RESISTANCE (20%):
   - Level support utama (minimal 3 level)
   - Level resistance utama (minimal 3 level)
   - Zone congestion atau accumulation

4. TECHNICAL INDICATORS (15%):
   - RSI kondisi (Overbought/Oversold/Neutral)
   - Momentum analysis
   - Volume analysis (jika terlihat)

5. TRADING RECOMMENDATION (10%):
   - REKOMENDASI: BUY / SELL / HOLD
   - Confidence level (1-100%)
   - Entry zone
   - Stop Loss level
   - Take Profit target
   - Risk/Reward ratio

FORMAT OUTPUT:
📊 TREND: [analisis trend]
🎯 LEVELS: Support: [levels], Resistance: [levels]
📈 PATTERN: [pola teridentifikasi]
⚡ MOMENTUM: [kekuatan momentum]
💰 RECOMMENDATION: [BUY/SELL/HOLD] (Confidence: X%)
🛡️ RISK MANAGEMENT: Entry: [zone], SL: [level], TP: [level], RR: [ratio]

JAWAB DALAM BAHASA INDONESIA!"""
        
        if market_context:
            base_prompt += f"\n\nKONTEKS PASAR SAAT INI:\n{market_context}"
        
        if additional_prompt:
            base_prompt += f"\n\nPETUNJUK TAMBAHAN:\n{additional_prompt}"
        
        return base_prompt
    
    def _call_vision_api(self, image_url: str, prompt: str, timeout: int) -> Dict:
        """Panggil API Ferdev Vision"""
        try:
            # Encode URL dan prompt
            encoded_url = quote(image_url)
            encoded_prompt = quote(prompt)
            
            # Build API URL
            api_url = f"{self.api_base}?link={encoded_url}&prompt={encoded_prompt}&apikey={self.api_key}"
            
            # Call API
            response = requests.get(api_url, timeout=timeout)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "success": False,
                    "message": f"API Error: {response.status_code}"
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "message": "API Timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Connection Error: {str(e)}"
            }
    
    def _parse_api_response(self, api_result: Dict) -> Dict[str, Any]:
        """Parse response dari API Ferdev"""
        if api_result.get("success"):
            raw_message = api_result.get("message", "")
            
            # Extract analysis dari raw message
            analysis = self._extract_analysis_from_text(raw_message)
            
            return {
                "success": True,
                "raw_response": raw_message,
                "analysis": analysis.get("analysis", raw_message[:500]),
                "recommendation": analysis.get("recommendation", "HOLD"),
                "confidence": analysis.get("confidence", 50),
                "levels": analysis.get("levels", {}),
                "risk_management": analysis.get("risk_management", {}),
                "timestamp": datetime.now()
            }
        else:
            # Fallback analysis jika API gagal
            return {
                "success": False,
                "analysis": "Analisis AI tidak tersedia. Gunakan analisis teknikal manual.",
                "recommendation": "HOLD",
                "confidence": 0,
                "levels": {},
                "risk_management": {},
                "timestamp": datetime.now()
            }
    
    def _extract_analysis_from_text(self, text: str) -> Dict:
        """Extract informasi struktur dari teks analisis"""
        # Default values
        result = {
            "analysis": text[:1000],
            "recommendation": "HOLD",
            "confidence": 50,
            "levels": {
                "support": [],
                "resistance": []
            },
            "risk_management": {
                "entry": "N/A",
                "stop_loss": "N/A",
                "take_profit": "N/A",
                "risk_reward": "N/A"
            }
        }
        
        # Cari rekomendasi dalam teks
        text_lower = text.lower()
        
        if "buy" in text_lower and "sell" not in text_lower.split("buy")[0]:
            result["recommendation"] = "BUY"
        elif "sell" in text_lower and "buy" not in text_lower.split("sell")[0]:
            result["recommendation"] = "SELL"
        
        # Cari confidence level
        import re
        confidence_match = re.search(r'confidence[:\s]*(\d+)%', text_lower)
        if confidence_match:
            result["confidence"] = int(confidence_match.group(1))
        
        return result

# Singleton instance
_analyzer_instance = None

def get_ferdev_analyzer(api_key: str = "key_17ToTo") -> FerdevVisionAnalyzer:
    """Get singleton instance of Ferdev analyzer"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FerdevVisionAnalyzer(api_key)
    return _analyzer_instance