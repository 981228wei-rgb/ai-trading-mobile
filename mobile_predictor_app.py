import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template_string, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

POSITIVE_WORDS = [
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "bullish",
    "upgrade", "upgraded", "record", "profit", "growth", "cut rates", "rate cut",
    "adoption", "inflow", "approval", "breakout",
    "大漲", "創高", "新高", "利多", "成長", "營收創高", "獲利", "買超",
    "降息", "擴產", "調升", "轉虧為盈", "強勁", "突破"
]

NEGATIVE_WORDS = [
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "bearish", "downgrade", "downgraded", "loss", "losses", "lawsuit", "hack",
    "hike", "rate hike", "inflation", "outflow", "ban", "breakdown",
    "重挫", "大跌", "跌停", "利空", "衰退", "虧損", "賣超", "升息",
    "調降", "違約", "調查", "縮水", "疲弱", "跌破"
]

MARKET_PRESETS = {
    "加密貨幣": [
        {"label": "BTC 比特幣", "symbol": "BTC-USD"},
        {"label": "ETH 以太幣", "symbol": "ETH-USD"},
        {"label": "SOL 索拉納", "symbol": "SOL-USD"},
        {"label": "BNB 幣安幣", "symbol": "BNB-USD"},
    ],
    "美股": [
        {"label": "NVDA 輝達", "symbol": "NVDA"},
        {"label": "AAPL 蘋果", "symbol": "AAPL"},
        {"label": "TSLA 特斯拉", "symbol": "TSLA"},
        {"label": "SPY 標普500", "symbol": "SPY"},
    ],
    "台股": [
        {"label": "2330 台積電", "symbol": "2330.TW"},
        {"label": "2454 聯發科", "symbol": "2454.TW"},
        {"label": "0050 元大台灣50", "symbol": "0050.TW"},
        {"label": "2317 鴻海", "symbol": "2317.TW"},
    ],
}


def normalize_symbol(market: str, raw_symbol: str) -> str:
    sym = (raw_symbol or "").strip().upper()
    if not sym:
        return "BTC-USD"
    if market == "加密貨幣":
        if "-" not in sym and not sym.endswith("USD"):
            return f"{sym}-USD"
        return sym
    if market == "台股":
        if sym.isdigit():
            return f"{sym}.TW"
        return sym
    return sym


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def fetch_news_headlines(symbol: str, limit: int = 6):
    query = quote(f"{symbol} stock OR crypto OR market")
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}®ion=US&lang=en-US"
    fallback_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    headlines = []
    for feed_url in (url, fallback_url):
        try:
            resp = requests.get(feed_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.text)
            for item in root.findall(".//item")[:limit]:
                title_el = item.find("title")
                pub_el = item.find("pubDate")
                if title_el is not None and title_el.text:
                    headlines.append({
                        "title": title_el.text.strip(),
                        "published": pub_el.text.strip() if pub_el is not None and pub_el.text else "",
                    })
            if headlines:
                break
        except Exception:
            continue
    return headlines[:limit]


def score_news(headlines):
    if not headlines:
        return {"score": 0.0, "label": "中性（無近期新聞）", "items": []}

    items = []
    total = 0.0
    for h in headlines:
        t_low = h["title"].lower()
        pos = sum(1 for w in POSITIVE_WORDS if w.lower() in t_low)
        neg = sum(1 for w in NEGATIVE_WORDS if w.lower() in t_low)
        diff = pos - neg
        item_score = float(np.clip(diff * 0.35, -1.0, 1.0))
        total += item_score
        sentiment = "偏多" if item_score > 0.1 else ("偏空" if item_score < -0.1 else "中性")
        items.append({
            "title": h["title"],
            "published": h["published"],
            "sentiment": sentiment,
            "score": round(item_score, 2),
        })

    avg_score = float(np.clip(total / len(items), -1.0, 1.0))
    if avg_score >= 0.15:
        label = f"偏多 ({avg_score:+.2f})"
    elif avg_score <= -0.15:
        label = f"偏空 ({avg_score:+.2f})"
    else:
        label = f"中性 ({avg_score:+.2f})"
    return {"score": round(avg_score, 3), "label": label, "items": items}


def run_prediction(market: str, raw_symbol: str, target_days: int = 3, leverage=None, capital=None):
    symbol = normalize_symbol(market, raw_symbol)
    target_days = max(1, min(int(target_days or 3), 15))

    with ThreadPoolExecutor(max_workers=2) as pool:
        hist_fut = pool.submit(yf.download, symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        news_fut = pool.submit(fetch_news_headlines, symbol, 6)
        df = hist_fut.result()
        headlines = news_fut.result()

    if df is None or df.empty or len(df) < 40:
        raise ValueError(f"找不到 {symbol} 足夠的歷史行情資料，請確認代碼是否正確（例如 BTC-USD、NVDA、2330.TW）。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(df) < 40:
        raise ValueError(f"{symbol} 有效交易日資料不足。")

    close = df["Close"]
    df["ret_1d"] = close.pct_change()
    df["ret_3d"] = close.pct_change(3)
    df["ret_5d"] = close.pct_change(5)
    df["ma10"] = close.rolling(10).mean()
    df["ma20"] = close.rolling(20).mean()
    df["ma60"] = close.rolling(60).mean()
    df["ma10_gap"] = (close - df["ma10"]) / df["ma10"]
    df["ma20_gap"] = (close - df["ma20"]) / df["ma20"]
    df["rsi14"] = compute_rsi(close, 14)
    macd, signal, hist = compute_macd(close)
    df["macd_hist"] = hist / close.replace(0, np.nan)
    df["vol_20"] = df["ret_1d"].rolling(20).std()
    df["atr14"] = compute_atr(df, 14)
    df["atr_pct"] = df["atr14"] / close.replace(0, np.nan)

    df["future_ret"] = close.shift(-target_days) / close - 1.0
    df["target"] = (df["future_ret"] > 0).astype(int)

    features = [
        "ret_1d", "ret_3d", "ret_5d",
        "ma10_gap", "ma20_gap", "rsi14",
        "macd_hist", "vol_20", "atr_pct"
    ]

    model_df = df.dropna(subset=features + ["target"]).copy()
    if len(model_df) < 35:
        raise ValueError("可訓練樣本數不足，請嘗試其他標的。")

    split_idx = int(len(model_df) * 0.8)
    X_train, y_train = model_df[features].iloc[:split_idx], model_df["target"].iloc[:split_idx]
    X_test, y_test = model_df[features].
