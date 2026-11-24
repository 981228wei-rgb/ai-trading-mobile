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
    X_test, y_test = model_df[features].iloc[split_idx:], model_df["target"].iloc[split_idx:]

    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=6,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    test_acc = float(accuracy_score(y_test, clf.predict(X_test))) if len(X_test) > 0 else 0.5

    latest_row = df[features].iloc[[-1]].ffill().bfill()
    prob_up = float(clf.predict_proba(latest_row)[0][1])

    news_info = score_news(headlines)
    news_adj = news_info["score"] * 0.12
    blended_prob = float(np.clip(prob_up + news_adj, 0.05, 0.95))

    last_price = float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) >= 2 else last_price
    chg_1d_pct = ((last_price - prev_price) / prev_price * 100.0) if prev_price else 0.0

    atr_val = float(df["atr14"].iloc[-1]) if pd.notna(df["atr14"].iloc[-1]) else last_price * 0.02
    vol_val = float(df["vol_20"].iloc[-1]) if pd.notna(df["vol_20"].iloc[-1]) else 0.02
    expected_move_pct = float(vol_val * np.sqrt(target_days) * 100.0)

    if blended_prob >= 0.58:
        direction = "偏多看漲"
        action = "偏多操作（嚴守停損）"
    elif blended_prob <= 0.42:
        direction = "偏空看跌"
        action = "偏空/觀望（控制風險）"
    else:
        direction = "震盪中性"
        action = "觀望或區間試單"

    stop_loss = round(last_price - 1.5 * atr_val, 4)
    take_profit = round(last_price + 2.2 * atr_val, 4)

    risk_box = None
    if capital is not None and capital > 0:
        lev = float(leverage) if (leverage is not None and leverage > 0) else 1.0
        max_risk_amt = capital * 0.02
        stop_dist_pct = max(abs(last_price - stop_loss) / last_price, 0.005)
        suggested_notional = min(capital * lev, max_risk_amt / stop_dist_pct)
        suggested_units = suggested_notional / last_price if last_price > 0 else 0.0
        risk_box = {
            "capital": round(capital, 2),
            "leverage": round(lev, 2),
            "max_risk_2pct": round(max_risk_amt, 2),
            "suggested_notional": round(suggested_notional, 2),
            "suggested_units": round(suggested_units, 4),
        }

    rsi_now = float(df["rsi14"].iloc[-1]) if pd.notna(df["rsi14"].iloc[-1]) else 50.0
    ma20_now = float(df["ma20"].iloc[-1]) if pd.notna(df["ma20"].iloc[-1]) else last_price
    ma60_now = float(df["ma60"].iloc[-1]) if pd.notna(df["ma60"].iloc[-1]) else last_price

    return {
        "symbol": symbol,
        "market": market,
        "target_days": target_days,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_price": round(last_price, 4),
        "chg_1d_pct": round(chg_1d_pct, 2),
        "direction": direction,
        "action": action,
        "ml_prob_up": round(prob_up * 100, 1),
        "blended_prob_up": round(blended_prob * 100, 1),
        "test_accuracy": round(test_acc * 100, 1),
        "expected_move_pct": round(expected_move_pct, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "indicators": {
            "rsi14": round(rsi_now, 1),
            "ma20": round(ma20_now, 4),
            "ma60": round(ma60_now, 4),
            "atr14": round(atr_val, 4),
        },
        "news": news_info,
        "risk": risk_box,
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI 行情與風控助手</title>
  <style>
    :root {
      --bg: #0b1020;
      --card: #131b31;
      --card-2: #1a2440;
      --text: #eef2ff;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --up: #22c55e;
      --down: #ef4444;
      --warn: #f59e0b;
      --border: rgba(148, 163, 184, 0.18);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 14px;
    }
    .container { max-width: 680px; margin: 0 auto; }
    .header {
      background: linear-gradient(135deg, #172554, #0f172a);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 12px;
    }
    .header h1 { margin: 0 0 4px; font-size: 1.25rem; }
    .header p { margin: 0; color: var(--muted); font-size: 0.85rem; }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px;
      margin-bottom: 12px;
    }
    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    label { display: block; font-size: 0.8rem; color: var(--muted); margin-bottom: 4px; }
    select, input {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: var(--card-2);
      color: var(--text);
      font-size: 0.95rem;
    }
    .presets { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 12px; }
    .chip {
      background: var(--card-2);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      cursor: pointer;
    }
    .chip:active { transform: scale(0.98); }
    .btn {
      width: 100%;
      padding: 12px;
      border: none;
      border-radius: 12px;
      background: var(--accent);
      color: #061224;
      font-weight: 700;
      font-size: 1rem;
      cursor: pointer;
    }
    .btn:disabled { opacity: 0.6; cursor: not-allowed; }
    .status-banner {
      display: flex; justify-content: space-between; align-items: center;
      padding: 12px; border-radius: 12px; background: var(--card-2); margin-bottom: 10px;
    }
    .tag-up { color: var(--up); font-weight: 700; }
    .tag-down { color: var(--down); font-weight: 700; }
    .tag-warn { color: var(--warn); font-weight: 700; }
    .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .metric { background: var(--card-2); padding: 10px; border-radius: 10px; }
    .metric .k { font-size: 0.75rem; color: var(--muted); }
    .metric .v { font-size: 1.05rem; font-weight: 700; margin-top: 2px; }
    .news-item { padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
    .news-item:last-child { border-bottom: none; }
    .err { color: #fca5a5; background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.35); padding: 10px; border-radius: 10px; }
    .disclaimer { font-size: 0.75rem; color: var(--muted); text-align: center; margin-top: 8px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>📱 AI 行情與風控助手</h1>
      <p>外出隨時查詢技術面、隨機森林機率、新聞情緒與停損試算</p>
    </div>

    <div class="card">
      <div class="grid-2">
        <div>
          <label for="market">市場類別</label>
          <select id="market" onchange="renderPresets()">
            <option value="加密貨幣">加密貨幣</option>
            <option value="美股">美股</option>
            <option value="台股">台股</option>
          </select>
        </div>
        <div>
          <label for="days">預測天數（1～15）</label>
          <input id="days" type="number" min="1" max="15" value="3" />
        </div>
      </div>

      <div style="margin-top:10px;">
        <label for="symbol">標的代碼</label>
        <input id="symbol" type="text" value="BTC-USD" placeholder="例如 BTC-USD、NVDA、2330.TW" />
      </div>

      <div id="presets" class="presets"></div>

      <div class="grid-2" style="margin-bottom:12px;">
        <div>
          <label for="capital">本金（選填，風控試算）</label>
          <input id="capital" type="number" step="any" placeholder="例如 10000" />
        </div>
        <div>
          <label for="leverage">槓桿倍數（選填）</label>
          <input id="leverage" type="number" step="any" placeholder="例如 1 或 2" />
        </div>
      </div>

      <button id="runBtn" class="btn" onclick="runAnalyze()">開始分析</button>
    </div>

    <div id="resultArea"></div>
    <div class="disclaimer">僅供量化分析與風控參考，不構成投資建議；交易請嚴控風險。</div>
  </div>

  <script>
    const PRESETS = __PRESETS_JSON__;

    function renderPresets() {
      const m = document.getElementById("market").value;
      const box = document.getElementById("presets");
      const list = PRESETS[m] || [];
      box.innerHTML = "";
      list.forEach((item, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "chip";
        btn.textContent = item.label;
        btn.onclick = () => {
          document.getElementById("symbol").value = item.symbol;
          runAnalyze();
        };
        box.appendChild(btn);
        if (idx === 0 && !document.getElementById("symbol").dataset.manual) {
          document.getElementById("symbol").value = item.symbol;
        }
      });
    }

    document.getElementById("symbol").addEventListener("input", () => {
      document.getElementById("symbol").dataset.manual = "1";
    });

    async function runAnalyze() {
      const btn = document.getElementById("runBtn");
      const area = document.getElementById("resultArea");
      btn.disabled = true;
      btn.textContent = "分析中（抓行情、跑模型與新聞）...";
      area.innerHTML = `<div class
