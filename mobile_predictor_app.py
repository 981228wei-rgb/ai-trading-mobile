import os
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

POSITIVE_WORDS = [
    "beat", "beats", "surge", "surges", "soar", "soars", "rally", "bullish",
    "upgrade", "upgraded", "record", "profit", "growth", "cut rates", "rate cut",
    "adoption", "inflow", "approval", "breakout", "gain", "gains", "jump", "jumps",
    "大漲", "創高", "新高", "利多", "成長", "營收創高", "獲利", "買超",
    "降息", "擴產", "調升", "轉虧為盈", "強勁", "突破", "看漲", "上揚", "回升"
]

NEGATIVE_WORDS = [
    "miss", "misses", "plunge", "plunges", "drop", "drops", "fall", "falls",
    "bearish", "downgrade", "downgraded", "loss", "losses", "lawsuit", "hack",
    "hike", "rate hike", "inflation", "outflow", "ban", "breakdown", "sink", "sinks",
    "重挫", "大跌", "跌停", "利空", "衰退", "虧損", "賣超", "升息",
    "調降", "違約", "調查", "縮水", "疲弱", "跌破", "看跌", "走跌", "回檔"
]

SYMBOL_KEYWORDS = {
    "BTC-USD": "Bitcoin BTC",
    "ETH-USD": "Ethereum ETH",
    "SOL-USD": "Solana SOL",
    "BNB-USD": "BNB Binance",
    "NVDA": "NVDA 輝達",
    "AAPL": "AAPL 蘋果",
    "TSLA": "TSLA 特斯拉",
    "SPY": "SPY 標普500",
    "2330.TW": "2330 台積電",
    "2454.TW": "2454 聯發科",
    "0050.TW": "0050 元大台灣50",
    "2317.TW": "2317 鴻海",
}

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


def _parse_rss(feed_url: str, source_tag: str, limit: int = 5):
    items = []
    try:
        resp = requests.get(feed_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return items
        root = ET.fromstring(resp.text)
        for item in root.findall(".//item")[:limit]:
            title_el = item.find("title")
            pub_el = item.find("pubDate")
            if title_el is not None and title_el.text:
                raw_title = title_el.text.strip()
                clean_title = raw_title.rsplit(" - ", 1)[0].strip() if " - " in raw_title else raw_title
                items.append({
                    "title": f"[{source_tag}] {clean_title}",
                    "published": pub_el.text.strip() if pub_el is not None and pub_el.text else "",
                })
    except Exception:
        pass
    return items


def fetch_news_headlines(symbol: str, limit: int = 8):
    base_kw = SYMBOL_KEYWORDS.get(symbol, symbol.replace("-USD", "").replace(".TW", ""))
    en_query = quote(f"{base_kw} market OR stock OR crypto")
    zh_query = quote(f"{base_kw} 股價 OR 行情 OR 財報 OR 市場")

    feeds = [
        (f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(symbol)}®ion=US&lang=en-US", "Yahoo"),
        (f"https://news.google.com/rss/search?q={zh_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant", "Google中文"),
        (f"https://news.google.com/rss/search?q={en_query}&hl=en-US&gl=US&ceid=US:en", "Google英文"),
    ]

    headlines = []
    seen_titles = set()
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_parse_rss, url, tag, 4) for url, tag in feeds]
        for fut in futures:
            for item in fut.result():
                key = item["title"].lower()
                if key not in seen_titles:
                    seen_titles.add(key)
                    headlines.append(item)
                    if len(headlines) >= limit:
                        return headlines[:limit]

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


def fetch_live_price(symbol: str):
    try:
        info = yf.Ticker(symbol).fast_info
        for key in ("lastPrice", "regularMarketPrice", "previousClose"):
            val = info.get(key) if hasattr(info, "get") else getattr(info, key, None)
            if val is not None and float(val) > 0:
                return float(val)
    except Exception:
        pass
    return None


def run_prediction(market: str, raw_symbol: str, target_days: int = 3, leverage=None, capital=None):
    symbol = normalize_symbol(market, raw_symbol)
    target_days = max(1, min(int(target_days or 3), 15))

    with ThreadPoolExecutor(max_workers=3) as pool:
        hist_fut = pool.submit(yf.download, symbol, period="1y", interval="1d", progress=False, auto_adjust=True)
        news_fut = pool.submit(fetch_news_headlines, symbol, 8)
        live_fut = pool.submit(fetch_live_price, symbol)
        df = hist_fut.result()
        headlines = news_fut.result()
        live_price = live_fut.result()

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

    last_price = live_price if live_price is not None else float(close.iloc[-1])
    prev_price = float(close.iloc[-2]) if len(close) >= 2 else last_price
    chg_1d_pct = ((last_price - prev_price) / prev_price * 100.0) if prev_price else 0.0

    atr_val = float(df["atr14"].iloc[-1]) if pd.notna(df["atr14"].iloc[-1]) else last_price * 0.02
    vol_val = float(df["vol_20"].iloc[-1]) if pd.notna(df["vol_20"].iloc[-1]) else 0.02
    expected_move_pct = float(vol_val * np.sqrt(target_days) * 100.0)

    if blended_prob >= 0.58:
        direction = "偏多看漲"
        action = "偏多操作（嚴守停損）"
        entry_price = round(last_price - 0.5 * atr_val, 4)
    elif blended_prob <= 0.42:
        direction = "偏空看跌"
        action = "偏空/觀望（控制風險）"
        entry_price = round(last_price + 0.5 * atr_val, 4)
    else:
        direction = "震盪中性"
        action = "觀望或區間試單"
        entry_price = round(last_price, 4)

    stop_loss = round(last_price - 1.5 * atr_val, 4)
    take_profit = round(last_price + 2.2 * atr_val, 4)

    risk_box = None
    if capital is not None and capital > 0:
        lev = float(leverage) if (leverage is not None and leverage > 0) else 1.0
        max_risk_amt = capital * 0.02
        stop_dist_pct = max(abs(entry_price - stop_loss) / max(entry_price, 1e-8), 0.005)
        suggested_notional = min(capital * lev, max_risk_amt / stop_dist_pct)
        suggested_units = suggested_notional / entry_price if entry_price > 0 else 0.0
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
    taipei_now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "symbol": symbol,
        "market": market,
        "target_days": target_days,
        "updated_at": taipei_now,
        "last_price": round(last_price, 4),
        "chg_1d_pct": round(chg_1d_pct, 2),
        "direction": direction,
        "action": action,
        "entry_price": entry_price,
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


@app.route("/")
def index():
    return render_template("index.html", presets_json=json.dumps(MARKET_PRESETS, ensure_ascii=False))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json(force=True) or {}
        market = body.get("market", "加密貨幣")
        symbol = body.get("symbol", "BTC-USD")
        days = int(body.get("days", 3))
        cap_raw = str(body.get("capital", "")).strip()
        lev_raw = str(body.get("leverage", "")).strip()
        cap_val = float(cap_raw) if cap_raw else None
        lev_val = float(lev_raw) if lev_raw else None

        res = run_prediction(
            market=market,
            raw_symbol=symbol,
            target_days=days,
            leverage=lev_val,
            capital=cap_val,
        )
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
