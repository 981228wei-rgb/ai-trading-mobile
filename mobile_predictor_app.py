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
from sklearn.model_selection import TimeSeriesSplit

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

TRACK_PROFILES = {
    "加密貨幣": {
        "track_name": "加密貨幣軌（結構控損・高盈虧比）",
        "bull_threshold": 0.58,
        "bear_threshold": 0.42,
        "entry_atr_mult": 0.35,
        "stop_atr_mult": 1.25,
        "tp_atr_mult": 2.35,
        "target_ret_min": 0.008,
        "max_risk_pct": 0.010,
        "min_stop_pct": 0.010,
        "extreme_atr_pct": 0.085,
        "extreme_lev": 5.0,
        "extreme_move_pct": 12.0,
        "fee_rate": 0.0005,
    },
    "加密貨幣_極短線": {
        "track_name": "加密貨幣極短線軌（高槓桿・緊停損防洗盤）",
        "bull_threshold": 0.57,
        "bear_threshold": 0.43,
        "entry_atr_mult": 0.20,
        "stop_atr_mult": 0.85,
        "tp_atr_mult": 1.65,
        "target_ret_min": 0.0025,
        "max_risk_pct": 0.006,
        "min_stop_pct": 0.004,
        "extreme_atr_pct": 0.035,
        "extreme_lev": 10.0,
        "extreme_move_pct": 6.0,
        "fee_rate": 0.0006,
    },
    "股票": {
        "track_name": "股票軌（順均線・控波段）",
        "bull_threshold": 0.55,
        "bear_threshold": 0.45,
        "entry_atr_mult": 0.30,
        "stop_atr_mult": 1.15,
        "tp_atr_mult": 2.00,
        "target_ret_min": 0.003,
        "max_risk_pct": 0.015,
        "min_stop_pct": 0.008,
        "extreme_atr_pct": 0.065,
        "extreme_lev": 3.0,
        "extreme_move_pct": 10.0,
        "fee_rate": 0.0005,
    },
}


def get_track_profile(market: str, target_days: float) -> dict:
    if market == "加密貨幣":
        if target_days < 1.0:
            return TRACK_PROFILES["加密貨幣_極短線"]
        return TRACK_PROFILES["加密貨幣"]
    return TRACK_PROFILES["股票"]


def normalize_symbol(market: str, raw_symbol: str) -> str:
    sym = (raw_symbol or "").strip().upper()
    if not sym:
        return "BTC-USD" if market == "加密貨幣" else "NVDA"
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


def analyze_price_structure(df: pd.DataFrame, last_price: float, range_label: str = "20根K線") -> dict:
    recent = df.tail(20).copy()
    last_bar = df.iloc[-1]

    open_p = float(last_bar["Open"])
    high_p = max(float(last_bar["High"]), float(last_price))
    low_p = min(float(last_bar["Low"]), float(last_price))
    close_p = float(last_price)

    total_range = max(high_p - low_p, 1e-8)
    body = abs(close_p - open_p)
    upper_wick = max(high_p - max(open_p, close_p), 0.0)
    lower_wick = max(min(open_p, close_p) - low_p, 0.0)

    if lower_wick >= total_range * 0.45 and lower_wick > upper_wick * 1.5:
        candle_signal = "長下影線（下檔有承接/防插針）"
    elif upper_wick >= total_range * 0.45 and upper_wick > lower_wick * 1.5:
        candle_signal = "長上影線（上方有賣壓/防追高）"
    elif body >= total_range * 0.65:
        candle_signal = "強勢長紅K（多方動能明確）" if close_p >= open_p else "強勢長黑K（空方賣壓宣洩）"
    else:
        candle_signal = "十字/小實體震盪K（多空拉鋸）"

    ma20 = float(df["ma20"].iloc[-1]) if pd.notna(df["ma20"].iloc[-1]) else close_p
    ma60 = float(df["ma60"].iloc[-1]) if pd.notna(df["ma60"].iloc[-1]) else close_p

    if close_p > ma20 and ma20 > ma60:
        ma_structure = "多頭排列（站穩 MA20 與 MA60 之上）"
    elif close_p < ma20 and ma20 < ma60:
        ma_structure = "空頭排列（位於 MA20 與 MA60 之下）"
    elif close_p >= ma20:
        ma_structure = "短線偏強（站上 MA20，留意 MA60 壓力）"
    else:
        ma_structure = "短線偏弱（跌破 MA20，留意下方支撐）"

    high_20 = max(float(recent["High"].max()), close_p)
    low_20 = min(float(recent["Low"].min()), close_p)
    range_20 = max(high_20 - low_20, 1e-8)
    pos_in_range = round(float(np.clip((close_p - low_20) / range_20 * 100.0, 0.0, 100.0)), 1)

    return {
        "candle_signal": candle_signal,
        "ma_structure": ma_structure,
        "high_20": round(high_20, 4),
        "low_20": round(low_20, 4),
        "range_pos_pct": pos_in_range,
        "range_label": range_label,
        "summary": f"{ma_structure}｜{candle_signal}｜{range_label}區間 {pos_in_range}%",
    }


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


def fetch_history_df(symbol: str, use_intraday: bool = False) -> pd.DataFrame:
    df = None
    period = "90d" if use_intraday else "2y"
    interval = "1h" if use_intraday else "1d"
    try:
        df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)
    except Exception:
        df = None

    if df is None or df.empty:
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True, threads=False)
        except Exception:
            df = None

    if (df is None or df.empty) and use_intraday:
        try:
            df = yf.Ticker(symbol).history(period="2y", interval="1d", auto_adjust=True)
        except Exception:
            df = None

    return df


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


def _build_feature_df(df: pd.DataFrame, horizon_steps: int, target_ret_min: float) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    vol = out["Volume"].replace(0, np.nan)
    vol_ma20 = vol.rolling(20).mean()

    out["ret_1d"] = close.pct_change()
    out["ret_3d"] = close.pct_change(3)
    out["ret_5d"] = close.pct_change(5)
    out["ma10"] = close.rolling(10).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma10_gap"] = (close - out["ma10"]) / out["ma10"].replace(0, np.nan)
    out["ma20_gap"] = (close - out["ma20"]) / out["ma20"].replace(0, np.nan)
    out["ma20_slope"] = out["ma20"].pct_change(3)
    out["rsi14"] = compute_rsi(close, 14)
    _, _, hist = compute_macd(close)
    out["macd_hist"] = hist / close.replace(0, np.nan)
    out["vol_20"] = out["ret_1d"].rolling(20).std()
    out["vol_ratio"] = (vol / vol_ma20).clip(0.1, 5.0).fillna(1.0)
    out["atr14"] = compute_atr(out, 14)
    out["atr_pct"] = out["atr14"] / close.replace(0, np.nan)

    out["future_ret"] = close.shift(-horizon_steps) / close - 1.0
    out["target"] = (out["future_ret"] > target_ret_min).astype(int)
    return out


FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d",
    "ma10_gap", "ma20_gap", "ma20_slope",
    "rsi14", "macd_hist", "vol_20", "vol_ratio", "atr_pct"
]


def _evaluate_time_series_cv(X: pd.DataFrame, y: pd.Series) -> float:
    n_samples = len(X)
    if n_samples < 60:
        return 0.5
    n_splits = min(4, max(2, n_samples // 35))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        if len(train_idx) < 25 or len(val_idx) < 8:
            continue
        m = RandomForestClassifier(
            n_estimators=160,
            max_depth=5,
            min_samples_leaf=6,
            random_state=42,
            n_jobs=1,
        )
        m.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = m.predict(X.iloc[val_idx])
        scores.append(float(accuracy_score(y.iloc[val_idx], preds)))
    return float(np.mean(scores)) if scores else 0.5


def _decide_signal(
    profile: dict,
    prob_up: float,
    blended_prob: float,
    test_acc: float,
    news_score: float,
    last_price: float,
    prev_price: float,
    atr_val: float,
    rsi_now: float,
    range_pos_pct: float,
    low_20: float,
    lev_val: float,
    is_intraday_scalp: bool,
):
    chg_pct = ((last_price - prev_price) / prev_price * 100.0) if prev_price else 0.0
    atr_pct_now = atr_val / max(last_price, 1e-8)

    extreme_high_risk = (
        atr_pct_now >= profile["extreme_atr_pct"]
        and (lev_val >= profile["extreme_lev"] or abs(chg_pct) >= profile["extreme_move_pct"])
    )
    indecisive_conflict = (
        abs(blended_prob - 0.50) <= 0.03
        and test_acc < 0.48
        and abs(news_score) >= 0.25
        and ((news_score > 0 and prob_up < 0.48) or (news_score < 0 and prob_up > 0.52))
    )

    entry_mult = profile["entry_atr_mult"]
    stop_mult = profile["stop_atr_mult"]
    tp_mult = profile["tp_atr_mult"]

    low_zone_no_short = (
        range_pos_pct <= 20.0
        or rsi_now <= 35.0
        or chg_pct >= (2.5 if is_intraday_scalp else 5.0)
    )

    if extreme_high_risk or indecisive_conflict:
        direction = "建議先不做單"
        entry_price = round(last_price, 4)
        stop_loss = round(last_price - stop_mult * atr_val, 4)
        take_profit = round(last_price + tp_mult * atr_val, 4)
        rr_ratio = round(tp_mult / max(stop_mult, 1e-8), 2)
        action = "極短線波動與槓桿風險過高，建議先空手觀望，不硬做單" if extreme_high_risk else "訊號處於五五波與多空拉鋸，建議先不做單"
    elif blended_prob >= profile["bull_threshold"]:
        direction = "偏多看漲"
        entry_price = round(min(last_price, last_price - entry_mult * atr_val), 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        risk_per_unit = max(entry_price - stop_loss, 1e-8)
        reward_per_unit = max(take_profit - entry_price, 0.0)
        rr_ratio = round(reward_per_unit / risk_per_unit, 2)
        action = f"極短線回測偏多（停損 {stop_mult:.2f} ATR / 停利 {tp_mult:.2f} ATR，盈虧比 1:{rr_ratio}）"
    elif blended_prob <= profile["bear_threshold"] and not low_zone_no_short:
        direction = "偏空看跌"
        entry_price = round(max(last_price, last_price + entry_mult * atr_val), 4)
        stop_loss = round(entry_price + stop_mult * atr_val, 4)
        take_profit = round(entry_price - tp_mult * atr_val, 4)
        risk_per_unit = max(stop_loss - entry_price, 1e-8)
        reward_per_unit = max(entry_price - take_profit, 0.0)
        rr_ratio = round(reward_per_unit / risk_per_unit, 2)
        action = f"極短線反彈偏空（停損 {stop_mult:.2f} ATR / 停利 {tp_mult:.2f} ATR，盈虧比 1:{rr_ratio}）"
    elif blended_prob <= profile["bear_threshold"] and low_zone_no_short:
        direction = "低點防反彈（不追空）"
        entry_price = round(last_price, 4)
        min_buf = max(stop_mult * atr_val, last_price * profile["min_stop_pct"])
        raw_stop = min(last_price - min_buf, low_20 - 0.25 * atr_val)
        stop_loss = round(max(raw_stop, last_price - max(stop_mult * 1.25, 1.0) * atr_val), 4)
        if stop_loss >= entry_price:
            stop_loss = round(entry_price - min_buf, 4)
        take_profit = round(last_price + tp_mult * atr_val, 4)
        risk_per_unit = max(entry_price - stop_loss, 1e-8)
        reward_per_unit = max(take_profit - entry_price, 0.0)
        rr_ratio = round(reward_per_unit / risk_per_unit, 2)
        action = "處於低位/超賣或急拉區，禁止低點追空，等反彈遇阻再看"
    else:
        direction = "震盪中性"
        entry_price = round(last_price, 4)
        stop_loss = round(last_price - stop_mult * atr_val, 4)
        take_profit = round(last_price + tp_mult * atr_val, 4)
        rr_ratio = round(tp_mult / max(stop_mult, 1e-8), 2)
        action = f"觀望為主（盈虧比參考 1:{rr_ratio}，等明確突破再動手）"

    return direction, action, entry_price, stop_loss, take_profit, rr_ratio, chg_pct


def run_prediction(market: str, raw_symbol: str, target_days=3, leverage=None, capital=None):
    raw_days = float(target_days if target_days is not None else 3)
    target_days = max(0.05, min(raw_days, 15.0))
    profile = get_track_profile(market, target_days)
    symbol = normalize_symbol(market, raw_symbol)

    is_intraday_scalp = (market == "加密貨幣" and target_days < 1.0)
    horizon_steps = max(2, int(round(target_days * 24))) if is_intraday_scalp else max(1, int(round(target_days)))

    with ThreadPoolExecutor(max_workers=3) as pool:
        hist_fut = pool.submit(fetch_history_df, symbol, is_intraday_scalp)
        news_fut = pool.submit(fetch_news_headlines, symbol, 8)
        live_fut = pool.submit(fetch_live_price, symbol)
        df = hist_fut.result()
        headlines = news_fut.result()
        live_price = live_fut.result()

    if df is None or df.empty or len(df) < 60:
        raise ValueError(f"找不到 {symbol} 足夠的歷史行情資料，請確認代碼是否正確（例如 BTC-USD、NVDA、2330.TW）。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 60:
        raise ValueError(f"{symbol} 有效K線資料不足。")

    feat_df = _build_feature_df(df, horizon_steps, profile["target_ret_min"])
    model_df = feat_df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    if len(model_df) < 50:
        raise ValueError("可訓練樣本數不足，請嘗試其他標的。")

    X_all = model_df[FEATURE_COLS]
    y_all = model_df["target"]
    test_acc = _evaluate_time_series_cv(X_all, y_all)

    clf = RandomForestClassifier(
        n_estimators=180,
        max_depth=5,
        min_samples_leaf=6,
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_all, y_all)

    latest_row = feat_df[FEATURE_COLS].iloc[[-1]].ffill().bfill()
    prob_up = float(clf.predict_proba(latest_row)[0][1])

    news_info = score_news(headlines)
    news_adj = news_info["score"] * 0.08 if abs(news_info["score"]) >= 0.15 else 0.0
    blended_prob = float(np.clip(prob_up + news_adj, 0.08, 0.92))

    last_price = live_price if live_price is not None else float(feat_df["Close"].iloc[-1])
    prev_price = float(feat_df["Close"].iloc[-2]) if len(feat_df) >= 2 else last_price

    range_label = "20根1h" if is_intraday_scalp else "20日"
    structure_info = analyze_price_structure(feat_df, last_price, range_label=range_label)
    rsi_now = float(feat_df["rsi14"].iloc[-1]) if pd.notna(feat_df["rsi14"].iloc[-1]) else 50.0
    ma20_now = float(feat_df["ma20"].iloc[-1]) if pd.notna(feat_df["ma20"].iloc[-1]) else last_price
    ma60_now = float(feat_df["ma60"].iloc[-1]) if pd.notna(feat_df["ma60"].iloc[-1]) else last_price

    atr_val = float(feat_df["atr14"].iloc[-1]) if pd.notna(feat_df["atr14"].iloc[-1]) else last_price * 0.01
    vol_val = float(feat_df["vol_20"].iloc[-1]) if pd.notna(feat_df["vol_20"].iloc[-1]) else 0.01
    expected_move_pct = float(vol_val * np.sqrt(horizon_steps) * 100.0)
    lev_val = float(leverage) if (leverage is not None and leverage > 0) else 1.0

    direction, action, entry_price, stop_loss, take_profit, rr_ratio, chg_1d_pct = _decide_signal(
        profile=profile,
        prob_up=prob_up,
        blended_prob=blended_prob,
        test_acc=test_acc,
        news_score=news_info["score"],
        last_price=last_price,
        prev_price=prev_price,
        atr_val=atr_val,
        rsi_now=rsi_now,
        range_pos_pct=structure_info["range_pos_pct"],
        low_20=structure_info["low_20"],
        lev_val=lev_val,
        is_intraday_scalp=is_intraday_scalp,
    )

    news_score_val = news_info["score"]
    news_aligned_bull = (blended_prob >= 0.66) and (news_score_val >= -0.10) and (prob_up >= 0.62)
    news_aligned_bear = (blended_prob <= 0.34) and (news_score_val <= 0.10) and (prob_up <= 0.38)
    high_win_qualified = (
        direction in ("偏多看漲", "偏空看跌")
        and test_acc >= 0.56
        and rr_ratio >= 1.50
        and (news_aligned_bull or news_aligned_bear)
    )
    prob_edge = abs(blended_prob - 0.50) * 100.0
    edge_score = round(prob_edge * 1.6 + max(0.0, (test_acc - 0.50) * 100.0) * 1.2 + min(rr_ratio, 3.0) * 4.0, 1)

    risk_box = None
    if capital is not None and capital > 0:
        max_risk_amt = capital * profile["max_risk_pct"]
        stop_dist = abs(entry_price - stop_loss)
        stop_dist_pct = max(stop_dist / max(entry_price, 1e-8), profile["min_stop_pct"])
        suggested_notional = min(capital * lev_val, max_risk_amt / stop_dist_pct)
        suggested_units = suggested_notional / entry_price if entry_price > 0 else 0.0
        risk_box = {
            "capital": round(capital, 2),
            "leverage": round(lev_val, 2),
            "risk_pct_label": f"{profile['max_risk_pct'] * 100:.1f}%",
            "max_risk_2pct": round(max_risk_amt, 2),
            "suggested_notional": round(suggested_notional, 2),
            "suggested_units": round(suggested_units, 6),
        }

    taipei_now = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
    days_display = f"{target_days:g}"
    win_prob_display = round(blended_prob * 100, 1) if direction == "偏多看漲" else round((1.0 - blended_prob) * 100, 1)

    return {
        "symbol": symbol,
        "market": market,
        "track_name": profile["track_name"],
        "target_days": days_display,
        "updated_at": taipei_now,
        "last_price": round(last_price, 4),
        "chg_1d_pct": round(chg_1d_pct, 2),
        "direction": direction,
        "action": action,
        "entry_price": entry_price,
        "rr_ratio": rr_ratio,
        "win_prob_display": win_prob_display,
        "high_win_qualified": high_win_qualified,
        "edge_score": edge_score,
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
        "structure": structure_info,
        "news": news_info,
        "risk": risk_box,
    }


def run_backtest(market: str, raw_symbol: str, target_days=3, leverage=None):
    raw_days = float(target_days if target_days is not None else 3)
    target_days = max(0.05, min(raw_days, 15.0))
    profile = get_track_profile(market, target_days)
    symbol = normalize_symbol(market, raw_symbol)
    is_intraday_scalp = (market == "加密貨幣" and target_days < 1.0)
    lev_val = float(leverage) if (leverage is not None and leverage > 0) else 1.0
    fee_cost = profile["fee_rate"] * 2.0
    backtest_days = 30

    df = fetch_history_df(symbol, is_intraday_scalp)
    if (df is None or df.empty or len(df) < 60) and is_intraday_scalp:
        df = fetch_history_df(symbol, False)
        is_intraday_scalp = False

    if df is None or df.empty or len(df) < 45:
        raise ValueError(f"{symbol} 歷史 K 線不足，無法執行 30 天回測。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 45:
        raise ValueError(f"{symbol} 有效 K 線資料不足。")

    horizon_steps = max(2, int(round(target_days * 24))) if is_intraday_scalp else max(1, int(round(target_days)))
    horizon_steps = min(horizon_steps, max(1, len(df) // 6))

    feat_df = _build_feature_df(df, horizon_steps, profile["target_ret_min"]).dropna(subset=FEATURE_COLS + ["target"]).copy()
    n_rows = len(feat_df)
    if n_rows < 35:
        raise ValueError(f"{symbol} 有效回測樣本不足。")

    idx_dt = pd.to_datetime(feat_df.index)
    if getattr(idx_dt, "tz", None) is not None:
        idx_dt = idx_dt.tz_convert("UTC").tz_localize(None)
    feat_df = feat_df.copy()
    feat_df.index = idx_dt

    end_ts = feat_df.index[-1]
    cutoff_ts = end_ts - pd.Timedelta(days=backtest_days)
    window_candidates = np.where(feat_df.index >= cutoff_ts)[0]
    if len(window_candidates) == 0:
        window_candidates = np.arange(max(25, n_rows - 20), n_rows)

    min_train_rows = min(25, max(15, n_rows // 3))
    start_idx = max(min_train_rows, int(window_candidates[0]))
    stop_idx = max(start_idx + 1, n_rows - horizon_steps)
    if start_idx >= stop_idx:
        start_idx = max(min_train_rows, n_rows - horizon_steps - 5)
        stop_idx = max(start_idx + 1, n_rows - horizon_steps)

    if start_idx >= stop_idx:
        raise ValueError(f"{symbol} 最近 30 天可結算的 K 線不足。")

    if is_intraday_scalp:
        step_stride = 8 if (stop_idx - start_idx) > 200 else 3
    else:
        step_stride = 1

    window_start_str = str(feat_df.index[start_idx])[:16]
    window_end_str = str(feat_df.index[min(stop_idx, n_rows - 1)])[:16]

    trades = []
    signals_evaluated = 0
    retrain_every = 8 if is_intraday_scalp else 5
    max_eval_steps = 65
    clf = None
    steps_since_fit = 0

    for idx in range(start_idx, stop_idx, step_stride):
        if signals_evaluated >= max_eval_steps:
            break
        try:
            train_slice = feat_df.iloc[:idx]
            if len(train_slice) < min_train_rows:
                continue

            if clf is None or steps_since_fit >= retrain_every:
                clf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    min_samples_leaf=4,
                    random_state=42,
                    n_jobs=1,
                )
                clf.fit(train_slice[FEATURE_COLS], train_slice["target"])
                steps_since_fit = 0

            steps_since_fit += 1
            signals_evaluated += 1

            sample_x = feat_df[FEATURE_COLS].iloc[[idx]].ffill().bfill()
            if sample_x.isna().any().any():
                continue
            prob_up = float(clf.predict_proba(sample_x)[0][1])
            blended_prob = float(np.clip(prob_up, 0.08, 0.92))

            row = feat_df.iloc[idx]
            last_price = float(row["Close"])
            prev_price = float(feat_df.iloc[idx - 1]["Close"]) if idx >= 1 else last_price
            atr_val = float(row["atr14"]) if pd.notna(row["atr14"]) and row["atr14"] > 0 else last_price * 0.01
            rsi_now = float(row["rsi14"]) if pd.notna(row["rsi14"]) else 50.0

            win_slice = feat_df.iloc[max(0, idx - 19): idx + 1]
            high_20 = float(win_slice["High"].max())
            low_20 = float(win_slice["Low"].min())
            range_pos_pct = float(np.clip((last_price - low_20) / max(high_20 - low_20, 1e-8) * 100.0, 0.0, 100.0))

            direction, _, entry_price, stop_loss, take_profit, _, _ = _decide_signal(
                profile=profile,
                prob_up=prob_up,
                blended_prob=blended_prob,
                test_acc=0.55,
                news_score=0.0,
                last_price=last_price,
                prev_price=prev_price,
                atr_val=atr_val,
                rsi_now=rsi_now,
                range_pos_pct=range_pos_pct,
                low_20=low_20,
                lev_val=lev_val,
                is_intraday_scalp=is_intraday_scalp,
            )

            if direction not in ("偏多看漲", "偏空看跌"):
                continue

            future_bars = feat_df.iloc[idx + 1: idx + 1 + horizon_steps]
            if future_bars.empty:
                continue

            is_long = (direction == "偏多看漲")
            exit_price = float(future_bars["Close"].iloc[-1])

            for _, fbar in future_bars.iterrows():
                f_high = float(fbar["High"])
                f_low = float(fbar["Low"])
                if is_long:
                    if f_low <= stop_loss:
                        exit_price = stop_loss
                        break
                    if f_high >= take_profit:
                        exit_price = take_profit
                        break
                else:
                    if f_high >= stop_loss:
                        exit_price = stop_loss
                        break
                    if f_low <= take_profit:
                        exit_price = take_profit
                        break

            raw_ret = ((exit_price - entry_price) / max(entry_price, 1e-8)) if is_long else ((entry_price - exit_price) / max(entry_price, 1e-8))
            net_ret = (raw_ret - fee_cost) * lev_val
            if np.isfinite(net_ret):
                trades.append(float(net_ret))
        except Exception:
            continue

    stride_note = f"（1h 每 {step_stride} 根抽樣・每 {retrain_every} 步重訓）" if is_intraday_scalp else ""

    if not trades:
        return {
            "symbol": symbol,
            "market": market,
            "target_days": f"{target_days:g}",
            "backtest_days": backtest_days,
            "window_range": f"{window_start_str} ～ {window_end_str}",
            "leverage": round(lev_val, 2),
            "timeframe": f"1h K線（近30天極短線{stride_note}）" if is_intraday_scalp else "1d 日線（近30天）",
            "bars_tested": signals_evaluated,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "note": f"近 30 天（{window_start_str} ～ {window_end_str}）內符合進場條件的訊號較少。",
        }

    arr = np.array(trades, dtype=float)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    win_rate = round(float(len(wins) / len(arr) * 100.0), 1)
    avg_ret = round(float(np.mean(arr) * 100.0), 2)

    gross_win = float(np.sum(wins)) if len(wins) else 0.0
    gross_loss = abs(float(np.sum(losses))) if len(losses) else 0.0
    profit_factor = round(gross_win / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)

    equity = np.cumprod(1.0 + np.clip(arr, -0.95, 5.0))
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-8)
    max_dd_pct = round(float(np.max(dd) * 100.0), 2)

    return {
        "symbol": symbol,
        "market": market,
        "target_days": f"{target_days:g}",
        "backtest_days": backtest_days,
        "window_range": f"{window_start_str} ～ {window_end_str}",
        "leverage": round(lev_val, 2),
        "timeframe": f"1h K線（近30天極短線{stride_note}）" if is_intraday_scalp else "1d 日線（近30天）",
        "bars_tested": signals_evaluated,
        "trade_count": int(len(arr)),
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd_pct,
        "note": f"回測區間：{window_start_str} ～ {window_end_str}（近{backtest_days}天{stride_note}，含雙邊手續費 {fee_cost * 100:.2f}% 與 {lev_val:g}x 槓桿）。",
    }


def scan_high_win_recommendations(market: str, target_days=3, leverage=None, capital=None, top_n: int = 3):
    candidates = MARKET_PRESETS.get(market) or []
    results = []
    with ThreadPoolExecutor(max_workers=min(4, max(1, len(candidates)))) as pool:
        futs = [
            pool.submit(run_prediction, market, item["symbol"], target_days, leverage, capital)
            for item in candidates
        ]
        for fut in futs:
            try:
                r = fut.result()
                if r.get("high_win_qualified"):
                    results.append(r)
            except Exception:
                continue

    results.sort(key=lambda x: x.get("edge_score", 0.0), reverse=True)
    return results[:top_n]


@app.route("/")
def index():
    return render_template("index.html", presets_json=json.dumps(MARKET_PRESETS, ensure_ascii=False))


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        body = request.get_json(force=True, silent=True) or {}
        market = body.get("market", "加密貨幣")
        symbol = body.get("symbol", "BTC-USD")
        days = float(body.get("days", 3))
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


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    try:
        body = request.get_json(force=True, silent=True) or {}
        market = body.get("market", "加密貨幣")
        days = float(body.get("days", 3))
        cap_raw = str(body.get("capital", "")).strip()
        lev_raw = str(body.get("leverage", "")).strip()
        cap_val = float(cap_raw) if cap_raw else None
        lev_val = float(lev_raw) if lev_raw else None

        recs = scan_high_win_recommendations(
            market=market,
            target_days=days,
            leverage=lev_val,
            capital=cap_val,
            top_n=3,
        )
        return jsonify({
            "market": market,
            "target_days": f"{max(0.05, min(days, 15.0)):g}",
            "checked_at": datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S"),
            "criteria": "方向機率 ≥66%（或 ≤34% 且不追低空）＋ 驗證準確 ≥56% ＋ 盈虧比 ≥1.5 ＋ 新聞不打架",
            "recommendations": recs,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    try:
        body = request.get_json(force=True, silent=True) or {}
        market = body.get("market", "加密貨幣")
        symbol = body.get("symbol", "BTC-USD")
        days = float(body.get("days", 3))
        lev_raw = str(body.get("leverage", "")).strip()
        lev_val = float(lev_raw) if lev_raw else None

        res = run_backtest(
            market=market,
            raw_symbol=symbol,
            target_days=days,
            leverage=lev_val,
        )
        return jsonify(res)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.errorhandler(500)
def handle_500(exc):
    return jsonify({"error": f"伺服器內部錯誤：{exc}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
