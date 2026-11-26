import os
import json
import time
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

# 短期訊號快取（防抖動）：同標的 180 秒內避免因微幅跳價頻繁翻臉
_SIGNAL_CACHE = {}
_SIGNAL_CACHE_TTL_SEC = 180

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
        "track_name": "加密貨幣波段/長線軌（預測天數 ≥ 1天・全維度結構控損）",
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
        "allow_short": True,
    },
    "加密貨幣_極短線": {
        "track_name": "加密貨幣短線合約軌（預測天數 < 1天・防軋空緊控損）",
        "bull_threshold": 0.60,
        "bear_threshold": 0.39,
        "entry_atr_mult": 0.22,
        "stop_atr_mult": 0.90,
        "tp_atr_mult": 1.75,
        "target_ret_min": 0.003,
        "max_risk_pct": 0.005,
        "min_stop_pct": 0.005,
        "extreme_atr_pct": 0.032,
        "extreme_lev": 10.0,
        "extreme_move_pct": 5.0,
        "fee_rate": 0.0006,
        "allow_short": True,
    },
    "美股": {
        "track_name": "美股機構多因子波段軌（預測天數 ≥ 1天・技術/基本/籌碼/宏觀）",
        "bull_threshold": 0.55,
        "bear_threshold": 0.45,
        "entry_atr_mult": 0.30,
        "stop_atr_mult": 1.15,
        "tp_atr_mult": 2.00,
        "target_ret_min": 0.004,
        "max_risk_pct": 0.015,
        "min_stop_pct": 0.008,
        "extreme_atr_pct": 0.065,
        "extreme_lev": 3.0,
        "extreme_move_pct": 10.0,
        "fee_rate": 0.0005,
        "allow_short": True,
    },
    "美股_短線": {
        "track_name": "美股機構短線軌（預測天數 < 1天・嚴控波動與流動性）",
        "bull_threshold": 0.57,
        "bear_threshold": 0.43,
        "entry_atr_mult": 0.24,
        "stop_atr_mult": 0.95,
        "tp_atr_mult": 1.75,
        "target_ret_min": 0.0025,
        "max_risk_pct": 0.008,
        "min_stop_pct": 0.005,
        "extreme_atr_pct": 0.045,
        "extreme_lev": 5.0,
        "extreme_move_pct": 7.0,
        "fee_rate": 0.0005,
        "allow_short": True,
    },
    "台股": {
        "track_name": "台股機構全維度順勢軌（僅做多・不做空單・基本/籌碼/技術/宏觀）",
        "bull_threshold": 0.55,
        "bear_threshold": 0.45,
        "entry_atr_mult": 0.30,
        "stop_atr_mult": 1.15,
        "tp_atr_mult": 2.05,
        "target_ret_min": 0.004,
        "max_risk_pct": 0.015,
        "min_stop_pct": 0.008,
        "extreme_atr_pct": 0.065,
        "extreme_lev": 2.5,
        "extreme_move_pct": 9.5,
        "fee_rate": 0.001425,
        "allow_short": False,
    },
    "台股_短線": {
        "track_name": "台股短線顺勢軌（僅做多・不做空單・嚴控回測與跳空風險）",
        "bull_threshold": 0.57,
        "bear_threshold": 0.43,
        "entry_atr_mult": 0.24,
        "stop_atr_mult": 0.95,
        "tp_atr_mult": 1.80,
        "target_ret_min": 0.0025,
        "max_risk_pct": 0.008,
        "min_stop_pct": 0.005,
        "extreme_atr_pct": 0.045,
        "extreme_lev": 3.0,
        "extreme_move_pct": 7.0,
        "fee_rate": 0.001425,
        "allow_short": False,
    },
}


def is_short_term_contract(target_days: float) -> bool:
    """預測天數小於 1 天才視為短線合約；1 天以上歸類為長線／波段。"""
    return float(target_days) < 1.0


def get_track_profile(market: str, target_days: float) -> dict:
    short_mode = is_short_term_contract(target_days)
    if market == "加密貨幣":
        return TRACK_PROFILES["加密貨幣_極短線"] if short_mode else TRACK_PROFILES["加密貨幣"]
    if market == "台股":
        return TRACK_PROFILES["台股_短線"] if short_mode else TRACK_PROFILES["台股"]
    return TRACK_PROFILES["美股_短線"] if short_mode else TRACK_PROFILES["美股"]


def normalize_symbol(market: str, raw_symbol: str) -> str:
    sym = (raw_symbol or "").strip().upper()
    if not sym:
        if market == "加密貨幣":
            return "BTC-USD"
        if market == "台股":
            return "2330.TW"
        return "NVDA"
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
    upper_wick_ratio = round(float(upper_wick / total_range), 3)
    lower_wick_ratio = round(float(lower_wick / total_range), 3)

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
        "upper_wick_ratio": upper_wick_ratio,
        "lower_wick_ratio": lower_wick_ratio,
        "range_label": range_label,
        "summary": f"{ma_structure}｜{candle_signal}｜{range_label}區間 {pos_in_range}%",
    }


def fetch_chip_and_flow_context(market: str, symbol: str, df: pd.DataFrame, macro_info: dict, news_score: float) -> dict:
    """
    第五維／籌碼與市場結構面：
    - 台股：整合外資／投信／三大法人與融資借券氛圍、量價結構、大盤連動（TWII）與限空流動性環境
    - 美股：整合機構風險偏好／去槓桿與擴散環境、量能結構、SPY 市場廣度代理指標
    - 加密貨幣：整合槓桿波動與量價結構
    """
    vol_ratio_now = float(df["vol_ratio"].iloc[-1]) if "vol_ratio" in df.columns and pd.notna(df["vol_ratio"].iloc[-1]) else 1.0
    ret_5d_now = float(df["ret_5d"].iloc[-1]) if "ret_5d" in df.columns and pd.notna(df["ret_5d"].iloc[-1]) else 0.0

    if market == "台股":
        twii_val, twii_chg = _safe_last_and_chg("^TWII")
        score = 0.0
        notes = []

        if twii_val is not None:
            if twii_chg >= 0.8:
                score += 0.25
                notes.append(f"加權指數走強 ({twii_chg:+.2f}%) 法人偏多回補")
            elif twii_chg <= -1.2:
                score -= 0.30
                notes.append(f"加權指數重挫 ({twii_chg:+.2f}%) 留意融資與法人調節")
            else:
                notes.append(f"加權指數 {twii_val:.0f} ({twii_chg:+.2f}%)")

        if vol_ratio_now >= 1.35 and ret_5d_now > 0:
            score += 0.20
            notes.append(f"量能放大 ({vol_ratio_now:.2f}x) 吸籌動能較佳")
        elif vol_ratio_now < 0.75:
            score -= 0.10
            notes.append(f"量能偏縮 ({vol_ratio_now:.2f}x) 追價力道保守")

        if news_score >= 0.15:
            score += 0.10
            notes.append("題材與法人買盤預期偏正面")
        elif news_score <= -0.15:
            score -= 0.15
            notes.append("消息面偏空，留意投信/外資換手與借券壓力")

        score = float(np.clip(score, -1.0, 1.0))
        status = "籌碼與量價偏多（吸籌/順勢）" if score >= 0.20 else ("籌碼與流動性偏緊（防調節）" if score <= -0.20 else "籌碼與量價中性")
        return {
            "score": round(score, 3),
            "status": status,
            "summary": "｜".join(notes) if notes else "台股籌碼與量價中性",
            "short_allowed": False,
        }

    if market == "美股":
        spy_val, spy_chg = _safe_last_and_chg("SPY")
        score = 0.0
        notes = []

        if spy_val is not None:
            if spy_chg >= 0.6:
                score += 0.20
                notes.append(f"SPY 走揚 ({spy_chg:+.2f}%) 機構風險偏好回升")
            elif spy_chg <= -1.0:
                score -= 0.25
                notes.append(f"SPY 回檔 ({spy_chg:+.2f}%) 留意對沖基金去槓桿/擴散")
            else:
                notes.append(f"SPY {spy_val:.2f} ({spy_chg:+.2f}%)")

        if macro_info.get("risk_off"):
            score -= 0.25
            notes.append("宏觀風險升溫，機構部位偏向防禦與減槓桿")

        if vol_ratio_now >= 1.4 and ret_5d_now > 0.01:
            score += 0.15
            notes.append(f"成交量放大 ({vol_ratio_now:.2f}x) 資金流入明確")
        elif vol_ratio_now >= 1.5 and ret_5d_now < -0.02:
            score -= 0.20
            notes.append(f"放量下跌 ({vol_ratio_now:.2f}x) 機構派發/去槓桿壓力")

        score = float(np.clip(score, -1.0, 1.0))
        status = "機構資金偏多" if score >= 0.20 else ("機構去槓桿/防禦警戒" if score <= -0.20 else "機構籌碼中性")
        return {
            "score": round(score, 3),
            "status": status,
            "summary": "｜".join(notes) if notes else "美股機構與量能結構中性",
            "short_allowed": True,
        }

    # 加密貨幣市場結構與量能
    score = 0.0
    notes = []
    if vol_ratio_now >= 1.45 and ret_5d_now > 0.02:
        score += 0.20
        notes.append(f"成交量放大 ({vol_ratio_now:.2f}x) 動能推升")
    elif vol_ratio_now >= 1.45 and ret_5d_now < -0.02:
        score -= 0.20
        notes.append(f"放量下殺 ({vol_ratio_now:.2f}x) 留意清算與洗盤")
    else:
        notes.append(f"量能比 {vol_ratio_now:.2f}x")

    score = float(np.clip(score, -1.0, 1.0))
    status = "量能動能偏多" if score >= 0.20 else ("洗盤/清算壓力偏高" if score <= -0.20 else "量能結構中性")
    return {
        "score": round(score, 3),
        "status": status,
        "summary": "｜".join(notes),
        "short_allowed": True,
    }


def build_entry_triggers(
    market: str,
    target_days: float,
    last_price: float,
    ma20_now: float,
    ma60_now: float,
    high_20: float,
    low_20: float,
    rsi_now: float,
    range_pos_pct: float,
    allow_short: bool,
) -> dict:
    """不管長線或短線，都直接給出開多、開空（台股明確不做空）與觀望的具體訊號條件。"""
    short_mode = is_short_term_contract(target_days)
    mode_label = f"{market}・{'短線（< 1天）' if short_mode else '波段／長線（≥ 1天）'}"

    if market == "台股":
        long_signal = (
            f"站穩 {ma20_now:.4f} 或回測 {ma20_now:.4f}／{ma60_now:.4f} 支撐守穩不破、出現下影線或止跌紅K收盤，"
            f"且量價與法人/基本面結構偏多（目前區間 {range_pos_pct:.1f}%、RSI {rsi_now:.1f}），再考慮分批偏多。"
        )
        short_signal = "台股策略設定為「僅做多、不做空單」：遇跌破支撐或高檔轉弱時一律減碼／觀望，不開空單。"
        wait_signal = (
            f"股價卡在 {low_20:.4f} ～ {high_20:.4f} 中段震盪、量縮或尚未守穩關鍵均線時先別開單，等支撐確認再出手。"
        )
        return {
            "is_short_contract": short_mode,
            "horizon_category": mode_label,
            "allow_short": False,
            "long_signal": long_signal,
            "short_signal": short_signal,
            "wait_signal": wait_signal,
        }

    if short_mode:
        long_signal = (
            f"回測支撐（如 {ma20_now:.4f} 或 {low_20:.4f} 附近）守穩不破，出現下影線收回或止跌 K 線收線，"
            f"且 RSI 由低檔回升（避免追在急拉長上影線），再考慮開多單。"
        )
        short_signal = (
            f"跌破 {ma20_now:.4f} 或關鍵支撐且 K 線實體收在下方、反彈測壓不過，或衝高至 {high_20:.4f} 附近出現長上影線與漲勢停滯，"
            f"且未處於均線多頭排列與下檔強承接時，再考慮開空單。"
        )
        wait_signal = (
            f"價格卡在 {low_20:.4f} ～ {high_20:.4f} 中間震盪、量縮或大 K 線尚未收線時先別開單，等條件到位再出手。"
        )
    else:
        long_signal = (
            f"趨勢結構站穩 {ma20_now:.4f} 上方，或回踩 {ma20_now:.4f}／{ma60_now:.4f} 支撐區止跌收紅、"
            f"估值與區間位置不過熱（目前 {range_pos_pct:.1f}%，RSI {rsi_now:.1f}），再分批偏多布局。"
        )
        short_signal = (
            f"跌破 {ma20_now:.4f} 且反彈受阻轉弱，或靠近 {high_20:.4f} 阻力區出現明顯上影線／動能背離且跌勢確認，"
            f"不逆勢硬空底部或強勢多頭排列，再考慮偏空操作。"
        )
        wait_signal = (
            f"均線糾結或處於 {low_20:.4f} ～ {high_20:.4f} 區間中段無明確突破／跌破收盤時，先等方向表態再出手。"
        )

    return {
        "is_short_contract": short_mode,
        "horizon_category": mode_label,
        "allow_short": allow_short,
        "long_signal": long_signal,
        "short_signal": short_signal,
        "wait_signal": wait_signal,
    }


def build_exit_alert(
    chg_pct: float,
    rsi_now: float,
    range_pos_pct: float,
    upper_wick_ratio: float,
    lower_wick_ratio: float,
    is_short_contract_mode: bool,
    take_profit: float,
    last_price: float,
    allow_short: bool = True,
) -> dict:
    spike_thresh = 1.8 if is_short_contract_mode else 3.5
    if (
        (range_pos_pct >= 78.0 and (upper_wick_ratio >= 0.35 or rsi_now >= 70.0))
        or (chg_pct >= spike_thresh and upper_wick_ratio >= 0.30)
    ):
        return {
            "level": "high",
            "title": "多單留意分批止盈／上移停利",
            "reason": f"處於區間高檔或急拉後出現上影線賣壓（區間 {range_pos_pct:.1f}%、上影線 {upper_wick_ratio*100:.0f}%、RSI {rsi_now:.1f}），漲勢若停滯建議先減碼鎖利，目標參考 {take_profit}。",
        }
    if allow_short and range_pos_pct <= 22.0 and (lower_wick_ratio >= 0.35 or rsi_now <= 32.0):
        return {
            "level": "medium",
            "title": "空單留意回補止盈／防反彈",
            "reason": f"靠近低檔支撐或下影線承接（區間 {range_pos_pct:.1f}%、下影線 {lower_wick_ratio*100:.0f}%、RSI {rsi_now:.1f}），若跌勢放緩建議分批回補空單。",
        }
    return {
        "level": "normal",
        "title": "尚未觸發急轉止盈警戒",
        "reason": f"目前區間位置 {range_pos_pct:.1f}%、RSI {rsi_now:.1f}，請持續對照停損／停利與實際盤面量價變化。",
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


def _safe_last_and_chg(ticker_sym: str):
    try:
        hist = yf.Ticker(ticker_sym).history(period="5d", interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            return None, 0.0
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        closes = hist["Close"].dropna()
        if len(closes) == 0:
            return None, 0.0
        last_v = float(closes.iloc[-1])
        prev_v = float(closes.iloc[-2]) if len(closes) >= 2 else last_v
        chg_pct = ((last_v - prev_v) / prev_v * 100.0) if prev_v else 0.0
        return last_v, chg_pct
    except Exception:
        return None, 0.0


def fetch_macro_context(market: str) -> dict:
    with ThreadPoolExecutor(max_workers=3) as pool:
        vix_fut = pool.submit(_safe_last_and_chg, "^VIX")
        dxy_fut = pool.submit(_safe_last_and_chg, "DX-Y.NYB")
        tnx_fut = pool.submit(_safe_last_and_chg, "^TNX")
        vix_val, vix_chg = vix_fut.result()
        dxy_val, dxy_chg = dxy_fut.result()
        tnx_val, tnx_chg = tnx_fut.result()

    score = 0.0
    notes = []

    if vix_val is not None:
        if vix_val >= 28.0 or vix_chg >= 8.0:
            score -= 0.45
            notes.append(f"VIX 偏高 ({vix_val:.1f}, {vix_chg:+.1f}%) 風險升溫")
        elif vix_val >= 22.0:
            score -= 0.20
            notes.append(f"VIX 中高檔 ({vix_val:.1f}) 波動警戒")
        elif vix_val <= 15.5:
            score += 0.25
            notes.append(f"VIX 溫和 ({vix_val:.1f}) 風險偏好較穩")
        else:
            notes.append(f"VIX {vix_val:.1f} ({vix_chg:+.1f}%)")

    if dxy_val is not None:
        if dxy_chg >= 0.6:
            score -= 0.25 if market == "加密貨幣" else 0.15
            notes.append(f"美元走強 ({dxy_chg:+.2f}%) 壓抑風險資產")
        elif dxy_chg <= -0.6:
            score += 0.25 if market == "加密貨幣" else 0.15
            notes.append(f"美元轉弱 ({dxy_chg:+.2f}%) 流動性較友善")

    if tnx_val is not None:
        if tnx_chg >= 2.5:
            score -= 0.20
            notes.append(f"美債殖利率彈升 ({tnx_val:.2f}%, {tnx_chg:+.1f}%) 估值承壓")
        elif tnx_chg <= -2.5:
            score += 0.15
            notes.append(f"美債殖利率回落 ({tnx_val:.2f}%, {tnx_chg:+.1f}%)")

    score = float(np.clip(score, -1.0, 1.0))
    if score >= 0.20:
        regime = "偏多順風"
    elif score <= -0.25:
        regime = "偏空逆風/高警戒"
    else:
        regime = "中性震盪"

    summary = "｜".join(notes) if notes else "宏觀指標中性"
    return {
        "score": round(score, 3),
        "regime": regime,
        "summary": summary,
        "vix": round(vix_val, 2) if vix_val is not None else None,
        "dxy_chg_pct": round(dxy_chg, 2) if dxy_val is not None else None,
        "tnx_yield": round(tnx_val, 2) if tnx_val is not None else None,
        "risk_off": score <= -0.35 or (vix_val is not None and vix_val >= 28.0),
    }


def fetch_fundamental_context(market: str, symbol: str, prob_up: float, ma20_gap: float) -> dict:
    if market == "加密貨幣":
        if ma20_gap <= -0.12 and prob_up >= 0.55:
            score = 0.25
            status = "偏多（深蹲折價修復）"
        elif ma20_gap >= 0.15:
            score = -0.25
            status = "偏高（短線乖離偏大）"
        else:
            score = float(np.clip(ma20_gap * 1.5, -0.3, 0.3))
            status = "中性（以量價與籌碼結構為主）"
        return {
            "score": round(score, 3),
            "status": status,
            "summary": f"加密資產以均線乖離與量價結構評估：{status}",
            "pe": None,
            "roe_pct": None,
            "pb": None,
        }

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    pe = info.get("forwardPE") or info.get("trailingPE")
    pb = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    rev_growth = info.get("revenueGrowth")
    debt_to_equity = info.get("debtToEquity")

    score = 0.0
    parts = []

    if isinstance(pe, (int, float)) and np.isfinite(pe) and pe > 0:
        if pe <= 18:
            score += 0.25
            parts.append(f"PE {pe:.1f}x（估值合理/偏低）")
        elif pe >= 45:
            score -= 0.25
            parts.append(f"PE {pe:.1f}x（估值偏高）")
        else:
            parts.append(f"PE {pe:.1f}x")

    if isinstance(roe, (int, float)) and np.isfinite(roe):
        roe_pct = roe * 100.0 if abs(roe) <= 2.5 else float(roe)
        if roe_pct >= 15.0:
            score += 0.25
            parts.append(f"ROE {roe_pct:.1f}%（獲利穩健）")
        elif roe_pct < 5.0:
            score -= 0.20
            parts.append(f"ROE {roe_pct:.1f}%（獲利偏弱）")
        else:
            parts.append(f"ROE {roe_pct:.1f}%")
    else:
        roe_pct = None

    if isinstance(pb, (int, float)) and np.isfinite(pb) and pb > 0:
        if pb <= 2.0:
            score += 0.15
            parts.append(f"PB {pb:.2f}x")
        elif pb >= 12.0:
            score -= 0.15
            parts.append(f"PB {pb:.2f}x（淨值溢價高）")

    if isinstance(rev_growth, (int, float)) and np.isfinite(rev_growth):
        rg_pct = rev_growth * 100.0 if abs(rev_growth) <= 3.0 else float(rev_growth)
        if rg_pct >= 15.0:
            score += 0.15
            parts.append(f"營收成長 {rg_pct:+.1f}%")
        elif rg_pct <= -10.0:
            score -= 0.20
            parts.append(f"營收衰退 {rg_pct:+.1f}%")

    if isinstance(debt_to_equity, (int, float)) and np.isfinite(debt_to_equity) and debt_to_equity > 250:
        score -= 0.10
        parts.append(f"負債比偏高 ({debt_to_equity:.0f})")

    score = float(np.clip(score, -1.0, 1.0))
    if score >= 0.25:
        status = "基本面偏強"
    elif score <= -0.25:
        status = "基本面偏弱/估值緊繃"
    else:
        status = "基本面中性"

    summary = "｜".join(parts) if parts else "基本面財務數據中性或為 ETF/指數標的"
    return {
        "score": round(score, 3),
        "status": status,
        "summary": summary,
        "pe": round(float(pe), 1) if isinstance(pe, (int, float)) and np.isfinite(pe) and pe > 0 else None,
        "roe_pct": round(float(roe_pct), 1) if roe_pct is not None else None,
        "pb": round(float(pb), 2) if isinstance(pb, (int, float)) and np.isfinite(pb) and pb > 0 else None,
    }


def score_technical(prob_up: float, ma20_now: float, ma60_now: float, last_price: float, rsi_now: float, range_pos_pct: float) -> dict:
    ml_bias = (prob_up - 0.5) * 1.4
    trend_bias = 0.0
    if last_price > ma20_now and ma20_now > ma60_now:
        trend_bias = 0.30
    elif last_price < ma20_now and ma20_now < ma60_now:
        trend_bias = -0.30
    elif last_price >= ma20_now:
        trend_bias = 0.12
    else:
        trend_bias = -0.12

    pos_bias = 0.0
    if range_pos_pct <= 18.0:
        pos_bias = 0.12
    elif range_pos_pct >= 82.0:
        pos_bias = -0.12

    score = float(np.clip(ml_bias + trend_bias + pos_bias, -1.0, 1.0))
    if score >= 0.22:
        status = "技術面偏多"
    elif score <= -0.22:
        status = "技術面偏空"
    else:
        status = "技術面震盪"
    return {
        "score": round(score, 3),
        "status": status,
        "summary": f"ML看漲 {prob_up*100:.1f}%｜RSI {rsi_now:.1f}｜區間 {range_pos_pct:.1f}%",
    }


def score_sentiment(news_score: float, rsi_now: float, chg_pct: float) -> dict:
    rsi_adj = 0.0
    if rsi_now >= 75.0:
        rsi_adj = -0.25
    elif rsi_now <= 25.0:
        rsi_adj = 0.20

    chg_adj = float(np.clip(chg_pct * 0.03, -0.25, 0.25))
    score = float(np.clip(news_score * 0.65 + rsi_adj + chg_adj * 0.4, -1.0, 1.0))

    if rsi_now >= 75.0:
        status = "情緒偏熱（防追高）"
    elif rsi_now <= 25.0:
        status = "情緒偏冷/超賣（留意止跌與反轉）"
    elif score >= 0.20:
        status = "情緒偏多"
    elif score <= -0.20:
        status = "情緒偏空"
    else:
        status = "情緒中性"

    return {
        "score": round(score, 3),
        "status": status,
        "summary": f"新聞分數 {news_score:+.2f}｜短線動能 {chg_pct:+.2f}%｜RSI {rsi_now:.1f}",
    }


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
    high = out["High"]
    low = out["Low"]
    vol = out["Volume"].replace(0, np.nan)
    vol_ma20 = vol.rolling(20).mean()

    high_20 = high.rolling(20).max()
    low_20 = low.rolling(20).min()
    bar_range = (high - low).replace(0, np.nan)
    upper_wick = high - np.maximum(out["Open"], close)
    lower_wick = np.minimum(out["Open"], close) - low

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
    out["range_pos_20"] = ((close - low_20) / (high_20 - low_20).replace(0, np.nan)).clip(0.0, 1.0).fillna(0.5)
    out["upper_wick_ratio"] = (upper_wick / bar_range).clip(0.0, 1.0).fillna(0.0)
    out["lower_wick_ratio"] = (lower_wick / bar_range).clip(0.0, 1.0).fillna(0.0)

    out["future_ret"] = close.shift(-horizon_steps) / close - 1.0
    out["target"] = (out["future_ret"] > target_ret_min).astype(int)
    return out


FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d",
    "ma10_gap", "ma20_gap", "ma20_slope",
    "rsi14", "macd_hist", "vol_20", "vol_ratio", "atr_pct",
    "range_pos_20", "upper_wick_ratio", "lower_wick_ratio"
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


def _compute_composite_score(market: str, tech_score: float, fund_score: float, sent_score: float, macro_score: float, flow_score: float) -> float:
    """台股、美股、加密貨幣各自採用不同的全維度加權。"""
    if market == "台股":
        raw = (
            tech_score * 0.25
            + fund_score * 0.25
            + flow_score * 0.25
            + macro_score * 0.15
            + sent_score * 0.10
        )
    elif market == "美股":
        raw = (
            tech_score * 0.25
            + fund_score * 0.20
            + flow_score * 0.20
            + macro_score * 0.20
            + sent_score * 0.15
        )
    else:
        raw = (
            tech_score * 0.35
            + flow_score * 0.20
            + macro_score * 0.20
            + sent_score * 0.15
            + fund_score * 0.10
        )
    return round(float(np.clip(raw, -1.0, 1.0)), 3)


def _decide_signal(
    market: str,
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
    ma10_now: float,
    ma20_now: float,
    ma60_now: float,
    lev_val: float,
    is_short_contract_mode: bool,
    lower_wick_ratio: float = 0.0,
    tech_score: float = 0.0,
    fund_score: float = 0.0,
    sent_score: float = 0.0,
    macro_score: float = 0.0,
    flow_score: float = 0.0,
    macro_risk_off: bool = False,
):
    chg_pct = ((last_price - prev_price) / prev_price * 100.0) if prev_price else 0.0
    atr_pct_now = atr_val / max(last_price, 1e-8)
    allow_short = bool(profile.get("allow_short", True)) and (market != "台股")

    composite_score = _compute_composite_score(
        market=market,
        tech_score=tech_score,
        fund_score=fund_score,
        sent_score=sent_score,
        macro_score=macro_score,
        flow_score=flow_score,
    )

    uptrend_ma_stack = (last_price > ma20_now) and (ma20_now > ma60_now)
    broken_downtrend_structure = (last_price < ma20_now) and (ma20_now < ma60_now)
    sweep_and_reclaim = (range_pos_pct <= 15.0) and (last_price > low_20) and (chg_pct > -0.3)
    spike_fade_risk = (chg_pct >= (2.2 if is_short_contract_mode else 3.8)) and (range_pos_pct >= 65.0 or rsi_now >= 68.0)
    spike_whipsaw = (abs(chg_pct) >= (2.0 if is_short_contract_mode else 4.0)) and (atr_pct_now >= profile["extreme_atr_pct"] * 0.72)

    short_squeeze_or_bounce_risk = (
        uptrend_ma_stack
        or range_pos_pct <= 28.0
        or rsi_now <= 38.0
        or lower_wick_ratio >= 0.35
        or (last_price >= ma10_now and chg_pct > 0.25)
        or chg_pct >= (1.5 if is_short_contract_mode else 4.5)
    )

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
    macro_headwind_block_long = macro_risk_off and (composite_score < 0.05) and (range_pos_pct > 20.0)

    entry_mult = profile["entry_atr_mult"]
    stop_mult = profile["stop_atr_mult"]
    tp_mult = profile["tp_atr_mult"]

    is_bottom_zone = (
        (range_pos_pct <= 8.0 and rsi_now <= 24.0)
        if is_short_contract_mode
        else ((range_pos_pct <= 12.0) or (rsi_now <= 25.0))
    )
    near_resistance_stall = (range_pos_pct >= 82.0 and chg_pct <= 0.4)

    if extreme_high_risk or indecisive_conflict:
        direction = "建議先不做單"
        entry_price = round(last_price, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(tp_mult / max(stop_mult, 1e-8), 2)
        action = "波動/槓桿風險過高，易出現多空雙殺與插針，建議先空倉觀望" if extreme_high_risk else "訊號處於五五波與多空拉鋸，建議先不做單"
    elif spike_whipsaw:
        direction = "急動插針先等收穩"
        entry_price = round(last_price - np.sign(chg_pct) * max(entry_mult * 1.3, 0.35) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4) if (chg_pct < 0 or not allow_short) else round(entry_price + stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4) if (chg_pct < 0 or not allow_short) else round(entry_price - tp_mult * atr_val, 4)
        rr_ratio = round(tp_mult / max(stop_mult, 1e-8), 2)
        action = f"短線急動易有第一根假針，不追單，等K線收穩再看 {entry_price} 附近"
    elif spike_fade_risk and blended_prob >= profile["bull_threshold"]:
        direction = "急拉防回落與二探"
        entry_price = round(last_price - max(entry_mult * 1.35, 0.35) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(max(take_profit - entry_price, 0.0) / max(entry_price - stop_loss, 1e-8), 2)
        action = f"短線急拉後留意阻力回落與二探，不盲目追高，等回測/二探守穩 {entry_price} 再看"
    elif blended_prob >= profile["bull_threshold"] and macro_headwind_block_long:
        direction = "宏觀逆風留意回落"
        entry_price = round(last_price - max(entry_mult * 1.4, 0.35) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(max(take_profit - entry_price, 0.0) / max(entry_price - stop_loss, 1e-8), 2)
        action = f"宏觀逆風/風險偏緊，反彈不戀戰，等回測 {entry_price} 再評估"
    elif blended_prob >= profile["bull_threshold"] and near_resistance_stall:
        direction = "高檔阻力留意回落"
        entry_price = round(last_price - max(entry_mult * 1.25, 0.30) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(max(take_profit - entry_price, 0.0) / max(entry_price - stop_loss, 1e-8), 2)
        action = f"接近區間高檔阻力但動能未加速，防假突破，等回測 {entry_price} 再評估"
    elif blended_prob >= profile["bull_threshold"]:
        direction = "偏多看漲"
        if broken_downtrend_structure and not is_bottom_zone:
            entry_price = round(last_price - max(entry_mult * 1.4, 0.35) * atr_val, 4)
            entry_hint = f"空頭結構未完全修復，反彈不盲目戀戰，等回測/二探 {entry_price}"
        elif is_bottom_zone or (range_pos_pct <= 18.0 and rsi_now <= 32.0):
            entry_price = round(last_price, 4)
            entry_hint = f"假跌破收回或二探有止跌跡象，可評估現價 {entry_price} 附近分批" if sweep_and_reclaim else f"處於低檔止跌區，可評估現價 {entry_price} 附近進場"
        else:
            entry_price = round(last_price - max(entry_mult, 0.25) * atr_val, 4)
            entry_hint = f"全維度偏多（綜合 {composite_score:+.2f}），防衝高假突破，等回測 {entry_price} 掛單"
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(max(take_profit - entry_price, 0.0) / max(entry_price - stop_loss, 1e-8), 2)
        action = f"{entry_hint}（停損 {stop_mult:.2f} ATR / 停利 {tp_mult:.2f} ATR，盈虧比 1:{rr_ratio}）"
    elif blended_prob <= profile["bear_threshold"] and not allow_short:
        # 台股不做空單：空方訊號一律轉為觀望／等支撐止跌
        direction = "台股不做空（觀望等支撐）"
        entry_price = round(last_price - max(entry_mult * 1.35, 0.35) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(max(take_profit - entry_price, 0.0) / max(entry_price - stop_loss, 1e-8), 2)
        action = f"台股設定為不做空單，目前偏弱/震盪整理（綜合 {composite_score:+.2f}），不開空，等回測 {entry_price} 附近止跌再評估偏多"
    elif blended_prob <= profile["bear_threshold"] and allow_short and not short_squeeze_or_bounce_risk:
        direction = "偏空看跌"
        if range_pos_pct >= 85.0 and rsi_now >= 70.0 and chg_pct <= 0.2:
            entry_price = round(last_price, 4)
            entry_hint = f"處於高檔轉弱區，反彈不戀戰，可評估現價 {entry_price} 附近偏空"
        else:
            entry_price = round(last_price + max(entry_mult * 1.15, 0.30) * atr_val, 4)
            entry_hint = f"全維度偏空（綜合 {composite_score:+.2f}），防短線反彈軋空，等反彈受阻至 {entry_price} 再評估"
        stop_loss = round(entry_price + stop_mult * atr_val, 4)
        take_profit = round(entry_price - tp_mult * atr_val, 4)
        rr_ratio = round(max(entry_price - take_profit, 0.0) / max(stop_loss - entry_price, 1e-8), 2)
        action = f"{entry_hint}（停損 {stop_mult:.2f} ATR / 停利 {tp_mult:.2f} ATR，盈虧比 1:{rr_ratio}）"
    elif blended_prob <= profile["bear_threshold"] and allow_short and short_squeeze_or_bounce_risk:
        direction = "防反彈軋空（不追空）"
        entry_price = round(last_price + max(entry_mult * 1.35, 0.45) * atr_val, 4)
        stop_loss = round(entry_price + stop_mult * atr_val, 4)
        take_profit = round(entry_price - tp_mult * atr_val, 4)
        rr_ratio = round(max(entry_price - take_profit, 0.0) / max(stop_loss - entry_price, 1e-8), 2)
        if uptrend_ma_stack:
            action = f"均線仍呈多頭排列且有下影線承接，目前不建議直接開空；若要等空，需等反彈受阻至 {entry_price} 附近且轉弱再評估"
        else:
            action = f"短線易有反彈或下影線承接，不盲目追空；若等反彈至 {entry_price} 受阻再看（盈虧比 1:{rr_ratio}）"
    else:
        direction = "震盪中性"
        entry_price = round(last_price, 4) if is_bottom_zone else round(last_price - max(entry_mult, 0.25) * atr_val, 4)
        stop_loss = round(entry_price - stop_mult * atr_val, 4)
        take_profit = round(entry_price + tp_mult * atr_val, 4)
        rr_ratio = round(tp_mult / max(stop_mult, 1e-8), 2)
        no_short_note = "（台股不做空）" if not allow_short else ""
        action = f"全維度綜合 {composite_score:+.2f}{no_short_note}（區間磨底/上下洗盤，控管槓桿與倉位），參考點位 {entry_price}（盈虧比 1:{rr_ratio}）"

    return direction, action, entry_price, stop_loss, take_profit, rr_ratio, chg_pct, composite_score


def run_prediction(market: str, raw_symbol: str, target_days=3, leverage=None, capital=None):
    raw_days = float(target_days if target_days is not None else 3)
    target_days = max(0.05, min(raw_days, 15.0))
    short_mode = is_short_term_contract(target_days)
    profile = get_track_profile(market, target_days)
    symbol = normalize_symbol(market, raw_symbol)
    lev_val = float(leverage) if (leverage is not None and leverage > 0) else 1.0

    cache_key = (market, symbol, round(target_days, 4), round(lev_val, 2))
    now_ts = time.time()
    cached = _SIGNAL_CACHE.get(cache_key)
    if cached and (now_ts - cached["ts"]) < _SIGNAL_CACHE_TTL_SEC:
        res_copy = json.loads(json.dumps(cached["payload"]))
        live_price = fetch_live_price(symbol)
        if live_price is not None and live_price > 0:
            res_copy["last_price"] = round(live_price, 4)
        res_copy["updated_at"] = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y-%m-%d %H:%M:%S")
        res_copy["cache_note"] = "已啟用 3 分鐘防抖鎖定，避免短線微幅跳價造成訊號反覆"
        return res_copy

    use_intraday_bars = (market == "加密貨幣" and short_mode)
    horizon_steps = max(2, int(round(target_days * 24))) if use_intraday_bars else max(1, int(round(target_days)))

    with ThreadPoolExecutor(max_workers=4) as pool:
        hist_fut = pool.submit(fetch_history_df, symbol, use_intraday_bars)
        news_fut = pool.submit(fetch_news_headlines, symbol, 8)
        live_fut = pool.submit(fetch_live_price, symbol)
        macro_fut = pool.submit(fetch_macro_context, market)
        df = hist_fut.result()
        headlines = news_fut.result()
        live_price = live_fut.result()
        macro_info = macro_fut.result()

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

    last_price = live_price if live_price is not None else float(feat_df["Close"].iloc[-1])
    prev_price = float(feat_df["Close"].iloc[-2]) if len(feat_df) >= 2 else last_price
    chg_1d_pct = ((last_price - prev_price) / prev_price * 100.0) if prev_price else 0.0

    range_label = "20根1h" if use_intraday_bars else "20日"
    structure_info = analyze_price_structure(feat_df, last_price, range_label=range_label)
    rsi_now = float(feat_df["rsi14"].iloc[-1]) if pd.notna(feat_df["rsi14"].iloc[-1]) else 50.0
    ma10_now = float(feat_df["ma10"].iloc[-1]) if pd.notna(feat_df["ma10"].iloc[-1]) else last_price
    ma20_now = float(feat_df["ma20"].iloc[-1]) if pd.notna(feat_df["ma20"].iloc[-1]) else last_price
    ma60_now = float(feat_df["ma60"].iloc[-1]) if pd.notna(feat_df["ma60"].iloc[-1]) else last_price
    ma20_gap_now = (last_price - ma20_now) / max(ma20_now, 1e-8)

    news_info = score_news(headlines)
    fund_info = fetch_fundamental_context(market, symbol, prob_up, ma20_gap_now)
    tech_info = score_technical(prob_up, ma20_now, ma60_now, last_price, rsi_now, structure_info["range_pos_pct"])
    sent_info = score_sentiment(news_info["score"], rsi_now, chg_1d_pct)
    flow_info = fetch_chip_and_flow_context(market, symbol, feat_df, macro_info, news_info["score"])

    composite_raw = _compute_composite_score(
        market=market,
        tech_score=tech_info["score"],
        fund_score=fund_info["score"],
        sent_score=sent_info["score"],
        macro_score=macro_info["score"],
        flow_score=flow_info["score"],
    )
    blended_prob = float(np.clip(prob_up + composite_raw * 0.14, 0.08, 0.92))

    atr_val = float(feat_df["atr14"].iloc[-1]) if pd.notna(feat_df["atr14"].iloc[-1]) else last_price * 0.01
    vol_val = float(feat_df["vol_20"].iloc[-1]) if pd.notna(feat_df["vol_20"].iloc[-1]) else 0.01
    expected_move_pct = float(vol_val * np.sqrt(horizon_steps) * 100.0)

    direction, action, entry_price, stop_loss, take_profit, rr_ratio, _, composite_score = _decide_signal(
        market=market,
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
        ma10_now=ma10_now,
        ma20_now=ma20_now,
        ma60_now=ma60_now,
        lev_val=lev_val,
        is_short_contract_mode=short_mode,
        lower_wick_ratio=structure_info["lower_wick_ratio"],
        tech_score=tech_info["score"],
        fund_score=fund_info["score"],
        sent_score=sent_info["score"],
        macro_score=macro_info["score"],
        flow_score=flow_info["score"],
        macro_risk_off=macro_info["risk_off"],
    )

    allow_short = bool(profile.get("allow_short", True)) and (market != "台股")
    entry_triggers = build_entry_triggers(
        market=market,
        target_days=target_days,
        last_price=last_price,
        ma20_now=ma20_now,
        ma60_now=ma60_now,
        high_20=structure_info["high_20"],
        low_20=structure_info["low_20"],
        rsi_now=rsi_now,
        range_pos_pct=structure_info["range_pos_pct"],
        allow_short=allow_short,
    )

    exit_alert = build_exit_alert(
        chg_pct=chg_1d_pct,
        rsi_now=rsi_now,
        range_pos_pct=structure_info["range_pos_pct"],
        upper_wick_ratio=structure_info["upper_wick_ratio"],
        lower_wick_ratio=structure_info["lower_wick_ratio"],
        is_short_contract_mode=short_mode,
        take_profit=take_profit,
        last_price=last_price,
        allow_short=allow_short,
    )

    news_score_val = news_info["score"]
    high_lev_short_blocked = (direction == "偏空看跌") and (lev_val >= 10.0) and (chg_1d_pct > -0.35 or structure_info["lower_wick_ratio"] >= 0.25)
    news_aligned_bull = (blended_prob >= 0.65) and (news_score_val >= -0.10) and (prob_up >= 0.60) and (composite_score >= 0.12)
    news_aligned_bear = (
        allow_short
        and (direction == "偏空看跌")
        and (not high_lev_short_blocked)
        and (blended_prob <= 0.35)
        and (news_score_val <= 0.10)
        and (prob_up <= 0.40)
        and (composite_score <= -0.12)
    )
    high_win_qualified = (
        ((direction == "偏多看漲") or (allow_short and direction == "偏空看跌"))
        and test_acc >= 0.56
        and rr_ratio >= 1.50
        and (news_aligned_bull or news_aligned_bear)
    )
    prob_edge = abs(blended_prob - 0.50) * 100.0
    edge_score = round(prob_edge * 1.6 + max(0.0, (test_acc - 0.50) * 100.0) * 1.2 + min(rr_ratio, 3.0) * 4.0, 1)

    risk_box = None
    if capital is not None and capital > 0:
        max_risk_amt = capital * profile["max_risk_pct"]
        stop_dist = max(abs(entry_price - stop_loss), entry_price * profile["min_stop_pct"])
        stop_dist_pct = stop_dist / max(entry_price, 1e-8)
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

    payload = {
        "symbol": symbol,
        "market": market,
        "track_name": profile["track_name"],
        "target_days": days_display,
        "is_short_contract": short_mode,
        "allow_short": allow_short,
        "updated_at": taipei_now,
        "last_price": round(last_price, 4),
        "chg_1d_pct": round(chg_1d_pct, 2),
        "direction": direction,
        "action": action,
        "entry_triggers": entry_triggers,
        "exit_alert": exit_alert,
        "entry_price": entry_price,
        "rr_ratio": rr_ratio,
        "win_prob_display": win_prob_display,
        "high_win_qualified": high_win_qualified,
        "edge_score": edge_score,
        "four_dim_score": composite_score,
        "composite_score": composite_score,
        "dimensions": {
            "technical": tech_info,
            "fundamental": fund_info,
            "sentiment": sent_info,
            "macro": macro_info,
            "chip_flow": flow_info,
        },
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
    _SIGNAL_CACHE[cache_key] = {"ts": now_ts, "payload": payload}
    return payload


def run_backtest(market: str, raw_symbol: str, target_days=3, leverage=None):
    raw_days = float(target_days if target_days is not None else 3)
    target_days = max(0.05, min(raw_days, 15.0))
    short_mode = is_short_term_contract(target_days)
    profile = get_track_profile(market, target_days)
    symbol = normalize_symbol(market, raw_symbol)
    use_intraday_bars = (market == "加密貨幣" and short_mode)
    lev_val = float(leverage) if (leverage is not None and leverage > 0) else 1.0
    fee_cost = profile["fee_rate"] * 2.0
    backtest_days = 30
    allow_short = bool(profile.get("allow_short", True)) and (market != "台股")

    df = fetch_history_df(symbol, use_intraday_bars)
    if (df is None or df.empty or len(df) < 60) and use_intraday_bars:
        df = fetch_history_df(symbol, False)
        use_intraday_bars = False

    if df is None or df.empty or len(df) < 45:
        raise ValueError(f"{symbol} 歷史 K 線不足，無法執行 30 天回測。")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(df) < 45:
        raise ValueError(f"{symbol} 有效 K 線資料不足。")

    horizon_steps = max(2, int(round(target_days * 24))) if use_intraday_bars else max(1, int(round(target_days)))
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

    step_stride = (6 if (stop_idx - start_idx) > 200 else 2) if use_intraday_bars else 1
    window_start_str = str(feat_df.index[start_idx])[:16]
    window_end_str = str(feat_df.index[min(stop_idx, n_rows - 1)])[:16]

    trades = []
    signals_evaluated = 0
    retrain_every = 8 if use_intraday_bars else 5
    max_eval_steps = 80
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
            ma10_now = float(row["ma10"]) if pd.notna(row["ma10"]) else last_price
            ma20_now = float(row["ma20"]) if pd.notna(row["ma20"]) else last_price
            ma60_now = float(row["ma60"]) if pd.notna(row["ma60"]) else last_price
            lw_ratio = float(row["lower_wick_ratio"]) if pd.notna(row["lower_wick_ratio"]) else 0.0

            win_slice = feat_df.iloc[max(0, idx - 19): idx + 1]
            high_20 = float(win_slice["High"].max())
            low_20 = float(win_slice["Low"].min())
            range_pos_pct = float(np.clip((last_price - low_20) / max(high_20 - low_20, 1e-8) * 100.0, 0.0, 100.0))

            direction, _, entry_price, stop_loss, take_profit, _, _, _ = _decide_signal(
                market=market,
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
                ma10_now=ma10_now,
                ma20_now=ma20_now,
                ma60_now=ma60_now,
                lev_val=lev_val,
                is_short_contract_mode=short_mode,
                lower_wick_ratio=lw_ratio,
            )

            valid_dirs = ("偏多看漲", "偏空看跌") if allow_short else ("偏多看漲",)
            if direction not in valid_dirs:
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

    stride_note = f"（1h 每 {step_stride} 根抽樣・每 {retrain_every} 步重訓）" if use_intraday_bars else ""
    short_policy_note = "（台股僅回測偏多單，不做空）" if not allow_short else ""

    if not trades:
        return {
            "symbol": symbol,
            "market": market,
            "target_days": f"{target_days:g}",
            "backtest_days": backtest_days,
            "window_range": f"{window_start_str} ～ {window_end_str}",
            "leverage": round(lev_val, 2),
            "timeframe": f"1h K線（近30天短線合約{stride_note}）" if use_intraday_bars else f"1d 日線（近30天{short_policy_note}）",
            "bars_tested": signals_evaluated,
            "trade_count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "note": f"近 30 天（{window_start_str} ～ {window_end_str}）內符合進場條件的訊號較少{short_policy_note}。",
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
        "timeframe": f"1h K線（近30天短線合約{stride_note}）" if use_intraday_bars else f"1d 日線（近30天{short_policy_note}）",
        "bars_tested": signals_evaluated,
        "trade_count": int(len(arr)),
        "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd_pct,
        "note": f"回測區間：{window_start_str} ～ {window_end_str}（近{backtest_days}天{stride_note}，含雙邊成本 {fee_cost * 100:.2f}% 與 {lev_val:g}x 槓桿{short_policy_note}）。",
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
            "criteria": "方向機率 ≥65%（或 ≤35% 且防軋空/高槓桿保護）＋ 驗證準確 ≥56% ＋ 盈虧比 ≥1.5 ＋ 全維度與新聞不打架",
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
