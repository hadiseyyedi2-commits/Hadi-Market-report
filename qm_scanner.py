#!/usr/bin/env python3
"""
Hadi Seyyedi - QM Scanner (Python / GitHub Actions)
v1.05 - QM pattern + Neckline Retest + Stochastic(60/15, slowing=1) confirmation

Signal logic:
  1) Detect QM triangle from MACD phase divergence
  2) Wait for neckline breakout
  3) Wait for neckline retest (price touches break_level)
  4) At retest: check Stochastic %K, periods 60 and 15, slowing=1
       BUY : Stoch(60) > 50  AND  Stoch(15) < 50
       SELL: Stoch(60) < 50  AND  Stoch(15) > 50
  5) If Stoch fails -> wait for next retest (2nd, 3rd, 4th ...)
  6) Alert on first passing retest (freshness <= FRESH_BARS).
"""
import os
import sys
import json
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =====================================================================
# CONFIG
# =====================================================================
SYMBOLS = [
    "EURUSD", "EURAUD", "EURJPY", "EURGBP", "EURCAD",
    "GBPUSD", "GBPAUD", "GBPJPY", "GBPCAD",
    "AUDCAD", "AUDUSD", "AUDJPY", "AUDNZD", "AUDCHF",
    "XAUUSD", "XAGUSD",
    "NZDUSD", "NZDJPY",
    "USDJPY", "USDCAD", "USDCHF", "CHFJPY",
]

TIMEFRAMES = ["M15", "H1", "H4", "D1"]

# MACD
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9

# Stochastic retest filter
USE_STOCH_FILTER     = True
STOCH_LONG_PERIOD    = 60
STOCH_SHORT_PERIOD   = 15
STOCH_D_PERIOD       = 3
STOCH_SLOWING        = 1
STOCH_LONG_LEVEL     = 50.0
STOCH_SHORT_LEVEL    = 50.0

# Scan window
HISTORY_BARS = 500
FRESH_BARS   = 20

# First run: prime without sending (only affects the very first run after
# state["initialized"] becomes True)
STARTUP_ALERT = False

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

STATE_FILE = "state/qm_sent_signals.json"
MAX_ALERTED_KEYS = 5000

SYMBOL_MAP = {
    "EURUSD": "EURUSD=X", "EURAUD": "EURAUD=X", "EURJPY": "EURJPY=X",
    "EURGBP": "EURGBP=X", "EURCAD": "EURCAD=X",
    "GBPUSD": "GBPUSD=X", "GBPAUD": "GBPAUD=X", "GBPJPY": "GBPJPY=X",
    "GBPCAD": "GBPCAD=X",
    "AUDCAD": "AUDCAD=X", "AUDUSD": "AUDUSD=X", "AUDJPY": "AUDJPY=X",
    "AUDNZD": "AUDNZD=X", "AUDCHF": "AUDCHF=X",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    "NZDUSD": "NZDUSD=X", "NZDJPY": "NZDJPY=X",
    "USDJPY": "USDJPY=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    "CHFJPY": "CHFJPY=X",
}

TF_CONFIG = {
    "M15": ("15m", "60d"),
    "M30": ("30m", "60d"),
    "H1":  ("60m", "730d"),
    "H4":  "from_H1",
    "D1":  ("1d",  "5y"),
}

# =====================================================================
# INDICATORS
# =====================================================================
def calc_macd(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    return ema_fast - ema_slow


def calc_stoch_main(high: pd.Series, low: pd.Series, close: pd.Series,
                    k_period: int, d_period: int = 3, slowing: int = 1) -> pd.Series:
    """
    Equivalent of MT5 iStochastic buffer 0 (%K main line).
      rawK = 100 * (Close - LL(k)) / (HH(k) - LL(k))
      main = SMA(rawK, slowing)   (buffer 0 — used here)
      sig  = SMA(main, d_period)  (buffer 1 — NOT used)
    slowing=1 -> main = rawK (no smoothing)
    """
    lowest_low   = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    raw_k = 100.0 * (close - lowest_low) / denom
    if slowing <= 1:
        return raw_k
    return raw_k.rolling(window=slowing, min_periods=slowing).mean()


# =====================================================================
# QM PATTERN DETECTION
# =====================================================================
def build_phases(df: pd.DataFrame, macd: pd.Series) -> list:
    phases = []
    n = len(df)
    if n < 10:
        return phases

    current_sign = 0
    cur = None
    for i in range(n):
        m = macd.iloc[i]
        if pd.isna(m):
            continue
        sign = 1 if m > 0 else (-1 if m < 0 else 0)

        if sign == 0:
            if cur is not None:
                cur["last_idx"] = i - 1
                cur["last_time"] = df.index[i - 1]
                phases.append(cur)
                cur = None
                current_sign = 0
            continue

        if cur is None or sign != current_sign:
            if cur is not None:
                cur["last_idx"] = i - 1
                cur["last_time"] = df.index[i - 1]
                phases.append(cur)
            cur = {
                "sign": sign,
                "first_idx": i,
                "first_time": df.index[i],
                "high": float(df["High"].iloc[i]),
                "low":  float(df["Low"].iloc[i]),
                "high_time": df.index[i],
                "low_time":  df.index[i],
                "macd_extreme": float(m),
            }
            current_sign = sign
        else:
            hi = float(df["High"].iloc[i])
            lo = float(df["Low"].iloc[i])
            if hi > cur["high"]:
                cur["high"] = hi
                cur["high_time"] = df.index[i]
            if lo < cur["low"]:
                cur["low"] = lo
                cur["low_time"] = df.index[i]
            if sign > 0 and m > cur["macd_extreme"]:
                cur["macd_extreme"] = float(m)
            elif sign < 0 and m < cur["macd_extreme"]:
                cur["macd_extreme"] = float(m)

    if cur is not None:
        cur["last_idx"] = n - 1
        cur["last_time"] = df.index[n - 1]
        phases.append(cur)

    return phases


def scan_triangles(phases: list) -> list:
    triangles = []
    n = len(phases)
    for i in range(n - 3):
        # Bullish QM: + - + -
        if (phases[i]["sign"] > 0 and phases[i+1]["sign"] < 0
                and phases[i+2]["sign"] > 0 and phases[i+3]["sign"] < 0):
            p1, n1, p2, n2 = phases[i], phases[i+1], phases[i+2], phases[i+3]
            regular_bull = (n2["low"] < n1["low"]
                            and n2["macd_extreme"] > n1["macd_extreme"])
            opp_rel = (p2["high"] < p1["high"])
            if regular_bull and opp_rel:
                triangles.append({
                    "bullish": True,
                    "structure_time": p1["first_time"],
                    "confirm_time":   n2["last_time"],
                    "p1_extreme":  n1["low"],
                    "p2_extreme":  n2["low"],
                    "macd1": n1["macd_extreme"],
                    "macd2": n2["macd_extreme"],
                    "break_level": p2["high"],
                })
        # Bearish QM: - + - +
        if (phases[i]["sign"] < 0 and phases[i+1]["sign"] > 0
                and phases[i+2]["sign"] < 0 and phases[i+3]["sign"] > 0):
            n1, p1, n2, p2 = phases[i], phases[i+1], phases[i+2], phases[i+3]
            regular_bear = (p2["high"] > p1["high"]
                            and p2["macd_extreme"] < p1["macd_extreme"])
            opp_rel = (n2["low"] > n1["low"])
            if regular_bear and opp_rel:
                triangles.append({
                    "bullish": False,
                    "structure_time": n1["first_time"],
                    "confirm_time":   p2["last_time"],
                    "p1_extreme":  p1["high"],
                    "p2_extreme":  p2["high"],
                    "macd1": p1["macd_extreme"],
                    "macd2": p2["macd_extreme"],
                    "break_level": n2["low"],
                })
    return triangles


# =====================================================================
# STATE MACHINE
# =====================================================================
def get_state(triangle: dict, df: pd.DataFrame,
              stoch_long: pd.Series, stoch_short: pd.Series):
    """
    0 = waiting breakout
    1 = broken, waiting retest
    2 = retest active (checking stochastic)
    3 = SIGNAL
   -1 = invalidated
    """
    n = len(df)
    if n < 5:
        return 0, -1, None

    confirm_time = triangle["confirm_time"]
    try:
        loc = df.index.get_loc(confirm_time)
    except (KeyError, TypeError):
        return 0, -1, None
    if not isinstance(loc, (int, np.integer)):
        try:
            loc = int(np.asarray(loc).ravel()[0])
        except Exception:
            return 0, -1, None
    confirm_idx = int(loc)
    if confirm_idx < 1:
        return 0, -1, None

    state = 0
    state_idx = confirm_idx
    bullish = triangle["bullish"]
    break_level = triangle["break_level"]
    p2_extreme = triangle["p2_extreme"]

    for i in range(confirm_idx + 1, n - 1):
        c = float(df["Close"].iloc[i])
        h = float(df["High"].iloc[i])
        l = float(df["Low"].iloc[i])
        t = df.index[i]

        if bullish and l < p2_extreme:
            return -1, i, None
        if (not bullish) and h > p2_extreme:
            return -1, i, None

        if state == 0:
            if bullish and c > break_level:
                state = 1; state_idx = i
            elif (not bullish) and c < break_level:
                state = 1; state_idx = i
            continue

        if state == 1:
            touch = (l <= break_level) if bullish else (h >= break_level)
            if not touch:
                continue
            state = 2; state_idx = i

        if state == 2:
            stoch_ok = True
            if USE_STOCH_FILTER:
                sl = stoch_long.iloc[i]
                ss = stoch_short.iloc[i]
                if pd.isna(sl) or pd.isna(ss):
                    continue
                if bullish:
                    stoch_ok = (sl > STOCH_LONG_LEVEL and ss < STOCH_SHORT_LEVEL)
                else:
                    stoch_ok = (sl < STOCH_LONG_LEVEL and ss > STOCH_SHORT_LEVEL)

            if stoch_ok:
                return 3, i, t

            retest_done = (c > break_level) if bullish else (c < break_level)
            if retest_done:
                state = 1; state_idx = i

    return state, state_idx, None


# =====================================================================
# DATA FETCHING
# =====================================================================
def _download(ticker: str, interval: str, period: str):
    try:
        df = yf.download(
            ticker, interval=interval, period=period,
            progress=False, auto_adjust=False, threads=False,
        )
    except Exception as e:
        print(f"    fetch error {ticker} {interval}: {e}")
        return None

    if df is None or df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    for col in ("Open", "High", "Low", "Close"):
        if col not in df.columns:
            return None

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_data(ticker: str, tf_name: str):
    cfg = TF_CONFIG.get(tf_name)
    if cfg is None:
        return None
    if cfg == "from_H1":
        h1 = _download(ticker, "60m", "730d")
        if h1 is None or len(h1) < 40:
            return None
        try:
            df = h1.resample("4H").agg({
                "Open": "first", "High": "max",
                "Low": "min", "Close": "last", "Volume": "sum",
            }).dropna()
        except Exception as e:
            print(f"    resample error {ticker}: {e}")
            return None
        return df
    interval, period = cfg
    return _download(ticker, interval, period)


# =====================================================================
# SCANNING
# =====================================================================
def alert_key(symbol: str, tf: str, structure_time, bullish: bool) -> str:
    ts = int(pd.Timestamp(structure_time).value // 1_000_000_000)
    return f"QM|{symbol}|{tf}|{ts}|{'BUY' if bullish else 'SELL'}"


def scan_all() -> list:
    results = []
    for sym in SYMBOLS:
        ticker = SYMBOL_MAP.get(sym)
        if not ticker:
            continue
        for tf in TIMEFRAMES:
            try:
                df = fetch_data(ticker, tf)
                if df is None or len(df) < 100:
                    continue

                macd_full     = calc_macd(df["Close"])
                stoch_l_full  = calc_stoch_main(
                    df["High"], df["Low"], df["Close"],
                    STOCH_LONG_PERIOD, STOCH_D_PERIOD, STOCH_SLOWING)
                stoch_s_full  = calc_stoch_main(
                    df["High"], df["Low"], df["Close"],
                    STOCH_SHORT_PERIOD, STOCH_D_PERIOD, STOCH_SLOWING)

                if len(df) > HISTORY_BARS:
                    df            = df.iloc[-HISTORY_BARS:]
                    macd_full     = macd_full.iloc[-HISTORY_BARS:]
                    stoch_l_full  = stoch_l_full.iloc[-HISTORY_BARS:]
                    stoch_s_full  = stoch_s_full.iloc[-HISTORY_BARS:]

                phases    = build_phases(df, macd_full)
                triangles = scan_triangles(phases)

                for tri in triangles:
                    st, st_idx, at = get_state(tri, df, stoch_l_full, stoch_s_full)
                    if st != 3 or at is None:
                        continue

                    offset = (len(df) - 1) - st_idx
                    if offset > FRESH_BARS:
                        continue

                    results.append({
                        "key":         alert_key(sym, tf, tri["structure_time"], tri["bullish"]),
                        "symbol":      sym,
                        "tf":          tf,
                        "bullish":     tri["bullish"],
                        "break_level": tri["break_level"],
                        "alert_time":  at,
                        "offset":      offset,
                        "stoch_long":  float(stoch_l_full.iloc[st_idx]),
                        "stoch_short": float(stoch_s_full.iloc[st_idx]),
                    })
            except Exception:
                print(f"  [ERROR] {sym} {tf}:")
                traceback.print_exc()
            time.sleep(0.4)
    return results


# =====================================================================
# TELEGRAM
# =====================================================================
def send_telegram(text: str) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }, timeout=20)
        if not r.ok:
            print(f"    telegram failed: {r.status_code} {r.text[:200]}")
            return False
        return True
    except Exception as e:
        print(f"    telegram error: {e}")
        return False


def format_message(sig: dict) -> str:
    emoji = "🟢" if sig["bullish"] else "🔴"
    direction = "BUY" if sig["bullish"] else "SELL"
    qm_type = "Bullish QM" if sig["bullish"] else "Bearish QM"
    t = pd.Timestamp(sig["alert_time"])
    return (
        f"{emoji} <b>{direction}</b> · QM Retest\n"
        f"<b>{sig['symbol']}</b> · {sig['tf']}\n"
        f"{qm_type}\n"
        f"Break: <code>{sig['break_level']:.5f}</code>\n"
        f"Stoch({STOCH_LONG_PERIOD}): {sig['stoch_long']:.1f} | "
        f"Stoch({STOCH_SHORT_PERIOD}): {sig['stoch_short']:.1f}\n"
        f"Signal time: {t.strftime('%Y-%m-%d %H:%M')} UTC\n"
        f"Fresh: {sig['offset']} bar(s) ago"
    )


# =====================================================================
# STATE PERSISTENCE
# =====================================================================
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
                s.setdefault("alerted", [])
                s.setdefault("initialized", False)
                return s
        except Exception:
            pass
    return {"alerted": [], "initialized": False}


def save_state(state: dict):
    d = os.path.dirname(STATE_FILE)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# =====================================================================
# MAIN  —  FIXED LOGIC (v1.05)
# =====================================================================
def main() -> int:
    print(f"QM Scanner start: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        return 1

    state = load_state()
    is_initialized = bool(state.get("initialized", False))
    already = set(state["alerted"])

    # Only the FIRST run (before initialized=True) primes without sending.
    # After that, new signals are always sent.
    is_first_run = not is_initialized
    should_send_on_first_run = bool(STARTUP_ALERT)  # usually False

    print(f"State initialized={is_initialized} | "
          f"first_run={is_first_run} | start_alert={should_send_on_first_run} | "
          f"known_signals={len(already)}")

    print(f"Scanning {len(SYMBOLS)} symbols x {len(TIMEFRAMES)} timeframes...")
    signals = scan_all()
    print(f"Detected {len(signals)} active signal(s)")

    sent = 0
    primed = 0
    for sig in signals:
        if sig["key"] in already:
            continue

        if is_first_run and not should_send_on_first_run:
            tag = "prime"
        else:
            tag = "send"

        if tag == "send":
            ok = send_telegram(format_message(sig))
            if ok:
                state["alerted"].append(sig["key"])
                already.add(sig["key"])
                sent += 1
                print(f"  [SENT] {sig['symbol']:7s} {sig['tf']:3s} "
                      f"{'BUY' if sig['bullish'] else 'SELL':4s} "
                      f"break={sig['break_level']:.5f} "
                      f"Stoch60={sig['stoch_long']:.1f} Stoch15={sig['stoch_short']:.1f}")
            else:
                print(f"  [FAIL] {sig['symbol']} {sig['tf']} (telegram error)")
        else:
            state["alerted"].append(sig["key"])
            already.add(sig["key"])
            primed += 1
            print(f"  [PRIME] {sig['symbol']} {sig['tf']} "
                  f"{'BUY' if sig['bullish'] else 'SELL'}")

    state["initialized"] = True
    if len(state["alerted"]) > MAX_ALERTED_KEYS:
        state["alerted"] = state["alerted"][-MAX_ALERTED_KEYS:]
    save_state(state)

    print(f"Done. sent={sent}, primed={primed}, "
          f"total_alerted={len(state['alerted'])}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
