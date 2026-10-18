import os
import json
import time
import hashlib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Three-MACD scanner. Independent of market_report.py.
# All cross conditions are evaluated on CLOSED candles only.

MACD_DEFAULT = (12,  26,  9)
MACD_4X      = (48, 104, 36)
MACD_QUARTER = (3,    6,  2)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "hadiseyyedi2-commits/Hadi-Market-report")
GITHUB_BRANCH = os.getenv("GITHUB_REF_NAME", "main")
STATE_PATH = "state/three_macd_sent_signals.json"

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; HadiThreeMACDScanner/1.0)"}

SYMBOLS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X", "USDCAD": "USDCAD=X", "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "EURCHF": "EURCHF=X", "EURCAD": "EURCAD=X", "EURAUD": "EURAUD=X",
    "EURNZD": "EURNZD=X", "GBPJPY": "GBPJPY=X", "GBPCHF": "GBPCHF=X",
    "GBPCAD": "GBPCAD=X", "GBPAUD": "GBPAUD=X", "GBPNZD": "GBPNZD=X",
    "AUDJPY": "AUDJPY=X", "AUDCAD": "AUDCAD=X", "AUDNZD": "AUDNZD=X",
    "NZDJPY": "NZDJPY=X", "XAUUSD": "XAUUSD=X", "XAGUSD": "XAGUSD=X",
}

# Native source intervals are aggregated into these scanner timeframes.
TIMEFRAMES = ("D1", "H4", "H1", "M30", "M15", "M5")
INTERVAL_CONFIG = {
    "1d": ("2y", 1440),
    "1h": ("730d", 60),
    "5m": ("60d", 5),
}

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def add_macd(df, prefix, params):
    fast, slow, signal = params
    line = ema(df["Close"], fast) - ema(df["Close"], slow)
    sig = ema(line, signal)
    df[prefix + "_line"] = line
    df[prefix + "_signal"] = sig
    return df

def cross_up(a_prev, a_now, b_prev, b_now):
    return a_prev <= b_prev and a_now > b_now

def cross_down(a_prev, a_now, b_prev, b_now):
    return a_prev >= b_prev and a_now < b_now

def cross_zero_up(prev, now):
    return prev <= 0 and now > 0

def cross_zero_down(prev, now):
    return prev >= 0 and now < 0

def download_yahoo(symbol, interval, period):
    url = YAHOO_URL + requests.utils.quote(symbol, safe="")
    response = requests.get(
        url,
        params={"range": period, "interval": interval, "events": "history"},
        headers=HEADERS,
        timeout=35,
    )
    response.raise_for_status()
    chart = response.json().get("chart", {})
    if chart.get("error") or not chart.get("result"):
        raise RuntimeError(str(chart.get("error") or "No Yahoo data"))
    result = chart["result"][0]
    timestamps = result.get("timestamp") or []
    quote_list = result.get("indicators", {}).get("quote", [])
    if not timestamps or not quote_list:
        raise RuntimeError("Empty quote data")
    q = quote_list[0]
    df = pd.DataFrame({
        "Timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "Open": q.get("open", []), "High": q.get("high", []),
        "Low": q.get("low", []), "Close": q.get("close", []),
    })
    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    df = df.sort_values("Timestamp").drop_duplicates("Timestamp", keep="last")
    df = df.reset_index(drop=True)
    now = pd.Timestamp.now(tz="UTC")
    minutes = INTERVAL_CONFIG[interval][1]
    df = df.loc[df["Timestamp"] + pd.to_timedelta(minutes, unit="m") <= now]
    return df.reset_index(drop=True)

def resample_ohlc(df, rule, minutes):
    if df is None or df.empty:
        return None
    indexed = df.copy().set_index("Timestamp")
    grouped = indexed.resample(rule, origin="epoch", label="left", closed="left")
    out = grouped.agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last"
    })
    counts = grouped["Close"].count()
    expected = minutes // int((df["Timestamp"].diff().dropna().dt.total_seconds().median() or 60) / 60)
    # Keep only complete bars; incomplete current bars are excluded separately.
    out["__count"] = counts
    out = out.loc[out["__count"] >= expected].drop(columns="__count").dropna()
    out = out.reset_index()
    now = pd.Timestamp.now(tz="UTC")
    out = out.loc[out["Timestamp"] + pd.to_timedelta(minutes, unit="m") <= now]
    return out.reset_index(drop=True)

def build_timeframes(symbol):
    frames = {}
    frames["D1"] = download_yahoo(symbol, "1d", "2y")
    time.sleep(0.12)
    h1 = download_yahoo(symbol, "1h", "730d")
    frames["H1"] = h1
    frames["H4"] = resample_ohlc(h1, "4h", 240)
    time.sleep(0.12)
    m5 = download_yahoo(symbol, "5m", "60d")
    frames["M5"] = m5
    frames["M15"] = resample_ohlc(m5, "15min", 15)
    frames["M30"] = resample_ohlc(m5, "30min", 30)
    return frames

def enrich(df):
    if df is None or len(df) < 150:
        return None
    data = df.copy().reset_index(drop=True)
    data = add_macd(data, "d", MACD_DEFAULT)
    data = add_macd(data, "x", MACD_4X)
    data = add_macd(data, "q", MACD_QUARTER)
    return data.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

def scan_direction(data, direction, symbol_name, timeframe):
    """
    Ordered, closed-candle state machine.

    BUY:
      4X crosses above zero; build HIGH RANGE until its first bearish
      signal-line cross above zero (range excludes that cross candle).
      Default MACD is checked only at the quarter-MACD trigger: it must be above zero for BUY.
      Quarter MACD's downward zero-cross triggers only when 4X is above zero and its signal, and default MACD is above zero. No default-MACD cross is required before the trigger.
      Any HIGH RANGE touch after the first 4X bearish cross invalidates.

    SELL is the exact mirror.
    """
    if data is None or len(data) < 10:
        return None

    d = data.reset_index(drop=True)
    state = 0
    range_price = None
    range_time = None
    default_crossed_first = False
    default_crossed_back = False
    last_signal = None

    for i in range(1, len(d)):
        p, c = d.iloc[i - 1], d.iloc[i]
        x0, x1 = float(p["x_line"]), float(c["x_line"])
        xs0, xs1 = float(p["x_signal"]), float(c["x_signal"])
        d0, d1 = float(p["d_line"]), float(c["d_line"])
        q0, q1 = float(p["q_line"]), float(c["q_line"])
        t = c["Timestamp"]

        if direction == "BUY":
            signal_cross = cross_down(x0, x1, xs0, xs1)
            initial_cross = signal_cross and x1 > 0
            second_cross = cross_up(x0, x1, xs0, xs1) and x1 > 0
            default_first_cross = cross_zero_down(d0, d1)
            default_return_cross = cross_zero_up(d0, d1)
            quarter_trigger = cross_zero_down(q0, q1)
            touched_range = state >= 2 and float(c["High"]) >= range_price

            if state == 0:
                if cross_zero_up(x0, x1):
                    state = 1
                    range_price = float(c["High"])
                    range_time = t
                    default_crossed_first = default_crossed_back = False
                continue

            if state == 1:
                # A bearish 4X signal-line cross is valid only while 4X stays
                # above zero. If it is at/below zero on that candle, invalidate
                # this setup instead of continuing to search for a later cross.
                if signal_cross and x1 <= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                    continue
                # The range ends immediately before the first bearish 4X
                # signal-line-cross candle; that candle is not part of range.
                if initial_cross:
                    # HIGH RANGE uses the 100 candles immediately before the
                    # signal-line-cross candle; the crossover candle is excluded.
                    start = max(0, i - 100)
                    window = d.iloc[start:i]
                    if not window.empty:
                        idx = window["High"].astype(float).idxmax()
                        range_time = d.loc[idx, "Timestamp"]
                        range_price = float(d.loc[idx, "High"])
                    state = 2
                    default_crossed_first = default_first_cross
                else:
                    if default_first_cross:
                        default_crossed_first = True
                continue

            # Range touch from the candle after the initial 4X signal cross
            # through the trigger candle invalidates the whole setup.
            if touched_range:
                state = 0
                range_price = range_time = None
                default_crossed_first = default_crossed_back = False
                continue

            if state == 2:
                # Default MACD is intentionally NOT checked in intermediate
                # stages. Its only BUY condition is d1 > 0 on the trigger candle.
                if second_cross:
                    state = 3
                elif x1 <= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                continue

            if state == 3:
                if x1 <= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                    continue
                if x1 > xs1 and x1 > 0 and d1 > 0 and quarter_trigger:
                    last_signal = {
                        "symbol": symbol_name, "timeframe": timeframe, "direction": "BUY",
                        "time": t, "price": float(c["Close"]),
                        "range_name": "HIGH RANGE", "range_price": float(range_price),
                        "range_time": range_time,
                        "id": signal_id(symbol_name, timeframe, "BUY", t, range_price)
                    }
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False

        else:  # SELL: exact mirror
            signal_cross = cross_up(x0, x1, xs0, xs1)
            initial_cross = signal_cross and x1 < 0
            second_cross = cross_down(x0, x1, xs0, xs1) and x1 < 0
            default_first_cross = cross_zero_up(d0, d1)
            default_return_cross = cross_zero_down(d0, d1)
            quarter_trigger = cross_zero_up(q0, q1)
            touched_range = state >= 2 and float(c["Low"]) <= range_price

            if state == 0:
                if cross_zero_down(x0, x1):
                    state = 1
                    range_price = float(c["Low"])
                    range_time = t
                    default_crossed_first = default_crossed_back = False
                continue

            if state == 1:
                # Mirror condition: the first bullish 4X signal-line cross for
                # SELL is valid only while 4X remains below zero.
                if signal_cross and x1 >= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                    continue
                if initial_cross:
                    # LOW RANGE uses the 100 candles immediately before the
                    # signal-line-cross candle; the crossover candle is excluded.
                    start = max(0, i - 100)
                    window = d.iloc[start:i]
                    if not window.empty:
                        idx = window["Low"].astype(float).idxmin()
                        range_time = d.loc[idx, "Timestamp"]
                        range_price = float(d.loc[idx, "Low"])
                    state = 2
                    default_crossed_first = default_first_cross
                else:
                    if default_first_cross:
                        default_crossed_first = True
                continue

            if touched_range:
                state = 0
                range_price = range_time = None
                default_crossed_first = default_crossed_back = False
                continue

            if state == 2:
                # Default MACD is intentionally NOT checked in intermediate
                # stages. Its only SELL condition is d1 < 0 on the trigger candle.
                if second_cross:
                    state = 3
                elif x1 >= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                continue

            if state == 3:
                if x1 >= 0:
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False
                    continue
                if x1 < xs1 and x1 < 0 and d1 < 0 and quarter_trigger:
                    last_signal = {
                        "symbol": symbol_name, "timeframe": timeframe, "direction": "SELL",
                        "time": t, "price": float(c["Close"]),
                        "range_name": "LOW RANGE", "range_price": float(range_price),
                        "range_time": range_time,
                        "id": signal_id(symbol_name, timeframe, "SELL", t, range_price)
                    }
                    state = 0
                    range_price = range_time = None
                    default_crossed_first = default_crossed_back = False

    return last_signal

def signal_id(symbol, timeframe, direction, trigger_time, range_price):
    raw = f"{symbol}|{timeframe}|{direction}|{trigger_time}|{range_price:.10f}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def scan_timeframe(df, symbol, timeframe):
    data = enrich(df)
    if data is None:
        return []
    results = []
    for direction in ("BUY", "SELL"):
        signal = scan_direction(data, direction, symbol, timeframe)
        if signal:
            results.append(signal)
    return results

def format_signal(s):
    # Convert UTC timestamps to Tehran local time (Iran Standard Time, UTC+03:30).
    tehran_tz = "Asia/Tehran"
    trigger_time = pd.Timestamp(s["time"])
    range_time = pd.Timestamp(s["range_time"])
    if trigger_time.tzinfo is None:
        trigger_time = trigger_time.tz_localize("UTC")
    if range_time.tzinfo is None:
        range_time = range_time.tz_localize("UTC")
    trigger_local = trigger_time.tz_convert(tehran_tz)
    range_local = range_time.tz_convert(tehran_tz)
    return "\n".join([
        "★★ THREE MACD SCANNER ★★",
        f"{'🟢 BUY' if s['direction'] == 'BUY' else '🔻 SELL'} | {s['symbol']}",
        f"Timeframe: {s['timeframe']}",
        f"Trigger time (Tehran): {trigger_local.strftime('%Y-%m-%d %H:%M')}",
        f"Trigger price: {s['price']:.6f}",
        f"{s['range_name']}: {s['range_price']:.6f}",
        f"Range time (Tehran): {range_local.strftime('%Y-%m-%d %H:%M')}",
    ])

def github_headers():
    return {
        "Authorization": "Bearer " + GITHUB_TOKEN,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

def load_state():
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        return {}
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{STATE_PATH}"
    try:
        r = requests.get(url, headers=github_headers(), params={"ref": GITHUB_BRANCH}, timeout=20)
        if r.status_code == 200:
            import base64
            return json.loads(base64.b64decode(r.json()["content"]).decode("utf-8"))
        if r.status_code != 404:
            print("State read failed:", r.status_code, r.text[:200])
    except Exception as exc:
        print("State read exception:", exc)
    return {}

def save_state(state):
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("WARNING: GITHUB_TOKEN or GITHUB_REPOSITORY missing; sent IDs are not persisted.")
        return False
    import base64
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents/{STATE_PATH}"
    headers = github_headers()
    try:
        existing = requests.get(url, headers=headers, params={"ref": GITHUB_BRANCH}, timeout=20)
        payload = {
            "message": "Update three-MACD scanner signal state",
            "content": base64.b64encode(json.dumps(state, ensure_ascii=False, indent=2).encode()).decode(),
            "branch": GITHUB_BRANCH,
        }
        if existing.status_code == 200:
            payload["sha"] = existing.json()["sha"]
        elif existing.status_code != 404:
            print("State SHA lookup failed:", existing.status_code, existing.text[:200])
            return False
        r = requests.put(url, headers=headers, json=payload, timeout=25)
        if r.status_code in (200, 201):
            return True
        print("State write failed:", r.status_code, r.text[:300])
    except Exception as exc:
        print("State write exception:", exc)
    return False

def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing; report printed to logs only.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    # Telegram message limit is 4096 characters; send each signal separately.
    for chunk_start in range(0, len(message), 3900):
        r = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message[chunk_start:chunk_start + 3900],
        }, timeout=25)
        if r.status_code != 200 or not r.json().get("ok", False):
            print("Telegram send failed:", r.status_code, r.text[:300])
            return False
    return True

def main():
    print("=" * 64)
    print("HADI THREE-MACD SCANNER | closed candles only")
    print("Default MACD:", MACD_DEFAULT, "| 4X:", MACD_4X, "| Quarter:", MACD_QUARTER)
    print("Symbols:", len(SYMBOLS), "| Timeframes:", ", ".join(TIMEFRAMES))
    print("=" * 64)

    sent_state = load_state()
    found = []
    for name, yahoo_symbol in SYMBOLS.items():
        print("Scanning symbol:", name)
        try:
            frames = build_timeframes(yahoo_symbol)
        except Exception as exc:
            print("Symbol data failed:", name, str(exc)[:180])
            continue

        for timeframe in TIMEFRAMES:
            df = frames.get(timeframe)
            if df is None or len(df) < 150:
                print("Insufficient data:", name, timeframe, 0 if df is None else len(df))
                continue
            try:
                signals = scan_timeframe(df, name, timeframe)
                for s in signals:
                    print("FOUND:", s["direction"], name, timeframe, s["time"], s["price"])
                    found.append(s)
            except Exception as exc:
                print("Scan failed:", name, timeframe, str(exc)[:180])
        time.sleep(0.15)

    # Choose the newest unsent valid signal per symbol + timeframe.
    # A previously sent latest candidate must not hide another unsent candidate.
    latest_unsent = {}
    for s in found:
        if s["id"] in sent_state:
            continue
        key = f'{s["symbol"]}|{s["timeframe"]}'
        if key not in latest_unsent or pd.Timestamp(s["time"]) > pd.Timestamp(latest_unsent[key]["time"]):
            latest_unsent[key] = s

    new_signals = list(latest_unsent.values())
    if not new_signals:
        print("No new signals.")
        return

    new_signals.sort(key=lambda s: (s["symbol"], s["timeframe"], s["time"]))
    report = "\n\n--------------------\n\n".join(format_signal(s) for s in new_signals)
    print(report)
    if not send_telegram(report):
        print("Alerts not marked as sent because Telegram delivery failed.")
        return

    now = datetime.now(timezone.utc).isoformat()
    for s in new_signals:
        sent_state[s["id"]] = {
            "symbol": s["symbol"], "timeframe": s["timeframe"],
            "direction": s["direction"], "trigger_time": str(s["time"]),
            "trigger_price": s["price"], "sent_at": now,
        }
    # Bound state file growth.
    if len(sent_state) > 1500:
        sent_state = dict(list(sent_state.items())[-1000:])
    if not save_state(sent_state):
        print("WARNING: alert sent but state persistence failed; duplicate alerts may occur.")
    print("Sent", len(new_signals), "signal(s).")

if __name__ == "__main__":
    main()
